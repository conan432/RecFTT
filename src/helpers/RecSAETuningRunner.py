import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from utils import utils
from models.BaseModel import BaseModel
import numpy as np
from scipy import stats
import json
from models.sae.sae import SAE
from typing import Dict, List
import pandas as pd
import wandb
import gc
import torch.nn.functional as F

from helpers.BaseRunner import BaseRunner

class RecSAETuningRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        # 你的TuningRunner需要的所有参数
        parser = BaseRunner.parse_runner_args(parser)
        parser.add_argument('--sae_lr', type=float, default=1e-4, help='SAE and Controller Learning rate.')
        parser.add_argument('--sae_batch_size', type=int, default=32, help='SAE and Controller training batch size.')
        parser.add_argument('--eval_dims', type=str, default='',
                            help='Comma-separated dimensions to evaluate on (e.g., "gluten_free").')
        parser.add_argument('--sae_train', type=int, default=0,
							help='train sae or evaluate RecSAE')
        parser.add_argument('--sae_baseline', type=int, default=0,
						help='To save baseline result or not')
        parser.add_argument('--result_data_path', type=str, default="",
                            help='base path to save prediction list and RecSAE activations')
        parser.add_argument('--probe_position', type=str, default="",
                            help='default, emb')
        parser.add_argument('--mlp_lr', type=float, default=1e-4, help='MLP Controller specific learning rate.')
        return parser

    def __init__(self, args):
        super().__init__(args)
        self.sae_lr = args.sae_lr
        self.sae_k = args.sae_k
        self.sae_scale_size = args.sae_scale_size
        self.mlp_lr = args.mlp_lr
        self.sae_batch_size = args.sae_batch_size
        self.eval_dims = [dim.strip() for dim in args.eval_dims.split(',') if dim.strip()]
        self.item_metadata = None 
        self.recsae_model_path = args.recsae_model_path
        self.result_data_path = args.result_data_path
        self.device = args.device
        self.main_metric = args.main_metric     
        if any(dim in self.main_metric for dim in self.eval_dims):
            logging.info(f"Using dimension-specific main metric: {self.main_metric}")
        else: 
            self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])

    def _build_optimizer(self, model):
        params_to_train = []
        logging.info("Building optimizer for parameters with requires_grad=True:")

        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_train.append(param)
                logging.info(f"\t- Including parameter: {name}")

        if not params_to_train:
            raise ValueError("Optimizer creation failed: No parameters found with requires_grad=True. "
                            "Check if you have frozen all layers by mistake.")

        optimizer = torch.optim.Adam(params_to_train, lr=self.mlp_lr, weight_decay=self.l2)
        logging.info("Optimizer successfully created.")
        return optimizer

    def calculate_composite_loss(self, model, out_dict):
        final_user_vector = out_dict['tuned_components']['final_user_vector']
        tuned_components = out_dict['tuned_components']
        candidate_items = out_dict['candidate_items']
        
        candidate_vectors = out_dict['i_v'] 
        predictions_tuned = (final_user_vector.unsqueeze(1) * candidate_vectors).sum(dim=-1)
        
        pos_scores = predictions_tuned[:, 0]
        neg_scores = predictions_tuned[:, 1:]
        
        neg_softmax = (neg_scores - neg_scores.max(dim=1, keepdim=True).values).softmax(dim=1)
       
        loss_per_sample = -(((pos_scores[:, None] - neg_scores).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log()
        
        L_ranking_base = loss_per_sample.mean()
        
        L_tuning = torch.tensor(0.0, device=model.device)
        sae_module = model.sae_module
        
        if sae_module.loss_lambda > 0 and len(sae_module.tuning_item_set) > 0:
            final_user_vector = tuned_components['final_user_vector']
            all_candidate_vectors = out_dict['i_v'] 

            all_scores_tuned = (final_user_vector.unsqueeze(1) * all_candidate_vectors).sum(dim=-1)
            is_target_item_mask = torch.tensor(
                np.isin(candidate_items.cpu().numpy(), list(sae_module.tuning_item_set)),
                device=model.device, dtype=torch.bool
            )

            is_positive_mask = torch.zeros_like(is_target_item_mask, dtype=torch.bool)
            is_positive_mask[:, 0] = True

            target_pos_mask = is_positive_mask & is_target_item_mask
            nontarget_neg_mask = (~is_positive_mask) & (~is_target_item_mask)

            batch_losses = []
            for i in range(all_scores_tuned.shape[0]):
                if target_pos_mask[i, 0]:
                    target_pos_score = all_scores_tuned[i, 0]
                    
                    user_nontarget_neg_scores = all_scores_tuned[i][nontarget_neg_mask[i]]
                    
                    if user_nontarget_neg_scores.numel() > 0:
                        neg_score_sample = user_nontarget_neg_scores[torch.randint(0, user_nontarget_neg_scores.numel(), (1,))]
                        
                        diff = target_pos_score - neg_score_sample
                        batch_losses.append(-torch.log(torch.sigmoid(diff) + 1e-8))

            if batch_losses:
                L_tuning = torch.stack(batch_losses).mean()

        L_total = L_ranking_base + sae_module.loss_lambda * L_tuning
        
        return L_total, L_ranking_base, L_tuning

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        
        dataset.actions_before_epoch()
        model.sae_module.train() 
        model.own_part.eval() 

        total_backward_time = 0.0
        loss_lst, recon_loss_lst, tuning_loss_lst = [], [], []
        dl = DataLoader(dataset, batch_size=self.sae_batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)

        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            
            out_dict = model(batch)
            
            loss, recon_loss, tuning_loss = self.calculate_composite_loss(model, out_dict)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Epoch {epoch}: NaN or Inf loss detected! Skipping update.")
                continue
            
            backward_start_time = time()
            loss.backward()
            model.optimizer.step()
            total_backward_time += time() - backward_start_time
            
            loss_lst.append(loss.item())
            recon_loss_lst.append(recon_loss.item())
            if isinstance(tuning_loss, torch.Tensor):
                tuning_loss_lst.append(tuning_loss.item())
        
        avg_losses = {
            'Total Loss': np.mean(loss_lst),
            'Recon Loss': np.mean(recon_loss_lst),
            'Tuning Loss': np.mean(tuning_loss_lst) if tuning_loss_lst else 0.0,
            'Backward Time': total_backward_time 
        }
        logging.info(f"Epoch {epoch} Averages: Total Loss={avg_losses['Total Loss']:.4f}, "
                     f"Recon Loss={avg_losses['Recon Loss']:.4f}, Tuning Loss={avg_losses['Tuning Loss']:.4f}")
        
        return avg_losses

    def _initialize_metadata(self, corpus):
        if self.item_metadata is None:
            logging.info("Initializing item metadata for evaluation...")
            self.item_metadata = corpus.item_metadata

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        model = data_dict['train'].model
        corpus = data_dict['train'].corpus
        
        if self.item_metadata is None:
            self.item_metadata = corpus.item_metadata
        if model.sae_module.tuning_dims:
            merged_item_set = set()
            logging.info(f"Merging item sets for tuning dimensions: {model.sae_module.tuning_dims}")
            tuning_dims = [dim.strip() for dim in model.sae_module.tuning_dims.split(',') if dim.strip()]
            for dim in tuning_dims:
                item_set = corpus.item_metadata.get(dim, set())
                if item_set:
                    merged_item_set.update(item_set)
                else:
                    logging.warning(f"Item set for dimension '{dim}' is empty.")
            print(len(merged_item_set))
            model.sae_module.set_tuning_set(merged_item_set)

        for param in model.own_part.parameters():
            param.requires_grad = False
        logging.info("SASRec base model parameters have been frozen.")

        for param in model.sae_module.encoder.parameters():
            param.requires_grad = False
        model.sae_module.W_dec.requires_grad = False
        model.sae_module.b_dec.requires_grad = False
        # logging.info("SAE encoder/decoder have been frozen. Only training MLP controller.")
        
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                self._check_time()
                avg_losses = self.fit(data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                dev_result = self.evaluate(data_dict['dev'], self.topk, self.metrics, prediction_label="prediction_sae")
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                
                logging_str = 'Epoch {:<5} loss={:<.4f} [total:{:<3.1f}s, backward:{:<3.1f}s]\tdev=({})'.format(
                    epoch + 1, 
                    avg_losses['Total Loss'], 
                    training_time, 
                    avg_losses['Backward Time'], 
                    utils.format_metric(dev_result)
                )

                if max(main_metric_results) == main_metric_results[-1]:
                    model.save_model(model.recsae_model_path) 
                    logging_str += ' *'
                
                logging.info(logging_str)

                log_dict = {
                    'epoch': epoch + 1,
                    'train/Total Loss': avg_losses['Total Loss'],
                    'train/Recon Loss': avg_losses['Recon Loss'],
                    'train/Tuning Loss': avg_losses['Tuning Loss'],
                    'time/Training': training_time,
                    'time/Backward': avg_losses['Backward Time']
                }
              
                for key, value in dev_result.items():
                    log_dict[f'dev/{key}'] = value
                
                if hasattr(model.sae_module, 'fvu'):
                     
                     if hasattr(self, 'last_eval_fvu'):
                         log_dict['dev/SAE FVU'] = self.last_eval_fvu
                
                wandb.log(log_dict)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")

        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({})".format(
            best_epoch + 1, utils.format_metric(dev_results[best_epoch])))
        
        model.load_model(model.recsae_model_path) 

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list, prediction_label="prediction_sae") -> Dict[str, float]:
        self._initialize_metadata(dataset.corpus)
        predictions = self.predict(dataset, prediction_label=prediction_label)
        ground_truth_items = np.array(dataset.data['item_id'])
        if len(predictions) != len(ground_truth_items):
            ground_truth_items = np.array(dataset.data['item_id'])

        all_results = self.evaluate_method(predictions, topks, metrics)

        if not self.eval_dims or not self.item_metadata:
            return all_results
        
        if self.eval_dims:
            for dim in self.eval_dims:
                metric_set = self.item_metadata.get(dim)
                if metric_set:
                    dim_mask = np.isin(ground_truth_items, list(metric_set))
                    if dim_mask.sum() > 0:
                        predictions_for_dim = predictions[dim_mask]
                        dim_accuracy_results = self.evaluate_method(predictions_for_dim, topks, metrics)
                        for key, value in dim_accuracy_results.items():
                            all_results[f'{dim}_{key}'] = value
        if self.eval_dims:
            rec_items_clean = self._get_clean_rec_list(predictions, dataset)
            for dim in self.eval_dims:
                metric_set = self.item_metadata.get(dim)
                if metric_set:
                    for k in topks:
                        key = f'{dim}_Ratio@{k}'
                        topk_recs = rec_items_clean[:, :k]
                        counts = np.isin(topk_recs, list(metric_set)).sum(axis=1)
                        ratios = counts / k
                        all_results[key] = ratios.mean()

        return all_results

    def _get_clean_rec_list(self, predictions, dataset):
        _predictions = predictions[:, 1:]
        if dataset.model.test_all:
            _candidate = np.arange(1, dataset.model.item_num)
            _candidate_items = np.tile(_candidate, (len(predictions), 1))
        else:
            _candidate_items = np.array([d['item_id'] for d in dataset])
        sort_idx = np.argsort(-_predictions, axis=1)
        return _candidate_items[np.arange(len(_candidate_items))[:, None], sort_idx]
    
    def print_res(self, dataset: BaseModel.Dataset, prediction_label="prediction_sae", save_result=False, phase='dev') -> str:
        if save_result:
            dataset.model.set_sae_mode("test")
            logging.info(f"Running in 'test' mode to collect data for phase '{phase}'.")
        else:
            dataset.model.set_sae_mode("inference")

        result_str = super().print_res(dataset, prediction_label=prediction_label)
        
        if save_result:
            activations_path = self.result_data_path + f"_{phase}_activations.csv"
            embedding_path = self.result_data_path + f"_{phase}_embedding.npy"
            
            dataset.model.save_epoch_result(dataset, path=activations_path, emb_path=embedding_path)

        return result_str
