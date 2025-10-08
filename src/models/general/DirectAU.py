# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" DirectAU
Reference:
    "Towards Representation Alignment and Uniformity in Collaborative Filtering"
    Wang et al., KDD'2022.
CMD example:
    python main.py --model_name DirectAU --dataset Grocery_and_Gourmet_Food \
                   --emb_size 64 --lr 1e-3 --l2 1e-6 --epoch 500 --gamma 0.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel

from models.sae.sae import SAE
import logging
import numpy as np
import pandas as pd

class DirectAU(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'gamma']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Weight of the uniformity loss.')
        return GeneralModel.parse_model_args(parser)

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif 'Embedding' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.gamma = args.gamma
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']

        user_e = self.u_embeddings(user)
        item_e = self.i_embeddings(items)

        prediction = (user_e[:, None, :] * item_e).sum(dim=-1)  # [batch_size, -1]
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict.update({
                'user_e': user_e,
                'item_e': item_e.squeeze(1)
            })

        return out_dict

    def loss(self, output):
        user_e, item_e = output['user_e'], output['item_e']

        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + self.gamma * uniform

        return loss

    class Dataset(GeneralModel.Dataset):
        # No need to sample negative items
        def actions_before_epoch(self):
            self.data['neg_items'] = [[] for _ in range(len(self))]



TRAIN_MODE = 1
INFERENCE_MODE = 0
TEST_MODE = 2

class DirectAU_SAE(DirectAU):
    runner = 'RecSAERunner'
    sae_extra_params = ['sae_lr','sae_batch_size','sae_k','sae_scale_size']

    @staticmethod
    def parse_model_args(parser):
        parser = SAE.parse_model_args(parser)
        parser = DirectAU.parse_model_args(parser)
        return parser
    
    def __init__(self, args, corpus):
        DirectAU.__init__(self, args, corpus)
        self.sae_module = SAE(args, self.emb_size)
        self.mode = "" # train, inference
        self.recsae_model_path = args.recsae_model_path

        self.epoch_users = None
        self.epoch_history_items = None
        return
    
    def set_sae_mode(self, mode):
        if mode == 'train':
            self.mode = TRAIN_MODE
        elif mode == 'inference':
            self.mode = INFERENCE_MODE
        elif mode == 'test':
            self.mode = TEST_MODE
        else:
            raise ValueError(f"[SASRec-SAE] mode ERROR!!! mode = {mode}")

    def get_dead_latent_ratio(self):
        return self.sae_module.get_dead_latent_ratio(need_update = self.mode)

    
    # def forward(self, feed_dict):
    #     self.check_list = []
    #     user, items = feed_dict['user_id'], feed_dict['item_id']

    #     user_e = self.u_embeddings(user)
    #     item_e = self.i_embeddings(items)

    #     # print(items.shape, item_e.shape)
    #     # import ipdb;ipdb.set_trace()

    #     batch_size = feed_dict['user_id'].shape[0]
    #     save_result_flag = {TRAIN_MODE:False,INFERENCE_MODE:False, TEST_MODE:True}
    #     # if self.mode == INFERENCE_MODE or self.mode == TEST_MODE:

    #     batches = torch.split(item_e[0,1:,:], batch_size)
    #     sae_output_list = []
    #     for i, batch in enumerate(batches):
    #         batch_item_e_sae = self.sae_module(batch, save_result = save_result_flag[self.mode])
    #         sae_output_list.append(batch_item_e_sae)
    #     # import ipdb;ipdb.set_trace()
    #     ground_truth_emb = item_e[:,0,:]
    #     # print("ground_truth_emb",ground_truth_emb.shape)
    #     ground_truth_emb_sae= self.sae_module(ground_truth_emb, save_result = save_result_flag[self.mode])
    #     ground_truth_emb_sae = ground_truth_emb_sae.unsqueeze(1)
    #     candidate_emb_sae = torch.cat(sae_output_list, dim = 0).unsqueeze(0).expand(batch_size,-1,-1)
    #     item_e_sae = torch.cat((ground_truth_emb_sae, candidate_emb_sae), dim=1) 
    #     if self.mode == TEST_MODE:
    #         if self.epoch_users is None:
    #             self.epoch_users = feed_dict['item_id'].detach().cpu().numpy()
    #             self.epoch_history_items = [] #history.detach().cpu().numpy()
    #         else:
    #             self.epoch_users = np.concatenate((self.epoch_users, feed_dict['item_id'].detach().cpu().numpy()), axis=0)
    #             self.epoch_history_items = [] #np.concatenate((self.epoch_history_items, history.detach().cpu().numpy()), axis=0)
    #     # elif self.mode == TRAIN_MODE:
    #     #     # item_e_sae = self.sae_module(item_e, train_mode = True)
    #     #     batches = torch.split(item_e[0,1:,:], batch_size)
    #     #     sae_output_list = []
    #     #     for i, batch in enumerate(batches):
    #     #         batch_item_e_sae = self.sae_module(batch, train_mode = True)
    #     #         sae_output_list.append(batch_item_e_sae)
    #     #     # import ipdb;ipdb.set_trace()
    #     #     ground_truth_emb = item_e[:,0,:]
    #     #     # print("ground_truth_emb",ground_truth_emb.shape)
    #     #     ground_truth_emb_sae= self.sae_module(ground_truth_emb, train_mode = True)
    #     #     ground_truth_emb_sae = ground_truth_emb_sae.unsqueeze(1)
    #     #     candidate_emb_sae = torch.cat(sae_output_list, dim = 0).unsqueeze(0).expand(batch_size,-1,-1)
    #     #     item_e_sae = torch.cat((ground_truth_emb_sae, candidate_emb_sae), dim=1) 
    #     sae_prediction = (user_e[:, None, :] * item_e_sae).sum(dim=-1)  # [batch_size, -1]


    #     prediction = (user_e[:, None, :] * item_e).sum(dim=-1)  # [batch_size, -1]
    #     out_dict = {'prediction': prediction, "prediction_sae": sae_prediction}

    #     if feed_dict['phase'] == 'train':
    #         out_dict.update({
    #             'user_e': user_e,
    #             'item_e': item_e.squeeze(1)
    #         })

    #     return out_dict

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']

        user_e = self.u_embeddings(user)
        item_e = self.i_embeddings(items)

        save_result_flag = {INFERENCE_MODE:False, TEST_MODE:True}
        if self.mode == INFERENCE_MODE or self.mode == TEST_MODE:
            sae_output = self.sae_module(user_e, save_result = save_result_flag[self.mode])
            if self.mode == TEST_MODE:
                if self.epoch_users is None:
                    self.epoch_users = feed_dict['user_id'].detach().cpu().numpy()
                else:
                    self.epoch_users = np.concatenate((self.epoch_users, feed_dict['user_id'].detach().cpu().numpy()), axis=0)
        else:
            sae_output = self.sae_module(user_e, train_mode = True)

        prediction_sae = (sae_output[:, None, :] * item_e).sum(dim=-1)
        prediction = (user_e[:, None, :] * item_e).sum(dim=-1)  # [batch_size, -1]
        out_dict = {'prediction': prediction, "prediction_sae":prediction_sae}

        if feed_dict['phase'] == 'train':
            out_dict.update({
                'user_e': user_e,
                'item_e': item_e.squeeze(1)
            })

        return out_dict
    

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path

        if model_path == self.model_path:
            state_dict = torch.load(model_path)
            self.load_state_dict(state_dict, strict = False)
            for name, param in self.named_parameters():
                if name in state_dict:
                    param.requires_grad = False
        else:
            self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)
        return
	
    # def save_epoch_result(self, dataset, path = None):
    #     # import ipdb;ipdb.set_trace()
    #     self.sae_module.epoch_activations['user_id'] = self.epoch_users
    #     self.sae_module.epoch_activations['history'] = self.epoch_history_items
    #     df = pd.DataFrame()
    #     df['user_id'] = self.epoch_users
    #     df['history'] = [np.trim_zeros(row, 'b').tolist() for row in self.epoch_history_items]
    #     df['indices'] = [x.tolist() for x in self.sae_module.epoch_activations['indices']]
    #     df['values'] =[x.tolist() for x in self.sae_module.epoch_activations['values']]
    #     df.to_csv(path,sep = "\t",index=False)
    #     # with open(path,'w') as f:
    #     # 	f.write(json.dumps(self.sae_module.epoch_activations))

    #     self.sae_module.epoch_activations = {"indices": None, "values": None} 
    #     self.epoch_users = None
    #     self.epoch_history_items = None
    #     return
    
    def save_epoch_result(self, dataset, path = None):
        df = pd.DataFrame()
        df['user_id'] = self.epoch_users
        df['indices'] = [x.tolist() for x in self.sae_module.epoch_activations['indices']]
        df['values'] =[x.tolist() for x in self.sae_module.epoch_activations['values']]
        df.to_csv(path,sep = "\t",index=False)

        self.sae_module.epoch_activations = {"indices": None, "values": None} 
        self.epoch_users = None
        self.epoch_history_items = None
        return