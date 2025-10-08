# Create a new file: runners/FairnessRunner.py

import os
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List

from helpers.BaseRunner import BaseRunner
from models.BaseModel import BaseModel
from utils import utils

class TuningRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        # Add a new argument for evaluation dimensions
        parser.add_argument('--eval_dims', type=str, default='',
                            help='Comma-separated dimensions to evaluate on (e.g., "gluten_free,low_price").')
        return BaseRunner.parse_runner_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.eval_dims = [dim.strip() for dim in args.eval_dims.split(',') if dim.strip()]
        
        # These will be populated in the train method from the corpus
        self.item_metadata = None
        self.target_user_sets = {}

    def _initialize_metadata(self, corpus):
        if self.item_metadata is None:
            logging.info("Initializing item metadata for evaluation...")
            self.item_metadata = corpus.item_metadata

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list, prediction_label = None) -> Dict[str, float]:
        self._initialize_metadata(dataset.corpus)
        # 1. Standard evaluation
        predictions = self.predict(dataset)
        eval_results = self.evaluate_method(predictions, topks, metrics)

        # 2. Global dimension-specific evaluation
        if not self.eval_dims or not self.item_metadata:
            return eval_results
        
        _predictions = predictions[:, 1:]

        if dataset.model.test_all:
            _candidate = np.arange(1, dataset.model.item_num)
            _candidate_items = np.tile(_candidate, (len(predictions), 1))
        else:
            _candidate_items = np.array([d['item_id'] for d in dataset])

        sort_idx = np.argsort(-_predictions, axis=1)
        rec_items_clean = _candidate_items[np.arange(len(_candidate_items))[:, None], sort_idx]
        
        for dim in self.eval_dims:
            metric_set = self.item_metadata.get(dim)
            if metric_set is None:
                continue

            for k in topks:
                key_all = f'{dim}_Ratio@{k}'
                topk_recs = rec_items_clean[:, :k]
                counts = np.isin(topk_recs, list(metric_set)).sum(axis=1)
                ratios = counts / k
                eval_results[key_all] = ratios.mean()

        ground_truth_items = np.array(dataset.data['item_id'])
        
        for dim in self.eval_dims:
            metric_set = self.item_metadata.get(dim)
            if metric_set:
                dim_mask = np.isin(ground_truth_items, list(metric_set))
                
                if dim_mask.sum() > 0:
                    predictions_for_dim = predictions[dim_mask]
                    dim_accuracy_results = self.evaluate_method(predictions_for_dim, topks, metrics)
                    for key, value in dim_accuracy_results.items():
                        eval_results[f'{dim}_{key}'] = value 

        return eval_results

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        self._initialize_metadata(data_dict['train'].corpus)
        
        super().train(data_dict)