# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter

from utils import utils


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ML_1MTOPK_new',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()
        self.item_metadata = {}
        self.top_popular_items_in_top_categories = set()
        item_info_path = os.path.join(self.prefix, self.dataset, 'item_info.csv')
        logging.info(f"Loading item categories from: {item_info_path}")

        if os.path.exists(item_info_path):
            try:
                item_info_df = pd.read_csv(item_info_path, sep=self.sep)
                if 'genres' not in item_info_df.columns:
                    raise ValueError("Column 'genres' not found in item_info.csv")
                
                item_info_df['genres'] = item_info_df['genres'].fillna('')
                def parse_genres(genres_str):
                    if not isinstance(genres_str, str) or not genres_str.strip():
                        return []
                    return [genre.strip() for genre in genres_str.split('|') if genre.strip()]
                item_info_df['genres_list'] = item_info_df['genres'].apply(parse_genres)
                all_genres = list(chain.from_iterable(item_info_df['genres_list']))

                if all_genres:
                    category_counts = Counter(all_genres)

                    all_sorted_categories = [category for category, count in category_counts.most_common()]
                    top_10_categories = all_sorted_categories[:10]
                    logging.info(f"Total {len(all_sorted_categories)} categories found. Top 10: {category_counts.most_common(10)}")
                    category_to_items = {cat: set() for cat in all_sorted_categories}
                    
                    for _, row in item_info_df.iterrows():
                        item_id = row['item_id']
                        for category in row['genres_list']:
                            if category in category_to_items:
                                category_to_items[category].add(item_id)
                    
                    self.item_metadata = category_to_items
                    logging.info(f"Successfully loaded metadata for {len(self.item_metadata)} dimensions (genres).")
                    logging.info("Identifying top 10% popular items in top 10 categories...")
                    
                    item_popularity = self.all_df['item_id'].value_counts()
                    
                    for category in top_10_categories:
                        items_in_category = self.item_metadata.get(category, set())
                        
                        if not items_in_category:
                            logging.warning(f"Category '{category}' has no items, skipping.")
                            continue
                        category_item_popularity = item_popularity[item_popularity.index.isin(list(items_in_category))]
                        
                        if category_item_popularity.empty:
                            continue

                        top_10_percent_count = int(np.ceil(len(category_item_popularity) * 0.1))
                        top_items_for_category = category_item_popularity.nlargest(top_10_percent_count).index.tolist()
                        self.top_popular_items_in_top_categories.update(top_items_for_category)
                    
                    logging.info(f"Identified {len(self.top_popular_items_in_top_categories)} unique top popular items from top 10 categories.")
                else:
                    logging.warning("No genres found in item_info.csv. 'item_metadata' will be empty.")

            except Exception as e:
                logging.error(f"Failed to load and process item categories: {e}")
        else:
            logging.warning(f"Item info file not found at: {item_info_path}. 'item_metadata' will be empty.")
       
        self.train_clicked_set = dict()
        self.residual_clicked_set = dict()
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        key_columns = ['user_id','item_id','time']
        if 'label' in self.data_df['train'].columns:
            key_columns.append('label')
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))