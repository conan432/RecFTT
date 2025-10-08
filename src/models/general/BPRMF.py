# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BPRMF
Reference:
	"Bayesian personalized ranking from implicit feedback"
	Rendle et al., UAI'2009.
CMD example:
	python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch.nn as nn

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

from models.sae.sae import SAE
import logging
import numpy as np
import pandas as pd
import torch

class BPRMFBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return parser

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

	def forward(self, feed_dict):
		self.check_list = []
		u_ids = feed_dict['user_id']  # [batch_size]
		i_ids = feed_dict['item_id']  # [batch_size, -1]

		cf_u_vectors = self.u_embeddings(u_ids)
		cf_i_vectors = self.i_embeddings(i_ids)

		prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
		u_v = cf_u_vectors.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		i_v = cf_i_vectors
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class BPRMF(GeneralModel, BPRMFBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict =  BPRMFBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}

class BPRMFImpression(ImpressionModel, BPRMFBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return BPRMFBase.forward(self, feed_dict)
	

TRAIN_MODE = 1
INFERENCE_MODE = 0
TEST_MODE = 2

class BPRMF_SAE(BPRMF):
	reader = 'BaseReader'
	runner = "RecSAETuningRunner"
	sae_extra_params = ['loss_lambda', 'mlp_lr', 'tuning_dims'] # Renamed for consistency with SASRec_SAE

	@staticmethod
	def parse_model_args(parser):
		parser = SAE.parse_model_args(parser)
		parser = BPRMF.parse_model_args(parser)
		return parser
	
	def __init__(self, args, corpus):
		BPRMF.__init__(self, args, corpus)
		self.sae_module = SAE(args, self.emb_size)
		self.mode = "" # train, inference, test
		self.recsae_model_path = args.recsae_model_path

		# BPRMF doesn't have a transformer part, but we can encapsulate u_embeddings for clarity if needed
		self.own_part = nn.ModuleDict({
            'u_embeddings': self.u_embeddings,
            'i_embeddings': self.i_embeddings,
        }) 

		self.epoch_scale_vectors = None 
		self.epoch_users = None
		self.epoch_embedding = None 
		return
	
	def set_sae_mode(self, mode):
		if mode == 'train':
			self.mode = TRAIN_MODE
		elif mode == 'inference':
			self.mode = INFERENCE_MODE
		elif mode == 'test':
			self.mode = TEST_MODE
		else:
			raise ValueError(f"[BPRMF-SAE] mode ERROR!!! mode = {mode}")

	def get_dead_latent_ratio(self):
		return self.sae_module.get_dead_latent_ratio(need_update = (self.mode == TRAIN_MODE))

	def get_bprmf_user_embedding(self, feed_dict):
		u_ids = feed_dict['user_id']
		user_vectors_x = self.u_embeddings(u_ids)
		return user_vectors_x
	
	def _calculate_tuned_prediction_components(self, user_vectors_x, save_result=False):
		# For BPRMF, SAE operates on user_vectors_x
		sae_out_dict = self.sae_module(user_vectors_x, save_result=save_result)
		return sae_out_dict

	def forward(self, feed_dict):
		user_vectors_x = self.get_bprmf_user_embedding(feed_dict)
		i_ids = feed_dict['item_id']
		i_vectors = self.i_embeddings(i_ids)

		prediction_original = (user_vectors_x[:, None, :] * i_vectors).sum(-1)
		should_save_result = (self.mode == TEST_MODE)

		tuned_components = self._calculate_tuned_prediction_components(user_vectors_x, save_result=should_save_result)
		
		reconstruction_y = tuned_components['reconstruction_y'] 
		reconstruction_z = tuned_components['reconstruction_z'] 

		tuning_delta = reconstruction_z - reconstruction_y
		final_user_vector = user_vectors_x + tuning_delta

		prediction_sae = (final_user_vector[:, None, :] * i_vectors).sum(-1)

		if self.mode == TEST_MODE:
			if self.epoch_users is None:
				self.epoch_users = feed_dict['user_id'].detach().cpu().numpy()
				# BPRMF doesn't have history_items in feed_dict
				self.epoch_embedding = user_vectors_x.detach().cpu().numpy()
			else:
				self.epoch_users = np.concatenate((self.epoch_users, feed_dict['user_id'].detach().cpu().numpy()), axis=0)
				self.epoch_embedding = np.concatenate((self.epoch_embedding, user_vectors_x.detach().cpu().numpy()), axis=0)
			
			final_scale_vector = tuned_components['scale_vector']
			if self.epoch_scale_vectors is None:
				self.epoch_scale_vectors = final_scale_vector.detach().cpu().numpy()
			else:
				self.epoch_scale_vectors = np.concatenate((self.epoch_scale_vectors, final_scale_vector.detach().cpu().numpy()), axis=0)

		tuned_components['final_user_vector'] = final_user_vector
		output = {
			'prediction': prediction_original.view(feed_dict['batch_size'], -1),
			'prediction_sae': prediction_sae.view(feed_dict['batch_size'], -1), 
			'user_vectors_x': user_vectors_x,
			'candidate_items': i_ids, # Keep for potential future use or debugging
			'tuned_components': tuned_components
		}
		
		output['u_v'] = user_vectors_x.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		output['i_v'] = i_vectors
		
		return output
	
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
	
	def save_model(self, model_path):
		"""A dedicated save method is better for clarity."""
		torch.save(self.state_dict(), model_path)
		logging.info('Save model to ' + model_path)
	
	def save_epoch_result(self, dataset, path = None, emb_path = None):
		# self.sae_module.epoch_activations already has 'indices' and 'values' from SAE's forward pass in TEST_MODE
		
		df = pd.DataFrame()
		df['user_id'] = self.epoch_users
		df['indices'] = [x.tolist() for x in self.sae_module.epoch_activations['indices']]
		df['values'] =[x.tolist() for x in self.sae_module.epoch_activations['values']]
		
		if self.epoch_scale_vectors is not None:
			topk_indices = self.sae_module.epoch_activations['indices']
			scale_values_at_topk = []
			for i in range(len(topk_indices)):
				scales = self.epoch_scale_vectors[i][topk_indices[i]]
				scale_values_at_topk.append(scales.tolist())
			df['scale_factors'] = scale_values_at_topk
		
		df.to_csv(path,sep = "\t",index=False)
		
		np.save(emb_path, self.epoch_embedding)
		logging.info('save emb to '+ emb_path)

		# Reset epoch collection variables
		self.sae_module.epoch_activations = {"indices": None, "values": None} 
		self.epoch_users = None
		self.epoch_scale_vectors = None
		# self.epoch_history_items = None # Not applicable for BPRMF
		self.epoch_embedding = None
		return