# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class LightGCNBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			# D^-1/2 * A * D^-1/2
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)

	def forward(self, feed_dict):
		self.check_list = []
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)

		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		i_v = i_embed
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class LightGCN(GeneralModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}
	
class LightGCNImpression(ImpressionModel, LightGCNBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return LightGCNBase.forward(self, feed_dict)

class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj

		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()

	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def forward(self, users, items):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]

		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]

		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)

		user_all_embeddings = all_embeddings[:self.user_count, :]
		item_all_embeddings = all_embeddings[self.user_count:, :]

		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		return user_embeddings, item_embeddings


from models.sae.sae import SAE

TRAIN_MODE = 1
INFERENCE_MODE = 0
TEST_MODE = 2

from models.sae.sae import SAE
import logging
import numpy as np
import pandas as pd

class LightGCN_SAE(LightGCN):
	reader = 'BaseReader'
	runner = "RecSAETuningRunner"
	sae_extra_params = ['loss_lambda', 'mlp_lr', 'tuning_dims']


	@staticmethod
	def parse_model_args(parser):
		parser = SAE.parse_model_args(parser)
		parser = LightGCN.parse_model_args(parser)
		return parser


	def __init__(self, args, corpus):
		LightGCN.__init__(self, args, corpus)
		self.sae_module = SAE(args, self.emb_size)
		self.mode = ""
		self.recsae_model_path = args.recsae_model_path

		self.own_part = nn.ModuleDict({'encoder': self.encoder})

		self.epoch_scale_vectors = None
		self.epoch_users = None
		self.epoch_history_items = None
		self.epoch_embedding = None
		

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

	def get_lightgcn_embeddings(self, feed_dict):
		users, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(users, items)
		return u_embed, i_embed
	
	def _calculate_tuned_prediction_components(self, user_vectors_x, save_result=False):
		sae_out_dict = self.sae_module(user_vectors_x, save_result=save_result)
		return sae_out_dict

	def forward(self, feed_dict):
		user_vectors_x, i_vectors = self.get_lightgcn_embeddings(feed_dict)

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
			'candidate_items': feed_dict['item_id'],
			'tuned_components': tuned_components,
			'u_v': user_vectors_x.repeat(1,i_vectors.shape[1]).view(i_vectors.shape[0],i_vectors.shape[1],-1),
			'i_v': i_vectors
		}
		
		return output

	# def forward(self, feed_dict):
	# 	out_dict = LightGCNBase.forward(self, feed_dict)
	# 	return {'prediction': out_dict['prediction']}
	
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
	
	# def forward(self, feed_dict):
	# 	self.check_list = []
	# 	user, items = feed_dict['user_id'], feed_dict['item_id']
	# 	u_embed, i_embed = self.encoder(user, items)

	# 	save_result_flag = {INFERENCE_MODE:False, TEST_MODE:True}
	# 	if self.mode == INFERENCE_MODE or self.mode == TEST_MODE:
	# 		sae_output = self.sae_module(u_embed, save_result = save_result_flag[self.mode])
	# 		if self.mode == TEST_MODE:
	# 			if self.epoch_users is None:
	# 				self.epoch_users = feed_dict['user_id'].detach().cpu().numpy()
	# 				self.epoch_embedding = u_embed.detach().cpu().numpy()
	# 			else:
	# 				self.epoch_users = np.concatenate((self.epoch_users, feed_dict['user_id'].detach().cpu().numpy()), axis=0)
	# 				self.epoch_embedding = np.concatenate((self.epoch_embedding, u_embed.detach().cpu().numpy()), axis=0)
	# 	elif self.mode == TRAIN_MODE:
	# 		sae_output = self.sae_module(u_embed, train_mode = True)

	# 	prediction_sae = (sae_output[:, None, :] *  i_embed).sum(dim=-1)  # [batch_size, -1]
	# 	prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
	

	# 	return {'prediction': prediction.view(feed_dict['batch_size'], -1), "prediction_sae":prediction_sae.view(feed_dict['batch_size'], -1)}
	
	# def save_epoch_result(self, dataset, path = None, emb_path = None):

	# 	df = pd.DataFrame()

	# 	df['user_id'] = self.epoch_users
	# 	df['indices'] = [x.tolist() for x in self.sae_module.epoch_activations['indices']]
	# 	df['values'] =[x.tolist() for x in self.sae_module.epoch_activations['values']]

	# 	df.to_csv(path,sep = "\t",index=False)
	# 	# with open(path,'w') as f:
	# 	# 	f.write(json.dumps(self.sae_module.epoch_activations))

	# 	np.save(emb_path, self.epoch_embedding)
	# 	logging.info('save emb to '+ emb_path)

	# 	self.sae_module.epoch_activations = {"indices": None, "values": None} 
	# 	self.epoch_users = None
	# 	self.epoch_history_items = None
	# 	self.epoch_embedding = None
	# 	return

	def save_epoch_result(self, dataset, path=None, emb_path=None):
		df = pd.DataFrame()
		df['user_id'] = self.epoch_users
		df['indices'] = [x.tolist() for x in self.sae_module.epoch_activations['indices']]
		df['values'] = [x.tolist() for x in self.sae_module.epoch_activations['values']]
		
		if self.epoch_scale_vectors is not None:
			topk_indices = self.sae_module.epoch_activations['indices']
			scale_values_at_topk = []
			for i in range(len(topk_indices)):
				scales = self.epoch_scale_vectors[i][topk_indices[i]]
				scale_values_at_topk.append(scales.tolist())
			df['scale_factors'] = scale_values_at_topk
		
		df.to_csv(path, sep="\t", index=False)
		
		np.save(emb_path, self.epoch_embedding)
		logging.info('save emb to ' + emb_path)

		# Reset epoch collection variables
		self.sae_module.epoch_activations = {"indices": None, "values": None}
		self.epoch_users = None
		self.epoch_scale_vectors = None
		self.epoch_embedding = None
		return