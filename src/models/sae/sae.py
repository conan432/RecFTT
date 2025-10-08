import torch
import numpy as np
import json
import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging


class SAE(nn.Module):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--sae_k', type=int, default=32,
							help='top k activation')
		parser.add_argument('--sae_scale_size', type=int, default=32,
							help='scale size')
		parser.add_argument('--recsae_model_path', type=str, default='',
							help='Model save path.')
		parser.add_argument('--ablation_latent', type=int, default=-1,
						help='ablation Latent')
		parser.add_argument('--ablation_scale', type=int, default=1,
						help='ablation scale')
		parser.add_argument('--is_tuning', type=int, default=0,
							help='Tuning or not.')
		parser.add_argument('--controller_latents_path', type=str, default='',
                        help='Path to the file containing comma-separated latent indices to be controlled.')
		parser.add_argument('--tuning_dims', type=str, default='',
							help='The dimension to tune for (e.g., "gluten_free"). Must match a key in corpus.item_metadata.')
		parser.add_argument('--loss_lambda', type=float, default=0.1,
							help='Weight for the tuning loss component.')

		return parser
	
	def __init__(self,args,d_in):
		super(SAE, self).__init__()

		self.k = args.sae_k
		self.scale_size = args.sae_scale_size

		self.device = args.device
		self.dtype = torch.float32

		self.d_in = d_in
		self.hidden_dim = int(d_in * self.scale_size)

		self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
		self.encoder.bias.data.zero_()
		self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
		self.set_decoder_norm_to_unit_norm()
		self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))

		self.activate_latents = set()
		self.previous_activate_latents = None
		self.epoch_activations = {"indices": None, "values": None} 

		self.sae_baseline = args.sae_baseline
		self.rec_model_activations = None

		self.ablation_latent = args.ablation_latent
		self.ablation_scale = args.ablation_scale

		self.is_tuning = args.is_tuning
		self.tuning_dims = args.tuning_dims
		self.loss_lambda = args.loss_lambda
		self.tuning_item_set = set()

		self.controller_target_latents = None
		if self.is_tuning and args.tuning_dims:
			all_controllable_latents_set = set()
			tuning_dims = [dim.strip() for dim in args.tuning_dims.split(',') if dim.strip()]
			for dim in tuning_dims:
				if args.model_path.startswith('../model/BPRMF_SAE'):
					controller_path = f"ablation_analysis_results/bprmf_analyze/controller_{dim}.json"
					logging.info(f"Loading controller for dim '{dim}' from {controller_path}.")
				elif args.model_path.startswith('../model/LightGCN_SAE'):
					controller_path = f"ablation_analysis_results/lightgcn_analyze/controller_{dim}.json"
					logging.info(f"Loading controller for dim '{dim}' from {controller_path}.")
				elif args.model_path.startswith('../model/SASRec_SAE'):
					controller_path = f"ablation_analysis_results/controller_{dim}.json"
					logging.info(f"Loading controller for dim '{dim}' from {controller_path}.")
				else:
					logging.warning(f"Unrecognized model path prefix in {args.model_path}. Cannot determine controller path for dim '{dim}'.")
					continue
				try:
					with open(controller_path, 'r') as f:
						latent_ids = json.load(f).get("ablations", [])
					all_controllable_latents_set.update(latent_ids)
				except FileNotFoundError:
					logging.warning(f"Controller file for '{dim}' not found at {controller_path}.")
				
			unique_latents = sorted(list(all_controllable_latents_set))
			self.controller_target_latents = torch.tensor(unique_latents, device=self.device, dtype=torch.long)
			num_control_latents = len(self.controller_target_latents)
					
			if num_control_latents > 0:
				logging.info(f"Initializing a unified controller for {num_control_latents} unique latents from dims: {tuning_dims}")

				self.controller_multiplier = nn.Parameter(
                    torch.ones(num_control_latents, device=self.device, dtype=self.dtype)
                )
				self.controller_bias = nn.Parameter(
                    torch.zeros(num_control_latents, device=self.device, dtype=self.dtype)
                )
			else:
				logging.warning("No controllable latents found across all tuning dimensions. Controller disabled.")
				self.is_tuning = False
		return
	
	def set_tuning_set(self, item_set):
		self.tuning_item_set = item_set

	def reset_module_weight(self):
		self.encoder.reset_parameters()
		shape = self.W_dec.shape
		self.W_dec = nn.Parameter(torch.empty(shape, device=self.device))
		shape = self.b_dec.shape
		self.b_dec =  nn.Parameter(torch.empty(shape, device=self.device))

	def get_dead_latent_ratio(self, need_update = 0):
		ans =  1 - len(self.activate_latents)/self.hidden_dim
		# only update training situation for auxk_loss
		if need_update:
			# logging.info("[SAE] update previous activated Latent here")
			self.previous_activate_latents = torch.tensor(list(self.activate_latents)).to(self.device)
		self.activate_latents = set()
		return ans

	def set_decoder_norm_to_unit_norm(self):
		assert self.W_dec is not None, "Decoder weight was not initialized."
		eps = torch.finfo(self.W_dec.dtype).eps
		norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
		self.W_dec.data /= norm + eps

	def topk_activation(self, x, save_result):
		topk_values, topk_indices = torch.topk(x, self.k, dim=1)
		self.activate_latents.update(topk_indices.cpu().numpy().flatten())

		if save_result:
			# import ipdb;ipdb.set_trace()
			if self.epoch_activations["indices"] is None:
				self.epoch_activations["indices"] = topk_indices.detach().cpu().numpy()
				self.epoch_activations["values"] = topk_values.detach().cpu().numpy()
			else:
				self.epoch_activations["indices"] = np.concatenate((self.epoch_activations["indices"], topk_indices.detach().cpu().numpy()), axis=0)
				self.epoch_activations["values"] = np.concatenate((self.epoch_activations["values"], topk_values.detach().cpu().numpy()), axis=0)

		sparse_x = torch.zeros_like(x)
		sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
		return sparse_x, topk_indices
	

	def forward(self, x, save_result=False):
		sae_in = x - self.b_dec
		pre_acts_grad = self.encoder(sae_in)
		pre_acts_no_grad = pre_acts_grad.detach()
		z, topk_indices = self.topk_activation(nn.functional.relu(pre_acts_grad), save_result=save_result)
		with torch.no_grad():
			reconstruction_y = z.detach() @ self.W_dec + self.b_dec
		z_controlled = z
		final_scale_vector = torch.ones_like(z)
		final_bias_vector = torch.zeros_like(z)

		if self.is_tuning and self.controller_multiplier is not None:
			#target_latent_pre_acts = pre_acts_grad[:, self.controller_target_latents]
			target_latent_pre_acts = torch.index_select(pre_acts_grad, 1, self.controller_target_latents)
			scale_factors = self.controller_multiplier.unsqueeze(0).expand(x.size(0), -1)
			scale_vector = torch.ones_like(z)
			scale_vector.scatter_(1, self.controller_target_latents.repeat(z.size(0), 1), scale_factors)
			learned_bias = self.controller_bias.unsqueeze(0).expand(x.size(0), -1)
			bias_vector = torch.zeros_like(z)
			bias_vector.scatter_(1, self.controller_target_latents.repeat(z.size(0), 1), learned_bias)
			z_controlled = z * scale_vector + bias_vector
			final_scale_vector = scale_vector 
			final_bias_vector = bias_vector
		
		reconstruction_z = z_controlled @ self.W_dec + self.b_dec

		e = reconstruction_z - x
		total_variance = (x - x.mean(0)).pow(2).sum() + 1e-8
		self.fvu = e.pow(2).sum() / total_variance

		return {
			"reconstruction_y": reconstruction_y,     
			"reconstruction_z": reconstruction_z,     
			"final_user_vector": reconstruction_z,  
			"original_input": x,
			"sparse_activations": z,
			"scale_vector": final_scale_vector,
			"bias_vector": final_bias_vector,
			"tuned_components": {
				"reconstruction_z": reconstruction_z,
				"final_user_vector": reconstruction_z,
			},
			"topk_indices": topk_indices
		}