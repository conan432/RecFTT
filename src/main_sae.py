# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch
import wandb
import json

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
from utils import utils
from helpers.RecSAETuningRunner import RecSAETuningRunner

def parse_global_args(parser):
	parser.add_argument('--gpu', type=str, default='0',
						help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
	parser.add_argument('--verbose', type=int, default=logging.INFO,
						help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='',
						help='Logging file path')
	parser.add_argument('--random_seed', type=int, default=0,
						help='Random seed of numpy and pytorch')
	parser.add_argument('--load', type=int, default=0,
						help='Whether load model and continue to train')
	parser.add_argument('--train', type=int, default=1,
						help='To train the model or not.')
	parser.add_argument('--save_final_results', type=int, default=1,
						help='To save the final validation and test results or not.')
	parser.add_argument('--regenerate', type=int, default=0,
						help='Whether to regenerate intermediate files')
	return parser

def train_sae(args, model, runner, data_dict):
	model.load_model(args.model_path)
	
	logging.info('[Rec] Dev Before Training: ' + runner.print_res(data_dict['dev']))
	logging.info('[SAE] Dev Before Training: ' + runner.print_res(data_dict['dev'], prediction_label = 'prediction_sae'))

	logging.info('[Rec] Test Before Training: ' + runner.print_res(data_dict['test']))
	logging.info('[SAE] Test Before Training: ' + runner.print_res(data_dict['test'], prediction_label = 'prediction_sae'))

	if args.train > 0:
		runner.train(data_dict)

	# Evaluate final results
	eval_res = runner.print_res(data_dict['dev'], prediction_label = 'prediction_sae')
	logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	eval_res = runner.print_res(data_dict['test'], prediction_label = 'prediction_sae')
	logging.info(os.linesep + 'Test After Training: ' + eval_res)
	
	model.actions_after_train()
	logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def test_sae(args, model, runner, data_dict):
	if hasattr(model.sae_module, 'output_heads'):
		logging.info("Multi-head controller detected. Building heads before loading saved model...")
		
		# 从命令行参数中获取所有在训练时使用过的维度
		# 我们假设测试时也通过 --tuning_dims 传入了这些维度
		if not args.tuning_dims:
			logging.error("FATAL: --tuning_dims is required for testing a multi-head model.")
			return

		tuning_dims = [dim.strip() for dim in args.tuning_dims.split(',') if dim.strip()]
		
		for dim in tuning_dims:
			# 根据命名约定，找到对应的 controller 配置文件
			controller_path = f"ablation_analysis_results/controller_{dim}.json"
			try:
				with open(controller_path, 'r') as f:
					latent_ids = json.load(f).get("ablations", [])
				
				# 调用模型的方法，为这个维度动态添加输出头
				if hasattr(model.sae_module, 'add_output_head'):
					model.sae_module.add_output_head(dim, len(latent_ids))
				else:
					logging.error("Model is missing the 'add_output_head' method required for multi-head setup.")
					return
			except FileNotFoundError:
				logging.warning(f"Controller file for dimension '{dim}' not found at {controller_path} during testing. "
								"This may cause loading errors if the model was trained with this head.")
	model.load_model(args.recsae_model_path)

	if args.probe_position == 'emb':
		runner.prediction_embedding(data_dict)
		eval_res = runner.print_res(data_dict['test'], prediction_label = 'prediction_sae',save_result = False, phase='test')
	else:
		eval_res = runner.print_res(data_dict['test'], prediction_label = 'prediction_sae',save_result = True, phase='test')
	# eval_res = runner.print_res(data_dict['dev'], prediction_label = 'prediction_sae')
	# logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	
	logging.info(os.linesep + 'Test After Training: ' + eval_res)

	# torch.save(model, args.recsae_model_path+"h")
	
	if args.save_final_results==1: # save the prediction results
		# save_rec_results(data_dict['dev'], runner, 100)
		save_rec_results(data_dict['test'], runner, 100, predict_label = "prediction_sae", ablation_latent = args.ablation_latent, ablation_scale = args.ablation_scale)
	
def sae_baseline(args, model, runner, data_dict):
	model.load_model(args.recsae_model_path)
	model.sae_module.reset_module_weight()

	if args.probe_position == 'emb':
		runner.prediction_embedding(data_dict)
		eval_res = runner.print_res(data_dict['test'], prediction_label = 'prediction_sae',save_result = False)
	else:
		eval_res = runner.print_res(data_dict['test'], prediction_label = 'prediction_sae',save_result = True)
	# eval_res = runner.print_res(data_dict['dev'], prediction_label = 'prediction_sae')
	# logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	
	logging.info(os.linesep + 'Test After Training: ' + eval_res)

	# torch.save(model, args.recsae_model_path+"h")
	
	if args.save_final_results==1: # save the prediction results
		# save_rec_results(data_dict['dev'], runner, 100)
		save_rec_results(data_dict['test'], runner, 100, predict_label = "prediction_sae", sae_baseline = 1)
	
	return

def save_all_sae(args, model, runner, data_dict):
	model.load_model(args.recsae_model_path)
	model.set_sae_mode("test")

	train_res = runner.print_res(data_dict['train'], prediction_label = 'prediction',save_result = True, phase = 'train')
	dev_res = runner.print_res(data_dict['dev'], prediction_label = 'prediction',save_result = True, phase = 'dev')
	test_res = runner.print_res(data_dict['test'], prediction_label = 'prediction',save_result = True, phase = 'test')

	runner.prediction_embedding(data_dict['test'], model)
	
	return

def main():
	logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
	exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
			   'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
	logging.info(utils.format_arg_str(args, exclude_lst=exclude))

	# Random seed
	utils.init_seed(args.random_seed)

	# GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cpu')
	if args.gpu != '' and torch.cuda.is_available():
		args.device = torch.device('cuda')
	logging.info('Device: {}'.format(args.device))

	# Read data
	corpus_path = os.path.join(args.path, args.dataset, model_name.reader+args.data_appendix+ '.pkl')
	if not args.regenerate and os.path.exists(corpus_path):
		logging.info('Load corpus from {}'.format(corpus_path))
		corpus = pickle.load(open(corpus_path, 'rb'))
	else:
		corpus = reader_name(args)
		logging.info('Save corpus to {}'.format(corpus_path))
		pickle.dump(corpus, open(corpus_path, 'wb'))

	# Define model
	model = model_name(args, corpus).to(args.device)
	logging.info('#params: {}'.format(model.count_variables()))
	logging.info(model)

	# Define dataset
	data_dict = dict()
	is_tuning_task = args.sae_train and hasattr(args, 'use_mlp_controller') and args.use_mlp_controller
	for phase in ['train', 'dev', 'test']:
		data_dict[phase] = model_name.Dataset(model, corpus, phase)
		data_dict[phase].prepare()
	# if is_tuning_task:
	# 	
	# 	logging.info("Using SAETuningDataset for efficient fine-tuning.")
	# 	
	# 	from models.BaseModel import SAETuningDataset 
	# 	for phase in ['train', 'dev', 'test']:
	# 		data_dict[phase] = SAETuningDataset(model, corpus, phase)
	# 		data_dict[phase].prepare() # prepare() 会执行预计算
	# else:
	# 
	# 	logging.info("Using standard model dataset.")
	# 	for phase in ['train', 'dev', 'test']:
	# 		data_dict[phase] = model_name.Dataset(model, corpus, phase)
	# 		data_dict[phase].prepare()

	runner = runner_name(args)

	# if args.sae_baseline:
	# 	sae_baseline(args, model, runner, data_dict)
	# elif args.sae_train:
	# 	train_sae(args, model, runner, data_dict)
	# else:
	# 	test_sae(args, model, runner, data_dict)
	if args.sae_train:
		if args.use_mlp_controller: # 这是我们的Tuning任务触发器
			if os.path.exists(args.model_path):
				logging.info(f"Loading pre-trained SASRec+SAE model from: {args.model_path}")
				model.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=False)
			else:
				logging.warning(f"Pre-trained SAE model not found at {args.model_path}. "
								"Proceeding with randomly initialized weights, which may lead to poor performance.")

			logging.info("Starting Targeted Tuning Training workflow...")
			runner.train(data_dict) 
			logging.info(os.linesep + '-' * 45 + ' FINAL EVALUATION ' + '-' * 45)
			
			dev_res = runner.print_res(data_dict['dev'], prediction_label='prediction_sae')
			logging.info("Final Dev Performance on Best Model: " + dev_res)

			test_res = runner.print_res(
				data_dict['test'], 
				prediction_label='prediction_sae', 
				save_result=True,  
				phase='test'     
			)
			logging.info("Final Test Performance on Best Model: " + test_res)

			# 4. (可选) 保存最终的推荐结果
			if args.save_final_results:
				logging.info("Saving final test recommendations...")
				save_rec_results(data_dict['test'], runner, 100, predict_label="prediction_sae")
		else:
			# 调用旧的SAE训练流程
			logging.info("Starting Standard SAE Training workflow...")
			train_sae(args, model, runner, data_dict)
	elif args.sae_baseline:
		sae_baseline(args, model, runner, data_dict)
	else: # 测试模式
		# 如果Tuning模型的测试流程不同，这里也需要增加判断
		test_sae(args, model, runner, data_dict)
	# save_all_sae(args, model, runner, data_dict)

	

def save_rec_results(dataset, runner, topk, predict_label = 'prediction',sae_baseline = 0,ablation_latent = -1, ablation_scale = 1):
	# model_name = '{0}{1}'.format(init_args.model_name,init_args.model_mode)
	# result_path = os.path.join(runner.log_path,runner.save_appendix, 'rec-{}-{}.csv'.format(model_name,dataset.phase))
	if sae_baseline:
		result_path = runner.result_data_path + '_prediction_baseline.csv'
	else:
		if ablation_latent != -1:
			result_path = runner.result_data_path + f'_prediction_ablation_{args.ablation_latent}-{args.ablation_scale}.csv'
		else:
			result_path = runner.result_data_path + '_prediction.csv'
	print("[result_path], result_path")
	utils.check_dir(result_path)

	if init_args.model_mode == 'CTR': # CTR task 
		logging.info('Saving CTR prediction results to: {}'.format(result_path))
		predictions, labels = runner.predict(dataset)
		users, items= list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			items.append(info['item_id'][0])
		rec_df = pd.DataFrame(columns=['user_id', 'item_id', 'pCTR', 'label'])
		rec_df['user_id'] = users
		rec_df['item_id'] = items
		rec_df['pCTR'] = predictions
		rec_df['label'] = labels
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['TopK','']: # TopK Ranking task
		logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
		predictions = runner.predict(dataset, prediction_label = predict_label)  # n_users, n_candidates
		users, rec_items, rec_predictions = list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			item_scores = zip(info['item_id'], predictions[i])
			sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
			rec_items.append([x[0] for x in sorted_lst])
			rec_predictions.append([x[1] for x in sorted_lst])
		rec_df = pd.DataFrame(columns=['user_id', 'rec_items', 'rec_predictions'])
		rec_df['user_id'] = users
		rec_df['rec_items'] = rec_items
		rec_df['rec_predictions'] = rec_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['Impression','General','Sequential']: # List-wise reranking task: Impression is reranking task for general/seq baseranker. General/Sequential is reranking task for rerankers with general/sequential input.
		logging.info('Saving all recommendation results to: {}'.format(result_path))
		predictions = runner.predict(dataset)  # n_users, n_candidates
		users, pos_items, pos_predictions, neg_items, neg_predictions= list(), list(), list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			pos_items.append(info['pos_items'])
			neg_items.append(info['neg_items'])
			pos_predictions.append(predictions[i][:dataset.pos_len])
			neg_predictions.append(predictions[i][:dataset.neg_len])
		rec_df = pd.DataFrame(columns=['user_id', 'pos_items', 'pos_predictions', 'neg_items', 'neg_predictions'])
		rec_df['user_id'] = users
		rec_df['pos_items'] = pos_items
		rec_df['pos_predictions'] = pos_predictions
		rec_df['neg_items'] = neg_items
		rec_df['neg_predictions'] = neg_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	else:
		return 0
	logging.info("{} Prediction results saved!".format(dataset.phase))

if __name__ == '__main__':
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='SASRec', help='Choose a model to run.')
	init_parser.add_argument('--model_mode', type=str, default='', 
							 help='Model mode(i.e., suffix), for context-aware models to select "CTR" or "TopK" Ranking task;\
            						for general/seq models to select Normal (no suffix, model_mode="") or "Impression" setting;\
                  					for rerankers to select "General" or "Sequential" Baseranker.')
	init_args, init_extras = init_parser.parse_known_args()
	
	rec_model = init_args.model_name.split('_')[0]
	# model_name = eval('{0}.{0}{1}'.format(init_args.model_name,init_args.model_mode))
	model_name = eval('{0}.{1}{2}'.format(rec_model,init_args.model_name,init_args.model_mode))
	# model_name = eval('{0}.{1}'.format(rec_model,init_args.model_name))
	reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
	#runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner
	temp_parser = argparse.ArgumentParser(description='')
	temp_parser = model_name.parse_model_args(temp_parser)
	temp_args, _ = temp_parser.parse_known_args()
	if init_args.model_name.endswith('_SAE') and temp_args.use_mlp_controller:
		runner_name = RecSAETuningRunner
		logging.info("Task: Tuning. Using Runner: RecSAETuningRunner")
	else:
		runner_name = eval('{0}.{0}'.format(model_name.runner)) # 使用模型定义的默认Runner
		logging.info(f"Task: Standard. Using Runner: {model_name.runner}")
	# import ipdb;ipdb.set_trace()
	# Args
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = runner_name.parse_runner_args(parser)
	parser = model_name.parse_model_args(parser)
	args, extras = parser.parse_known_args()
	
	args.data_appendix = '' # save different version of data for, e.g., context-aware readers with different groups of context
	if 'Context' in model_name.reader:
		args.data_appendix = '_context%d%d%d'%(args.include_item_features,args.include_user_features,
										args.include_situation_features)

	log_args = [rec_model+init_args.model_mode, args.dataset+args.data_appendix, str(args.random_seed)]
	for arg in ['lr', 'l2'] + model_name.extra_log_args:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name = '__'.join(log_args).replace(' ', '__')

	log_args = [init_args.model_name+init_args.model_mode+args.probe_position, args.dataset+args.data_appendix, str(args.random_seed)]
	for arg in ['lr', 'l2'] + model_name.extra_log_args + model_name.sae_extra_params:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name_all = '__'.join(log_args).replace(' ', '__')

	if args.log_file == '':
		args.log_file = '../log/{}/{}.txt'.format(init_args.model_name+init_args.model_mode, log_file_name_all)
	if args.model_path == '':
		args.model_path = '../model/{}/{}.pt'.format(rec_model+init_args.model_mode, log_file_name)
	if args.recsae_model_path == "":
		args.recsae_model_path = '../model/{}/{}.pt'.format(init_args.model_name+init_args.model_mode, log_file_name_all)
	if args.result_data_path == "":
		args.result_data_path = '../log/{}/result_file/{}'.format(init_args.model_name+init_args.model_mode, log_file_name_all)
		check_dir = "../log/{}/result_file/".format(init_args.model_name+init_args.model_mode)
		if not os.path.exists(check_dir):
			os.makedirs(check_dir)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO) 
	if logger.hasHandlers():
		logger.handlers.clear()
	file_handler = logging.FileHandler(args.log_file)
	logger.addHandler(file_handler)
	stream_handler = logging.StreamHandler(sys.stdout)
	logger.addHandler(stream_handler)

	experiment_name = log_file_name_all 
	os.environ["WANDB_MODE"] = "offline"
	wandb.init(
		project="RecSAE_Tuning_Project", 
		config=args,
		name=experiment_name,
	)
	logging.info(init_args)
	main()
	wandb.finish()

