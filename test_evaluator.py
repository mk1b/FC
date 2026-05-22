import torch
import argparse
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import random

from model.TmpEncoder_stage5_clean import *
import os
import mne
import ast

from datasets.faced_dataset import LoadDataset as faced
from datasets.bciciv2a_dataset import LoadDataset as bciciv2a
from datasets.mumtaz_dataset import LoadDataset as mumtaz
from datasets.tuab_dataset import LoadDataset as tuab
from datasets.tuev_dataset import LoadDataset as tuev

from model.TmpEncoder_stage5_clean import Final
from model.TmpEncoder_continuous import Final as Final_continuous
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
	precision_recall_curve, auc, r2_score, mean_squared_error

hcp_positions_path = os.path.expanduser('~/simpletmp/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/simpletmp/processed_data/connectivity_matrix.txt')

class ConfiguredModel(nn.Module):
	def __init__(self, model, params):
		super().__init__()

		self.backbone = model
		self.params = params

		self.FFN = nn.Sequential(
						nn.Linear(200, self.params.nc),
					)

	def forward(self, x):
		Bz, num_chans, num_patches, patch_size = x.shape

		emb = self.backbone(x)
		emb = emb.mean(dim=(1, 2))

		out = self.FFN(emb)
		out = out.reshape(Bz, self.params.nc)

		return out

class Evaluator:
	def __init__(self, params, data_loader, model, epoch):

		self.params = params
		self.data_loader = data_loader

		self.model = model
		self.device = torch.device(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(self.device)

		state_dict_path = self.params.state_dict_path

		if self.params.use_pretrained_weights:
			map_location = self.device	
			state_dict = torch.load(state_dict_path, map_location=map_location)

			clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	

			self.model.load_state_dict(clean_state_dict)

		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)

	def get_metrics_for_multiclass(self):
		self.model.eval()

		truths = []
		preds = []
		losses = []

		with torch.no_grad():
			for i, (x, y) in enumerate(tqdm(self.data_loader, mininterval=1, desc="Evaluating Multiclass")):
				x = x.to(self.device)
				y = y.to(self.device).long()

				pred = self.model(x)

				pred_y = torch.max(pred, dim=-1)[1]

				loss = self.criterion_for_multiclass(pred, y)
				losses.append(loss.item())

				truths.extend(y.cpu().view(-1).numpy())
				preds.extend(pred_y.cpu().view(-1).numpy())

		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)

		acc = balanced_accuracy_score(truths, preds)
		f1 = f1_score(truths, preds, average='weighted')
		kappa = cohen_kappa_score(truths, preds)
		cm = confusion_matrix(truths, preds)

		return acc, kappa, f1, mean_loss, cm

	def get_metrics_for_binaryclass(self):
		self.model.eval() 

		truths = []
		preds = []
		scores = []
		losses = []

		with torch.no_grad():
			for i, (x, y) in enumerate(tqdm(self.data_loader, mininterval=1, desc="Evaluating Binary")):
				x = x.to(self.device)
				y = y.to(self.device).float()
				logit = self.model(x)

				logit_flat = logit.view(-1)
				y_flat = y.view(-1)

				score_y = torch.sigmoid(logit_flat)
				pred_y = torch.gt(score_y, 0.5).long()

				loss = self.criterion_for_binary_class(logit_flat, y_flat)
				losses.append(loss.item())

				truths.extend(y_flat.cpu().long().numpy())
				preds.extend(pred_y.cpu().numpy())
				scores.extend(score_y.cpu().numpy())

		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)
		scores = np.array(scores)

		acc = balanced_accuracy_score(truths, preds)
		cohen = cohen_kappa_score(truths, preds)

		precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
		pr_auc = auc(recall, precision)

		try:
			roc_auc = roc_auc_score(truths, scores)
		except ValueError:
			roc_auc = 0.0 

		return acc, pr_auc, roc_auc, cohen, mean_loss

	def get_metrics_for_regression(self):
		self.model.eval() 

		truths = []
		preds = []
		losses = []

		with torch.no_grad(): 
			for x, y in tqdm(self.data_loader, mininterval=1):
				x = x.to(self.device)
				y = y.to(self.device).float()

				pred = self.model(x)

				truths += y.cpu().squeeze(-1).numpy().tolist()
				preds += pred.cpu().squeeze(-1).numpy().tolist()

				loss = self.criterion_for_regression(pred.squeeze(-1), y.squeeze(-1))
				losses.append(loss.item())
		
		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)

		try:
			corrcoef = np.corrcoef(truths, preds)[0, 1]
		except Exception:
			corrcoef = 0.0
	
		r2 = r2_score(truths, preds)
		rmse = mean_squared_error(truths, preds) ** 0.5

		return corrcoef, r2, rmse, mean_loss

def seed_init(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True

def sorted_maps(ch_names):
	montage = mne.channels.make_standard_montage('standard_1005')
	all_pos = montage.get_positions()['ch_pos']
  
	# mapped_pos = {}
	pos_array = np.array([all_pos[ch] for ch in ch_names]) * 1000

	brain_regions = np.loadtxt(hcp_positions_path)
	used_pos = []

	for ch in ch_names:
		pos = all_pos[ch] * 1000
		idx = np.argmin(np.sum((brain_regions - pos)**2, axis=1))

		used_pos.append(idx)
	 
	correlations = np.loadtxt(connectivity_path)

	sub_corr = correlations[np.ix_(used_pos, used_pos)]
	sorted_map = torch.tensor(np.argsort(-sub_corr, axis=1))

	return sorted_map

def main():
	parser = argparse.ArgumentParser(description='argparser')
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--model_dir', type=str, default='~/simpletmp/saved_fm')

	parser.add_argument('--use_pretrained_weights', action='store_false')
	parser.add_argument('--state_dict_path', type=str, default='~/CantusCerebra/saved_fm')

	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--dataset_dir', type=str, default='~/CantusCerebra/data/TUAB/edf/process_refine')
	#/home/salvimanish/simpletmp/saved_fm/finetune_weights_17_tuev_169_modulo.pth
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout value')

	parser.add_argument('--in_dim', type=int, default=200, help='Number of samples in 1s raw')
	parser.add_argument('--out_dim', type=int, default=200, help='Output dimension')

	parser.add_argument('--d_model', type=int, default=200, help='Model operating dimension')
	parser.add_argument('--d_ffn', type=int, default=800, help='Standard 2-layer FFN dimensions')

	parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer layers')
	parser.add_argument('--num_heads', type=int, default=8, help='Number of Heads in MHSA')

	parser.add_argument('--convolution_set', type=str, default='[(1,), (3,), (5,)]', help='Concentrated convolution sizes. < num_chans')
	parser.add_argument('--seq_len', type=int, default=30, help='num_patches')

	parser.add_argument('--is_causal', action='store_true', help='If you want causal Temporal Attention')
	parser.add_argument('--need_key_padding', action='store_true', help='if any padding that could be added is to be ignored')

	parser.add_argument('--stride', type=int, default=1, help='stride for temp convs')

	parser.add_argument('--mode', type=str, default='bin')
	parser.add_argument('--epoch', type=int, default=0)

	parser.add_argument('--nc', type=int, default=1, help='Number of classes/outputs')
	parser.add_argument('--dataset', default='FACED', help='FACED,BCICIV2a, Mumtaz2016, TUEV, TUAB')
	parser.add_argument('--config', default='random', help='random,cont,hcp')

	params = parser.parse_args()
	params.state_dict_path = os.path.expanduser(params.state_dict_path)

	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	params.model_dir = os.path.expanduser(params.model_dir)
	seed_init(params.seed)
	#sorted_map = sorted_maps().to(f'cuda:{params.cuda}' if torch.cuda.is_available() else 'cpu')
	   
	if params.dataset == 'FACED':
		dataset = faced(params)
		sorted_map = sorted_maps([
			"Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", 
			"Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", 
			"Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2"
		])
		params.nc = 9
		random_map = torch.randint_like(sorted_map, high=32)
	if params.dataset == 'BCICIV2a':
		dataset = bciciv2a(params)
		sorted_map = sorted_maps([
			"Fz", 
			"FC3", "FC1", "FCz", "FC2", "FC4", 
			"C5", "C3", "C1", "Cz", "C2", "C4", "C6", 
			"CP3", "CP1", "CPz", "CP2", "CP4", 
			"P1", "Pz", "P2", "POz"
		])
		params.nc = 5
		random_map = torch.randint_like(sorted_map, high=22)
	if params.dataset == 'Mumtaz2016':
		dataset = mumtaz(params)
		sorted_map = sorted_maps([
			'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
			'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
		])
		random_map = torch.randint_like(sorted_map, high=19)
		params.nc = 1
	if params.dataset == 'TUEV':
		dataset = tuev(params)
		sorted_map = sorted_maps([
			'Fp1', 'F7', 'T7', 'P7', 'O1', 
			'Fp2', 'F8', 'T8', 'P8', 'O2', 
			'F3', 'C3', 'P3', 'F4', 'C4', 'P4'
		])
		params.nc = 6
		random_map = torch.randint_like(sorted_map, high=16)
	if params.dataset == 'TUAB':
		dataset = tuab(params)
		sorted_map = sorted_maps([
			'Fp1', 'F7', 'T7', 'P7', 'O1', 
			'Fp2', 'F8', 'T8', 'P8', 'O2', 
			'F3', 'C3', 'P3', 'F4', 'C4', 'P4'
		])
		params.nc = 1
		random_map = torch.randint_like(sorted_map, high=16)
	
	data_loader = dataset.get_data_loader()
	if(params.config == 'random'):
		mp = random_map
	else:
		mp = sorted_map
	
	if(params.config == 'cont'):
		model=Final_continuous(mp, in_dim=params.in_dim, out_dim=params.out_dim, d_model=params.d_model, num_layers=params.num_layers, convolution_set=ast.literal_eval(params.convolution_set), stride=params.stride, dropout=params.dropout, d_ffn=params.d_ffn, num_heads=params.num_heads)
	else:
		model=Final(mp, in_dim=params.in_dim, out_dim=params.out_dim, d_model=params.d_model, num_layers=params.num_layers, convolution_set=ast.literal_eval(params.convolution_set), stride=params.stride, dropout=params.dropout, d_ffn=params.d_ffn, num_heads=params.num_heads)
		
	model = ConfiguredModel(model, params=params)
	model.eval()
	evaluator = Evaluator(params, data_loader['test'], model, params.epoch)

	if params.mode == 'bin':
		acc, pr_auc, roc_auc, cohen, mean_loss = evaluator.get_metrics_for_binaryclass()
		print(f'b_acc: {acc}, pr_auc: {pr_auc}, auroc: {roc_auc}, kappa: {cohen}, mean_loss: {mean_loss}')
	if params.mode == 'mul':
		acc, kappa, f1, mean_loss, cm = evaluator.get_metrics_for_multiclass()
		print(f'b_acc:{acc}, kappa:{kappa}, f1:{f1}, mean_loss:{mean_loss}, cm:{cm}')
	if params.mode == 'reg':
		corrcoef, r2, rmse, mean_loss = evaluator.get_metrics_for_regression()
		print(f'corrcoef:{corrcoef}, r2:{r2}, rmse:{rmse}, mean_loss:{mean_loss}')

if __name__ == '__main__':
	main()
