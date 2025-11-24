# this code for our AT-M-GNN is based on DIG-IN-GNN's original code

import torch
import torch.nn as nn
from torch.nn import init
from model.neigh_gen_model import Discriminator

class DiGInLayer(nn.Module):
	def __init__(self, num_classes, inter1, lambda_1, nei_gen, device, adv_loss_weight=1.0):
		"""
		Initialize the model
		:param num_classes: 2 for binary classification
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(DiGInLayer, self).__init__()
		self.device = device
		self.inter1 = inter1

		self.nei_gen = nei_gen

		self.xent = nn.CrossEntropyLoss()

		# 添加深度处理模块
		self.deep_process = nn.Sequential(
			nn.Linear(inter1.embed_dim, inter1.embed_dim * 2), # 扩大维度
			nn.ReLU(),
			nn.Dropout(0.5), # 添加Dropout
			nn.Linear(inter1.embed_dim * 2, inter1.embed_dim) # 还原维度
		)

		self.weight1 = nn.Parameter(torch.FloatTensor(128, inter1.embed_dim))
		# self.weight1_1 = nn.Parameter(torch.FloatTensor(64, 256))
		self.weight2 = nn.Parameter(torch.FloatTensor(num_classes, 128))
		init.xavier_uniform_(self.weight1)
		# init.xavier_uniform_(self.weight1_1)
		init.xavier_uniform_(self.weight2)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1
		self.adv_loss_weight = adv_loss_weight # 保存对抗损失权重

	def forward(self, nodes, labels, train_flag=True, rl_train_flag = True, rl_has_trained = False):
		embeds1, label_scores, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list \
			= self.inter1(nodes, labels, train_flag, rl_train_flag, rl_has_trained, self.device)
		
		# 应用深度处理模块和残差连接
		processed_embeds = self.deep_process(embeds1.t()).t() # 转置以适应Linear层，处理后再转置回来
		final_embeds = embeds1 + processed_embeds # 残差连接

		scores = self.weight1.mm(final_embeds)
		# scores = self.weight1_1.mm(scores)
		scores = self.weight2.mm(scores)
		return scores.t(), label_scores, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list

	def to_prob(self, nodes, labels, train_flag=False, rl_train_flag = False, rl_has_trained = True):
		gnn_logits, label_logits, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list \
			= self.forward(nodes, labels, train_flag, rl_train_flag, rl_has_trained)
		gnn_scores = torch.softmax(gnn_logits, dim=1)
		label_scores = 0
		return gnn_scores, label_scores, rate_list

	def loss(self, nodes, labels, train_flag=True, rl_train_flag = True, rl_has_trained = False):
		gnn_scores, label_scores, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list \
			= self.forward(nodes, labels, train_flag, rl_train_flag, rl_has_trained)

		reward_loss = 0
		for idx in range(3):
			if len(rewards[idx]) != 0:
				reward_loss += self.xent(rewards[idx], label_lists[idx].squeeze())
		reward_loss /= 3

		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		
		# 添加G-mean相关的损失项
		probs = torch.softmax(gnn_scores, dim=1)
		preds = torch.argmax(probs, dim=1)
		tn = torch.sum((preds == 0) & (labels == 0)).float()
		fp = torch.sum((preds == 1) & (labels == 0)).float()
		fn = torch.sum((preds == 0) & (labels == 1)).float()
		tp = torch.sum((preds == 1) & (labels == 1)).float()
		
		# 计算TPR和TNR
		tpr = tp / (tp + fn + 1e-8)
		tnr = tn / (tn + fp + 1e-8)
		
		# G-mean损失
		gmean_loss = 1 - torch.sqrt(tpr * tnr)
		
		# 组合损失，增加G-mean损失的权重
		final_loss = gnn_loss + 0.5 * reward_loss + 0.4 * gmean_loss

		context_loss = []
		env_loss = []
		for i in range(len(self.nei_gen)):
			c_loss, e_loss = self.nei_gen[i].discriminator.forward(gen_feats[i],raw_feats[i], env_feats[i], env_raw_feats[i], labels)
			context_loss.append(c_loss)
			env_loss.append(e_loss)

		return final_loss, context_loss, env_loss

	# 新增：对抗训练函数
	def adversarial_training(self, nodes, labels, optimizer, epsilon):
		# 正常前向传播
		embeds1, label_scores, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list = \
			self.inter1(nodes, labels, True, True, True, self.device)
		embeds1 = embeds1.detach()

		# reward_loss
		reward_loss = 0
		for idx in range(3):
			if len(rewards[idx]) != 0:
				reward_loss += self.xent(rewards[idx], label_lists[idx].squeeze())
		reward_loss /= 3

		# gnn_loss
		scores = self.weight1.mm(embeds1)
		scores = self.weight2.mm(scores)
		gnn_scores = scores.t()
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		final_loss = gnn_loss + 0.5 * reward_loss

		# context_loss, env_loss
		context_loss = []
		env_loss = []
		for i in range(len(self.nei_gen)):
			c_loss, e_loss = self.nei_gen[i].discriminator.forward(gen_feats[i], raw_feats[i], env_feats[i], env_raw_feats[i], labels)
			context_loss.append(c_loss)
			env_loss.append(e_loss)

		# 将FGSM扰动改为PGD迭代扰动，以提升对抗训练后AUC、F1和Gmean
		iter_num = 20  # 增加迭代扰动次数
		
		# PGD对抗训练的随机初始化
		adv_embeds = embeds1.clone().detach()
		# 在 [-epsilon, epsilon] 范围内进行随机初始化
		random_noise = torch.FloatTensor(adv_embeds.shape).uniform_(-epsilon, epsilon).to(self.device)
		adv_embeds = adv_embeds + random_noise
		# 确保初始扰动在epsilon范围内
		delta = adv_embeds - embeds1
		delta = torch.clamp(delta, -epsilon, epsilon)
		adv_embeds = embeds1 + delta
		adv_embeds = adv_embeds.detach()

		for _ in range(iter_num):
			adv_embeds.requires_grad_(True)
			adv_scores = self.weight1.mm(adv_embeds)
			adv_scores = self.weight2.mm(adv_scores)
			adv_gnn_scores = adv_scores.t()
			adv_gnn_loss = self.xent(adv_gnn_scores, labels.squeeze())
			adv_grad = torch.autograd.grad(adv_gnn_loss, adv_embeds, retain_graph=True, create_graph=False)[0]
			adv_embeds = adv_embeds + (epsilon / iter_num) * adv_grad.sign()
			# 投影扰动到epsilon邻域内
			delta = adv_embeds - embeds1
			delta = torch.clamp(delta, -epsilon, epsilon)
			adv_embeds = embeds1 + delta
			adv_embeds = adv_embeds.detach()

		# 对抗前向传播（使用扰动后的adv_embeds）
		adv_scores = self.weight1.mm(adv_embeds)
		adv_scores = self.weight2.mm(adv_scores)
		adv_gnn_scores = adv_scores.t()
		adv_gnn_loss = self.xent(adv_gnn_scores, labels.squeeze())
		adv_final_loss = adv_gnn_loss + 0.5 * reward_loss  # reward_loss不变

		# context_loss, env_loss（对抗部分）
		adv_context_loss = []
		adv_env_loss = []
		for i in range(len(self.nei_gen)):
			c_loss, e_loss = self.nei_gen[i].discriminator.forward(gen_feats[i], raw_feats[i], env_feats[i], env_raw_feats[i], labels)
			adv_context_loss.append(c_loss)
			adv_env_loss.append(e_loss)

		# 合并损失
		total_loss = final_loss + self.adv_loss_weight * adv_final_loss # 应用对抗损失权重
		total_context_loss = [c1 + c2 for c1, c2 in zip(context_loss, adv_context_loss)]
		total_env_loss = [e1 + e2 for e1, e2 in zip(env_loss, adv_env_loss)]
		return total_loss, total_context_loss, total_env_loss
