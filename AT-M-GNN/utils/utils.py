import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from collections import defaultdict

from dgl.data.utils import load_graphs
import os

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
import cv2

"""
    Utility functions to handle data and evaluate model.
"""


def load_data(data, prefix="data/"):
    """
    Load graph structure, features, and labels.
    :param data: Dataset name.
    :param prefix: File path.
    :return: graphs, features, and labels.
    """

    if data == "yelp":
        data_file = loadmat(prefix + "YelpChi.mat")
        labels = data_file["label"].flatten()
        feat_data = data_file["features"].todense().A
        # load the preprocessed adj_lists
        with open(prefix + "yelp_homo_adjlists.pickle", "rb") as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + "yelp_rur_adjlists.pickle", "rb") as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + "yelp_rtr_adjlists.pickle", "rb") as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + "yelp_rsr_adjlists.pickle", "rb") as file:
            relation3 = pickle.load(file)
        file.close()
    elif data == "amazon":
        data_file = loadmat(prefix + "Amazon.mat")
        labels = data_file["label"].flatten()
        feat_data = data_file["features"].todense().A
        # load the preprocessed adj_lists
        with open(prefix + "amz_homo_adjlists.pickle", "rb") as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + "amz_upu_adjlists.pickle", "rb") as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + "amz_usu_adjlists.pickle", "rb") as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + "amz_uvu_adjlists.pickle", "rb") as file:
            relation3 = pickle.load(file)
    elif data == "tfinance":
        data_file = os.path.join(prefix, "tfinance")
        graph_tfinance, _ = load_graphs(data_file)
        graph_tfinance = graph_tfinance[0]
        labels = graph_tfinance.ndata["label"].argmax(1)
        labels = np.array(labels.cpu())
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            homo = pickle.load(file)
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            relation1 = pickle.load(file)
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            relation2 = pickle.load(file)
        with open(os.path.join(prefix, "homo_tfinance.pickle"), "rb") as file:
            relation3 = pickle.load(file)
        feat_data = np.array(graph_tfinance.ndata["feature"])

    return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
    """
    Row-normalize sparse matrix
    Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, "wb") as file:
        pickle.dump(adj_lists, file)
    file.close()


def pos_neg_split(nodes, labels):
    """
    Split nodes into positive and negative given their labels.
    :param nodes: A list of nodes.
    :param labels: A list of node labels.
    :returns: A tuple of two lists, containing the positive and negative nodes respectively.
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def test_sage(test_cases, labels, model, batch_size, thres=0.5):
    """
    Test the performance of GraphSAGE
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    """

    test_batch_num = int(len(test_cases) / batch_size) + 1
    gnn_pred_list = []
    gnn_prob_list = []
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes)

        gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    f1_binary_1_gnn = f1_score(
        labels, np.array(gnn_pred_list), pos_label=1, average="binary"
    )
    f1_binary_0_gnn = f1_score(
        labels, np.array(gnn_pred_list), pos_label=0, average="binary"
    )
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average="micro")
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average="macro")
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(
        f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"
        + f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}"
    )
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def prob2pred(y_prob, thres=0.5, labels=None, optimize_metric='f1_macro', multi_metrics=None):
    """
    Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :param labels: true labels for threshold optimization
    :param optimize_metric: metric to optimize ('f1_macro', 'gmean', 'f1_binary_1', 'f1_binary_0', 'precision_1', 'recall_1', 'balanced_accuracy')
    :param multi_metrics: dict for multi-metric optimization, e.g. {'f1_macro': 0.5, 'gmean': 0.3, 'f1_binary_1': 0.2}
    :returns: the predicted result with the same shape as y_prob
    """
    if labels is not None:
        # 计算最佳阈值
        best_metric = -1
        best_threshold = thres
        
        # 扩大阈值搜索范围并减小步长
        for threshold in np.arange(0.2, 0.81, 0.005):
            y_pred = np.zeros_like(y_prob, dtype=np.int32)
            y_pred[y_prob >= threshold] = 1
            y_pred[y_prob < threshold] = 0
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
            
            # 多指标联合优化
            if multi_metrics is not None:
                current_metric = 0
                for metric_name, weight in multi_metrics.items():
                    if metric_name == 'f1_macro':
                        metric_value = f1_score(labels, y_pred, average="macro")
                    elif metric_name == 'gmean':
                        tpr = tp / (tp + fn + 1e-8)
                        tnr = tn / (tn + fp + 1e-8)
                        metric_value = np.sqrt(tpr * tnr)
                    elif metric_name == 'f1_binary_1':
                        metric_value = f1_score(labels, y_pred, pos_label=1, average="binary")
                    elif metric_name == 'f1_binary_0':
                        metric_value = f1_score(labels, y_pred, pos_label=0, average="binary")
                    elif metric_name == 'precision_1':
                        metric_value = tp / (tp + fp + 1e-8)
                    elif metric_name == 'recall_1':
                        metric_value = tp / (tp + fn + 1e-8)
                    elif metric_name == 'balanced_accuracy':
                        tpr = tp / (tp + fn + 1e-8)
                        tnr = tn / (tn + fp + 1e-8)
                        metric_value = (tpr + tnr) / 2
                    elif metric_name == 'precision_0':
                        metric_value = tn / (tn + fn + 1e-8)
                    elif metric_name == 'recall_0':
                        metric_value = tn / (tn + fp + 1e-8)
                    else:
                        metric_value = 0
                    
                    current_metric += weight * metric_value
            else:
                # 单指标优化（原有逻辑）
                if optimize_metric == 'f1_macro':
                    current_metric = f1_score(labels, y_pred, average="macro")
                elif optimize_metric == 'gmean':
                    # 几何平均数
                    tpr = tp / (tp + fn + 1e-8)
                    tnr = tn / (tn + fp + 1e-8)
                    current_metric = np.sqrt(tpr * tnr)
                elif optimize_metric == 'f1_binary_1':
                    # 正样本F1分数
                    current_metric = f1_score(labels, y_pred, pos_label=1, average="binary")
                elif optimize_metric == 'f1_binary_0':
                    # 负样本F1分数
                    current_metric = f1_score(labels, y_pred, pos_label=0, average="binary")
                elif optimize_metric == 'precision_1':
                    # 正样本精确率
                    current_metric = tp / (tp + fp + 1e-8)
                elif optimize_metric == 'recall_1':
                    # 正样本召回率
                    current_metric = tp / (tp + fn + 1e-8)
                elif optimize_metric == 'balanced_accuracy':
                    # 平衡准确率
                    tpr = tp / (tp + fn + 1e-8)
                    tnr = tn / (tn + fp + 1e-8)
                    current_metric = (tpr + tnr) / 2
                elif optimize_metric == 'precision_0':
                    # 负样本精确率
                    current_metric = tn / (tn + fn + 1e-8)
                elif optimize_metric == 'recall_0':
                    # 负样本召回率
                    current_metric = tn / (tn + fp + 1e-8)
                else:
                    # 默认使用F1-macro
                    current_metric = f1_score(labels, y_pred, average="macro")
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_threshold = threshold
        
        # 使用最佳阈值进行预测
        y_pred = np.zeros_like(y_prob, dtype=np.int32)
        y_pred[y_prob >= best_threshold] = 1
        y_pred[y_prob < best_threshold] = 0
        return y_pred
    else:
        # 如果没有提供标签，使用默认阈值
        y_pred = np.zeros_like(y_prob, dtype=np.int32)
        y_pred[y_prob >= thres] = 1
        y_pred[y_prob < thres] = 0
        return y_pred


class rate_res:
    def __init__(self):
        self.rate = 0
        self.rate_f = 0
        self.rate_b = 0
        self.total_f = 0
        self.total_b = 0


def test_dig(test_cases, labels, model, batch_size, thres=0.5, rl_has_trained=True, optimize_threshold=True, optimize_metric='f1_macro', multi_metrics=None):
    """
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    :param thres: default threshold (used when optimize_threshold=False)
    :param rl_has_trained: whether RL model has been trained
    :param optimize_threshold: whether to optimize threshold on the entire dataset
    :param optimize_metric: metric to optimize ('f1_macro', 'gmean', 'f1_binary_1', 'f1_binary_0', 'precision_1', 'recall_1', 'balanced_accuracy', 'precision_0', 'recall_0')
    :param multi_metrics: dict for multi-metric optimization, e.g. {'f1_macro': 0.5, 'gmean': 0.3, 'f1_binary_1': 0.2}
    :returns: test or validation results
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    f1_label1 = 0.0
    acc_label1 = 0.00
    recall_label1 = 0.0
    gnn_pred_list = []
    gnn_prob_list = []

    r1 = rate_res()
    r2 = rate_res()
    r3 = rate_res()
    rate = [r1, r2, r3]

    # 首先收集所有预测概率和标签
    all_probs = []
    all_labels = []
    
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob, label_prob1, rate_list = model.to_prob(
            batch_nodes,
            batch_label,
            train_flag=False,
            rl_train_flag=False,
            rl_has_trained=rl_has_trained,
        )

        gnn_prob_arr = gnn_prob.detach().data.cpu().numpy()[:, 1]
        all_probs.extend(gnn_prob_arr.tolist())
        all_labels.extend(batch_label.tolist())

        if len(rate_list) == 3 and len(rate_list[0]) > 0:
            for i in range(len(rate_list)):
                rate[i].rate += rate_list[i][0]
                rate[i].rate_f += rate_list[i][1]
                rate[i].rate_b += rate_list[i][2]
                rate[i].total_f += rate_list[i][3]
                rate[i].total_b += rate_list[i][4]

    # 转换为numpy数组
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 如果启用阈值优化，在整个数据集上搜索最优阈值
    if optimize_threshold:
        if multi_metrics is not None:
            metric_names = list(multi_metrics.keys())
            print(f"\n   Searching optimal threshold for multi-metrics: {metric_names}")
            print(f"   Weights: {multi_metrics}")
        else:
            print(f"\n   Searching optimal threshold for {optimize_metric}...")
        gnn_pred = prob2pred(all_probs, thres=thres, labels=all_labels, optimize_metric=optimize_metric, multi_metrics=multi_metrics)
    else:
        # 使用默认阈值
        gnn_pred = prob2pred(all_probs, thres=thres, labels=None)

    # 计算各种指标
    for i in range(len(rate)):
        rate[i].rate /= test_batch_num
        rate[i].rate_f /= test_batch_num
        rate[i].rate_b /= test_batch_num

    auc_gnn = roc_auc_score(all_labels, all_probs)
    ap_gnn = average_precision_score(all_labels, all_probs)
    auc_label1 = 0
    ap_label1 = 0

    f1_binary_1_gnn = f1_score(all_labels, gnn_pred, pos_label=1, average="binary")
    f1_binary_0_gnn = f1_score(all_labels, gnn_pred, pos_label=0, average="binary")
    f1_micro_gnn = f1_score(all_labels, gnn_pred, average="micro")
    f1_macro_gnn = f1_score(all_labels, gnn_pred, average="macro")
    conf_gnn = confusion_matrix(all_labels, gnn_pred)
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(
        f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"
        + f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}"
    )
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    print(
        f"Label1 F1: {f1_label1 / test_batch_num:.4f}\tAccuracy: {acc_label1 / test_batch_num:.4f}"
        + f"\tRecall: {recall_label1 / test_batch_num:.4f}\tAUC: {auc_label1:.4f}\tAP: {ap_label1:.4f}"
    )
    for i in range(len(rate)):
        print(
            f"   relation: {i}\trate: {rate[i].rate:.4f}"
            + f"\trate_f: {rate[i].rate_f:.4f}"
            f"\trate_b: {rate[i].rate_b:.4f}"
            f"\ttotal_f: {rate[i].total_f}"
            f"\ttotal_b: {rate[i].total_b}"
        )

    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5


def pick_step(
    idx_train,
    y_train,
    adj_list,
    size,
    gen_train_flag,
    all_labels,
    b_pick=0.75,
    f_pick=0.45,
):
    # degree_train = [len(adj_list[node]) for node in idx_train]
    # lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    # lf_train2 = (y_train.sum() * y_train) - y_train.sum()
    # lf_train = lf_train + lf_train2
    # lf_train = np.sqrt(lf_train)
    max_l = len(y_train)

    # degree_train = [1 for node in idx_train]
    # lf_train = (y_train.sum()-len(y_train))*y_train + len(y_train)
    # smp_prob = np.array(degree_train) / lf_train
    # return random.choices(idx_train, weights=smp_prob, k=size)

    degree_train = [1 for node in idx_train]
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    # lf_train = [0.75 if i == max_l else 0.45 for i in lf_train]
    # lf_train = [0.85 if i == max_l else 0.45 for i in lf_train]
    lf_train = [b_pick if i == max_l else f_pick for i in lf_train]
    if gen_train_flag:
        # lf_train = np.sqrt(lf_train)
        lf_train = lf_train
    # lf_train = np.sqrt((y_train.sum() - len(y_train)) * y_train + len(y_XDtrain))
    smp_prob = np.array(degree_train) / lf_train
    return list((random.choices(idx_train, weights=smp_prob, k=size)))

    def get_numberof_posandneg(lf0=1, lf1=0.4):
        smp_prob = np.array(degree_train) / (
            [
                lf0 if i == max_l else lf1
                for i in ((y_train.sum() - len(y_train)) * y_train + len(y_train))
            ]
        )
        random.choices(idx_train, weights=smp_prob, k=size)
        ret = (all_labels[random.choices(idx_train, weights=smp_prob, k=size)]).sum()
        print(ret, size - ret)
        print(int(ret), int(size - ret))


# Random state.
RS = 50


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    colors0 = colors * 6
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors0.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("tight")
    ax.axis("off")

    ax.add_patch(
        plt.Rectangle((100, 100), 200, 100, color="blue", fill=False, linewidth=1)
    )

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=48)
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
        txts.append(txt)

    return f, ax, sc, txts


def getVISUAL(X, y, name, fmt="svg"):
    # digits = load_digits()
    # We first reorder the data points according to the handwritten numbers.
    # temp = digits.data[1]
    # X = np.vstack([digits.data[digits.target == i] for i in range(10)])
    # y = np.hstack([digits.target[digits.target == i]
    #                for i in range(10)])
    digits_proj = TSNE(random_state=RS).fit_transform(X)
    # scatter(digits_proj, y)
    f, ax, sc, txts = scatter(digits_proj, y)

    plt.title(name, y=-0.1, fontsize=40)
    plt.show()
    if fmt == "pdf":
        plt.savefig(name + ".pdf", format="pdf", dpi=1200)
    elif fmt == "svg":
        plt.savefig(name + ".svg", format="svg", dpi=1200)
    else:
        plt.savefig(name + "." + fmt, format=fmt, dpi=1200)


def draw(img, left, right, color):
    # img=cv2.imread(spath)
    img = cv2.rectangle(img, left, right, color, 3)
    return img


import csv

person = [("xxx", 18, 193), ("yyy", 18, 182), ("zzz", 19, 185)]
header = ["name", "age", "height"]


def writeHeader(file_name, header, mode):
    with open(file_name, mode, encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
    file_obj.close()


def writeRow(file_name, row, mode):
    with open(file_name, mode, encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(row)

    file_obj.close()
