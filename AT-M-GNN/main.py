import argparse
import yaml
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict
import random
import os
import datetime

from model.model_handler import ModelHandler
from utils.visualization import plot_auc_heatmap, plot_auc_3d, prepare_auc_matrix


################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    """Set seeds for reproducibility across different environments and hardware.
    Note: Minor differences may still exist.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(config):
    print_config(config)
    set_random_seed(config["seed"])
    # 确保optimize_threshold参数存在
    if "optimize_threshold" not in config:
        config["optimize_threshold"] = True
    model = ModelHandler(config)
    f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
    print("F1-Macro: {}".format(f1_mac_test))
    print("AUC: {}".format(auc_test))
    print("G-Mean: {}".format(gmean_test))


# 增加一个专门用于超参数敏感性分析的函数
def sensitivity_analysis(config):
    """
    执行超参数敏感性分析，特别是针对epsilon和adv_loss_weight对AUC指标的影响
    """
    print("开始执行超参数敏感性分析...")
    print("分析目标: AUC指标随epsilon和adv_loss_weight的变化")
    
    # 确保配置中有这两个超参数的列表
    if "epsilon" not in config or not isinstance(config["epsilon"], list):
        config["epsilon"] = [0.1, 0.2, 0.3, 0.4, 0.5]  # 默认的epsilon值范围
    
    if "adv_loss_weight" not in config or not isinstance(config["adv_loss_weight"], list):
        config["adv_loss_weight"] = [0.5, 1.0, 1.5, 2.0, 2.5]  # 默认的adv_loss_weight值范围
    
    # 调用多轮运行函数执行网格搜索
    multi_run_main(config, visualize=True)


def multi_run_main(config, visualize=False):
    print_config(config)
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    f1_list, f1_1_list, f1_0_list, auc_list, gmean_list = [], [], [], [], []
    configs = grid(config)
    
    # 用于可视化的配置和结果存储
    experiment_configs = []
    
    for i, cnf in enumerate(configs):
        print("Running {}:\n".format(i))
        for k in hyperparams:
            cnf["save_dir"] += "{}_{}_".format(k, cnf[k])
        print(cnf["save_dir"])
        set_random_seed(cnf["seed"])
        # 确保optimize_threshold参数存在
        if "optimize_threshold" not in cnf:
            cnf["optimize_threshold"] = True
        st = time.time()
        model = ModelHandler(cnf)
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
        f1_list.append(f1_mac_test)
        f1_1_list.append(f1_1_test)
        f1_0_list.append(f1_0_test)
        auc_list.append(auc_test)
        gmean_list.append(gmean_test)
        
        # 保存当前配置，用于可视化
        experiment_configs.append(cnf.copy())
        
        print("Running {} done, elapsed time {}s".format(i, time.time() - st))

    print("F1-Macro: {}".format(f1_list))
    print("AUC: {}".format(auc_list))
    print("G-Mean: {}".format(gmean_list))

    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list, ddof=1)
    f1_1_mean, f1_1_std = np.mean(f1_1_list), np.std(f1_1_list, ddof=1)
    f1_0_mean, f1_0_std = np.mean(f1_0_list), np.std(f1_0_list, ddof=1)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list, ddof=1)
    gmean_mean, gmean_std = np.mean(gmean_list), np.std(gmean_list, ddof=1)

    print("F1-Macro: {}+{}".format(f1_mean, f1_std))
    print("F1-binary-1: {}+{}".format(f1_1_mean, f1_1_std))
    print("F1-binary-0: {}+{}".format(f1_0_mean, f1_0_std))
    print("AUC: {}+{}".format(auc_mean, auc_std))
    print("G-Mean: {}+{}".format(gmean_mean, gmean_std))
    
    # 执行可视化（如果需要）
    if visualize and 'epsilon' in hyperparams and 'adv_loss_weight' in hyperparams:
        print("\n正在生成AUC指标随超参数变化的可视化图表...")
        
        # 准备AUC矩阵
        epsilon_values, adv_loss_weight_values, auc_matrix = prepare_auc_matrix(
            experiment_configs, auc_list
        )
        
        # 创建保存目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config.get("save_dir", "./results"), f"sensitivity_analysis_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成热图
        heatmap_path = os.path.join(save_dir, "auc_heatmap.png")
        plot_auc_heatmap(epsilon_values, adv_loss_weight_values, auc_matrix, heatmap_path)
        
        # 生成3D图
        plot3d_path = os.path.join(save_dir, "auc_3d.png")
        plot_auc_3d(epsilon_values, adv_loss_weight_values, auc_matrix, plot3d_path)
        
        print(f"可视化图表已保存至目录: {save_dir}")
        print("敏感性分析完成！")


# 添加grid函数（如果之前没有定义）
def grid(config):
    """将配置中的列表类型参数展开为网格搜索配置列表"""
    # 找到所有列表类型的参数
    grid_params = {k: v for k, v in config.items() if isinstance(v, list)}
    static_params = {k: v for k, v in config.items() if not isinstance(v, list)}
    
    # 生成所有参数组合
    from itertools import product
    param_combinations = product(*grid_params.values())
    param_names = list(grid_params.keys())
    
    # 创建配置列表
    configs = []
    for combination in param_combinations:
        config = static_params.copy()
        for name, value in zip(param_names, combination):
            config[name] = value
        configs.append(config)
    
    return configs


def print_config(config):
    """打印配置信息"""
    print("\n" + "=" * 50)
    print("配置信息")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r", encoding='utf-8') as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def get_args():
    """
    Get hyperparameters.
    :return: A collection of hyperparameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config",
        type=str,
        default="./config/yelp.yml",
        help="path to the config file",
    )
    parser.add_argument("--multi_run", action="store_true", default=False, help="Run multiple configurations.")
    parser.add_argument("--sensitivity_analysis", action="store_true", default=False, help="Run sensitivity analysis for epsilon and adv_loss_weight.")
    
    args, _ = parser.parse_known_args()
    return vars(args)


if __name__ == "__main__":
    # Get hyperparameters.
    cfg = get_args()
    config = get_config(cfg["config"])
    
    # 如果启用了敏感性分析
    if cfg["sensitivity_analysis"]:
        sensitivity_analysis(config)
    # 如果启用了多轮运行
    elif cfg["multi_run"]:
        multi_run_main(config)
    # 否则单轮运行
    else:
        main(config)