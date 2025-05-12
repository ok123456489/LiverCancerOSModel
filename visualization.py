import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_metrics(metrics_log, save_path=None):
    """
    绘制训练过程中的损失和评估指标曲线
    Args:
        metrics_log: 训练过程中记录的指标字典
        save_path: 图片保存路径，如果为None则不保存
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(metrics_log['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制C-index曲线
    plt.subplot(1, 2, 2)
    plt.plot(metrics_log['cindex'], label='C-index', color='orange')
    plt.title('Training C-index')
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_ablation_results(final_results, save_path=None):
    """
    绘制消融实验结果对比图
    Args:
        final_results: 消融实验结果字典
        save_path: 图片保存路径，如果为None则不保存
    """
    # 处理元组输入
    if isinstance(final_results, tuple):
        final_results = {f"Config {idx + 1}": res for idx, res in enumerate(final_results)}

    config_names = [f"Config {idx + 1}" for idx in range(len(final_results))]
    cindex_means = [res['mean_cindex'] for res in final_results.values()]
    cindex_stds = [res['std_cindex'] for res in final_results.values()]

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(config_names))

    plt.barh(y_pos, cindex_means, xerr=cindex_stds, align='center', alpha=0.7)
    plt.yticks(y_pos, config_names)
    plt.xlabel('C-index (mean ± std)')
    plt.title('Ablation Study Results')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_slice_attention(attention_dict, save_path=None):
    """
    绘制切片注意力权重图
    Args:
        attention_dict: 注意力权重字典 {序列名: 权重值}
        save_path: 图片保存路径，如果为None则不保存
    """
    sequences = list(attention_dict.keys())
    weights = list(attention_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(sequences, weights)
    plt.title('Average Slice Attention Weights')
    plt.ylabel('Attention Weight')
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()