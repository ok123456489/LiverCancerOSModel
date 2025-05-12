import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score
from torch import device
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Evaluate(model, dataloader, eval_times=[12, 24, 36]):
    """评估函数（支持C-index和多个时间点的AUC）"""
    model.eval()  # 将模型设置为评估模式
    pred_quantiles, pred_risks, true_times, events = [], [], [], []
    device = next(model.parameters()).device  # 获取模型所在的设备

    with torch.no_grad():
        for batch in dataloader:
            quantiles, risk = model(batch['mr'].to(device), batch['clinical'].to(device))
            pred_quantiles.append(quantiles.cpu())
            pred_risks.append(risk.cpu())
            true_times.append(batch['time'].cpu())
            events.append(batch['event'].cpu())

    # 合并结果
    pred_quantiles = torch.cat(pred_quantiles)
    pred_risks = torch.cat(pred_risks)
    true_times = torch.cat(true_times)
    events = torch.cat(events)

    # 计算指标
    cindex = concordance_index_censored(
        events.numpy().astype(bool),  # 事件标记
        true_times.numpy(),  # 生存时间
        -pred_risks.numpy())[0]  # 风险分越高生存期越短

    metrics = {
        'MAE': F.l1_loss(pred_quantiles[:, 1], true_times).item(),
        'Coverage': ((true_times >= pred_quantiles[:, 0]) & (true_times <= pred_quantiles[:, 2])).float().mean().item(),
        'c_index': cindex,
    }

    # 为每个 eval_time 计算 AUC
    for eval_time in eval_times:
        auc_mask = (true_times < eval_time) & (events == 1)
        if auc_mask.sum() > 0:  # 确保有足够的正例来计算 AUC
            auc = roc_auc_score(auc_mask.numpy(), -pred_risks.numpy())
            metrics[f'auc_{eval_time}m'] = auc
        else:
            metrics[f'auc_{eval_time}m'] = None  # 或者设置为 0 或其他默认值

    return metrics

def visualize_slice_attention(model, dataloader, seq_names):
    model.eval()
    attention_maps = {seq: [] for seq in seq_names}
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch['mr'])
            attn = model.get_slice_attention()
            if attn is not None:
                for i, seq in enumerate(seq_names):
                    attention_maps[seq].extend(attn[:,i].tolist())
    avg_attention = {seq: np.mean(vals) for seq, vals in attention_maps.items()}
    return avg_attention