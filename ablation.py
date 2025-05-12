import os

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from model import SurvivalModel
from data_loader import LiverSurvivalDataset
import seaborn as sns


from survival_trainer import SurvivalTrainer
from utils.matrics import Evaluate


def check_class_distribution(dataset, indices, name=""):
    # 获取生存时间和事件状态
    times = [dataset[i]['time'] for i in indices]
    events = [dataset[i]['event'] for i in indices]

    print(f"\n{name} set distribution:")
    print(f"  Samples: {len(indices)}")
    print(f"  Events: {sum(events)} ({sum(events) / len(events):.1%})")
    print(f"  Median survival time: {np.median(times):.1f} days")


def run_ablation(train_data_root, test_data_root, clinical_file, mri_sequences, num_slices, batch_size=8, k_folds=10,
                 seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载训练集和测试集
    train_dataset = LiverSurvivalDataset(
        data_root=train_data_root,
        clinical_excel_path=f"{train_data_root}/{clinical_file}",
        mri_sequences=mri_sequences,
        num_slices=num_slices
    )

    test_dataset = LiverSurvivalDataset(
        data_root=test_data_root,
        clinical_excel_path=f"{test_data_root}/{clinical_file}",
        mri_sequences=mri_sequences,
        num_slices=num_slices
    )

    configs = [
        #{'slice_fusion': 'avgpool', 'seq_fusion': 'concat', 'use_clinical': True, 'use_snn': True, 'batch_size': 8,'lr': 2e-4, 'epochs': 100, 'early_stop': 15},
         {'slice_fusion': 'attention', 'seq_fusion': 'concat', 'use_clinical': True, 'use_snn': True, 'batch_size': 8, 'lr': 2e-4, 'epochs': 100, 'early_stop': 15},
        # {'slice_fusion': 'attention', 'seq_fusion': 'dynamic', 'use_clinical': True, 'use_snn': True, 'batch_size': 16, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10},
        # {'slice_fusion': 'attention', 'seq_fusion': 'transformer', 'use_clinical': True, 'use_snn': True, 'batch_size': 16, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10},
        # {'slice_fusion': 'concat', 'seq_fusion': 'concat', 'use_clinical': True, 'use_snn': True, 'batch_size': 16, 'lr': 2e-4, 'epochs': 100, 'early_stop': 15},
        # {'slice_fusion': 'concat', 'seq_fusion': 'dynamic', 'use_clinical': True, 'use_snn': True, 'batch_size': 16, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10},
        # {'slice_fusion': 'concat', 'seq_fusion': 'transformer', 'use_clinical': True, 'use_snn': True, 'batch_size': 16, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10}
    ]

    final_results = defaultdict(list)
    all_training_logs = {}  # 新增：保存所有训练日志

    # 创建输出目录
    output_dir = os.path.join(train_data_root, 'ablation_results1')
    os.makedirs(output_dir, exist_ok=True)

    # 只在训练集上进行K折划分
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    train_indices = np.arange(len(train_dataset))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for cfg_idx, cfg in enumerate(configs):
        print(f"\n=== Testing Config: {cfg} ===")
        fold_metrics = defaultdict(list)
        cfg_training_logs = []  # 新增：保存当前配置的训练日志

        # 为当前配置创建子目录
        cfg_dir = os.path.join(output_dir, f"config_{cfg_idx}")
        os.makedirs(cfg_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
            print(f"\nFold {fold + 1}/{k_folds}:\n")
            check_class_distribution(train_dataset, train_idx, f"Fold {fold + 1} Train")
            check_class_distribution(test_dataset, range(len(test_dataset)), "Test")

            # 创建训练集和验证集
            train_set = Subset(train_dataset, train_idx)
            val_set = Subset(train_dataset, val_idx)

            # 测试集保持不变
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.get('batch_size', batch_size),
                shuffle=False,
                num_workers=8)

            train_loader = DataLoader(
                train_set,
                batch_size=cfg.get('batch_size', batch_size),
                shuffle=True,
                num_workers=8)

            val_loader = DataLoader(
                val_set,
                batch_size=cfg.get('batch_size', batch_size),
                shuffle=False,
                num_workers=8)

            model = SurvivalModel(
                slice_fusion_type=cfg['slice_fusion'],
                seq_fusion_type=cfg['seq_fusion'],
                backbone='resnet18',
                use_SNN=cfg['use_snn']).to(device)
            model.train()

            trainer = SurvivalTrainer(
                model=model,
                train_data=train_set,
                config={
                    'batch_size': cfg['batch_size'],
                    'epochs': cfg['epochs'],
                    'lr': cfg['lr'],
                    'early_stop': cfg['early_stop'],
                    'output_dir': cfg_dir,
                    'model_id': f"cfg_{cfg_idx}_fold_{fold}",  # 唯一标识符
                })

            train_log = trainer.run_training()
            cfg_training_logs.append(train_log)  # 保存训练日志

            # 加载最佳模型进行评估
            best_model = SurvivalTrainer.load_model(
                os.path.join(cfg_dir, f"best_cfg_{cfg_idx}_fold_{fold}.pth"),
                device=device
            )['model']

            # 使用测试集测试
            metrics_test = Evaluate(best_model, test_loader)

            fold_metrics['train_loss'].append(np.min(train_log['train_loss']))
            fold_metrics['MAE'].append(metrics_test['MAE'])
            fold_metrics['Coverage'].append(metrics_test['Coverage'])
            fold_metrics['test_cindex'].append(metrics_test['c_index'])
            fold_metrics['test_auc_12m'].append(metrics_test.get('auc_12m', 0))
            fold_metrics['test_auc_24m'].append(metrics_test.get('auc_24m', 0))
            fold_metrics['test_auc_36m'].append(metrics_test.get('auc_36m', 0))

            cindex_median = np.median(fold_metrics['test_cindex'])
            cindex_q1 = np.percentile(fold_metrics['test_cindex'], 25)
            cindex_q3 = np.percentile(fold_metrics['test_cindex'], 75)

            print(
                f"MAE: {np.mean(fold_metrics['MAE']):.4f} | "
                f"Coverage: {np.mean(fold_metrics['Coverage']):.4f} | "
                f"C-index: Median={cindex_median:.3f} (IQR {cindex_q1:.3f}-{cindex_q3:.3f}) | "
                f"AUC_12m: {np.mean(fold_metrics['test_auc_12m']):.3f} | "
                f"AUC_24m: {np.mean(fold_metrics['test_auc_24m']):.3f}| "
                f"AUC_36m: {np.mean(fold_metrics['test_auc_36m']):.3f}")

        final_results[str(cfg)] = {
            'mean_train_loss': np.mean(fold_metrics['train_loss']),
            'mean_cindex': np.mean(fold_metrics['test_cindex']),
            'median_cindex': np.median(fold_metrics['test_cindex']),
            'mean_auc_12m': np.mean(fold_metrics['test_auc_12m']),
            'mean_auc_24m': np.mean(fold_metrics['test_auc_24m']),
            'mean_auc_36m': np.mean(fold_metrics['test_auc_36m']),
            'mean_MAE': np.mean(fold_metrics['MAE']),
            'mean_Coverage': np.mean(fold_metrics['Coverage']),
            'std_cindex': np.std(fold_metrics['test_cindex']),
            'cindex_q1': np.percentile(fold_metrics['test_cindex'], 25),
            'cindex_q3': np.percentile(fold_metrics['test_cindex'], 75),
            'std_auc_12m': np.std(fold_metrics['test_auc_12m']),
            'std_auc_24m': np.std(fold_metrics['test_auc_24m']),
            'std_auc_36m': np.std(fold_metrics['test_auc_36m']),
            'fold_details': fold_metrics,
            'config_dir': cfg_dir  # 保存配置的目录路径
        }

        all_training_logs[str(cfg)] = cfg_training_logs  # 保存当前配置的所有fold训练日志

    print("\n== Results Summary ===\n")
    print("\nTest DataSet Result\n")
    for config, res in sorted(final_results.items(), key=lambda x: -x[1]['mean_cindex']):
        print(
            f"{config[:80]:<80} | "
            f"Loss: {res['mean_train_loss']:.4f}(train)/{res['mean_val_loss']:.4f}(val) | "
            f"MAE: {res['mean_MAE']:.4f} | "
            f"Coverage: {res['mean_Coverage']:.4f} | "
            f"C-index: {res['mean_cindex']:.3f}\pm{res['std_cindex']:.3f} | "
            f"Median(IQR)_C-index: Median={res['median_cindex']:.3f} (IQR {res['cindex_q1']:.3f}-{res['cindex_q3']:.3f}) | "
            f"AUC_12m: {res['mean_auc_12m']:.3f}\pm{res['std_auc_12m']:.3f} | "
            f"AUC_24m: {res['mean_auc_24m']:.3f}\pm{res['std_auc_24m']:.3f} | "
            f"AUC_36m: {res['mean_auc_36m']:.3f}\pm{res['std_auc_36m']:.3f}")

    return final_results, all_training_logs