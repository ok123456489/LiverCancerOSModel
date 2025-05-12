import os
import torch
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from model import SurvivalModel
from data_loader import LiverSurvivalDataset
from survival_trainer import SurvivalTrainer
from utils.matrics import Evaluate
from sklearn.model_selection import StratifiedKFold  ###


def check_class_distribution(dataset, indices, name=""):
    # 获取生存时间和事件状态
    times = [dataset[i]['time'] for i in indices]
    events = [dataset[i]['event'] for i in indices]

    print(f"\n{name} set distribution:")
    print(f"  Samples: {len(indices)}")
    print(f"  Events: {sum(events)} ({sum(events) / len(events):.1%})")
    print(f"  Median survival time: {np.median(times):.1f} days")


def create_stratification_labels(dataset, indices):
    """创建用于分层抽样的复合标签"""
    # 获取事件状态和时间分位数(分为3组)
    events = np.array([dataset[i]['event'] for i in indices])
    times = np.array([dataset[i]['time'] for i in indices])
    time_quantiles = np.quantile(times, [0.33, 0.66])
    time_strata = np.digitize(times, time_quantiles)

    # 创建复合标签: event_status + time_strata
    return [f"{int(e)}_{int(t)}" for e, t in zip(events, time_strata)]


def run_ablation(data_root, clinical_file, mri_sequences, num_slices, batch_size=8, k_folds=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    full_dataset = LiverSurvivalDataset(
        data_root=data_root,
        clinical_excel_path=f"{data_root}/{clinical_file}",
        mri_sequences=mri_sequences,
        num_slices=num_slices
    )

    configs = [
        {'slice_fusion': 'avgpool', 'seq_fusion': 'concat', 'use_clinical': True, 'use_snn': True, 'batch_size': 8,
         'lr': 2e-4, 'epochs': 100, 'early_stop': 20}
        # {'slice_fusion': 'attention', 'seq_fusion': 'concat', 'use_clinical': True, 'use_snn': True, 'batch_size': 8, 'lr': 2e-4, 'epochs': 100, 'early_stop': 15},
        # {'slice_fusion': 'attention', 'seq_fusion': 'dynamic', 'use_clinical': True, 'use_snn': True, 'batch_size': 8, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10},
        # {'slice_fusion': 'attention', 'seq_fusion': 'transformer', 'use_clinical': True, 'use_snn': True, 'batch_size': 8, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10},
        # {'slice_fusion': 'concat', 'seq_fusion': 'concat', 'use_clinical': True, 'use_snn': True, 'batch_size': 8, 'lr': 2e-4, 'epochs': 100, 'early_stop': 15},
        # {'slice_fusion': 'concat', 'seq_fusion': 'dynamic', 'use_clinical': True, 'use_snn': True, 'batch_size': 8, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10},
        # {'slice_fusion': 'concat', 'seq_fusion': 'transformer', 'use_clinical': True, 'use_snn': True, 'batch_size': 8, 'lr': 2e-4, 'epochs': 50, 'early_stop': 10}

    ]

    final_results = defaultdict(list)
    all_training_logs = {}  # 新增：保存所有训练日志

    # 创建输出目录
    output_dir = os.path.join(data_root, 'ablation_results1')
    os.makedirs(output_dir, exist_ok=True)

    # 确保数据集能提供 patient_ids（每个样本对应一个病人ID）
    # patient_ids = full_dataset.patient_ids  # 需要 LiverSurvivalDataset 支持此属性
    # events = full_dataset.events
    # kf = KFold(n_splits=k_folds,shuffle=True,random_state=seed)
    # patient_indices = np.arange(len(full_dataset))
    patient_indices = np.arange(len(full_dataset))
    test_size_absolute = 27  # 每折测试集的绝对大小
    total_samples = len(patient_indices)
    test_size_proportion = test_size_absolute / total_samples  # 计算测试集比例
    # kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    kf = StratifiedShuffleSplit(n_splits=k_folds, test_size=test_size_proportion, random_state=42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 创建分层标签
    strat_labels = create_stratification_labels(full_dataset, patient_indices)
    # skf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for cfg_idx, cfg in enumerate(configs):
        print(f"\n=== Testing Config: {cfg} ===")
        fold_metrics = defaultdict(list)
        cfg_training_logs = []  # 新增：保存当前配置的训练日志

        # 为当前配置创建子目录
        cfg_dir = os.path.join(output_dir, f"config_{cfg_idx}")
        os.makedirs(cfg_dir, exist_ok=True)

        # for fold, (train_idx, test_idx) in enumerate(kf.split(patient_indices)):
        for fold, (train_idx, test_idx) in enumerate(kf.split(patient_indices, strat_labels)):
            print(f"\nFold {fold + 1}/{k_folds}:")
            check_class_distribution(full_dataset, train_idx, f"Fold {fold + 1} Train")
            check_class_distribution(full_dataset, test_idx, f"Fold {fold + 1} Test")
            train_set = Subset(full_dataset, train_idx)
            test_set = Subset(full_dataset, test_idx)
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.get('batch_size', batch_size),
                shuffle=True,
                num_workers=8)
            test_loader = DataLoader(
                test_set,
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

            metrics = Evaluate(best_model, test_loader)

            fold_metrics['train_loss'].append(np.min(train_log['train_loss']))
            fold_metrics['MAE'].append(metrics['MAE'])
            fold_metrics['Coverage'].append(metrics['Coverage'])
            fold_metrics['test_cindex'].append(metrics['c_index'])
            fold_metrics['test_auc_12m'].append(metrics.get('auc_12m', 0))
            fold_metrics['test_auc_24m'].append(metrics.get('auc_24m', 0))
            fold_metrics['test_auc_36m'].append(metrics.get('auc_36m', 0))
            print(
                f"MAE: {metrics['MAE']:.4f} | "
                f"Coverage: {metrics['Coverage']:.4f} | "
                f"C-index: {metrics['c_index']:.3f} | "
                f"AUC_12m: {metrics.get('auc_12m', 0):.3f} | "
                f"AUC_24m: {metrics.get('auc_24m', 0):.3f} | "
                f"AUC_36m: {metrics.get('auc_36m', 0):.3f}")
        final_results[str(cfg)] = {
            'mean_train_loss': np.mean(fold_metrics['train_loss']),
            'mean_cindex': np.mean(fold_metrics['test_cindex']),
            'mean_auc_12m': np.mean(fold_metrics['test_auc_12m']),
            'mean_auc_24m': np.mean(fold_metrics['test_auc_24m']),
            'mean_auc_36m': np.mean(fold_metrics['test_auc_36m']),
            'mean_MAE': np.mean(fold_metrics['MAE']),
            'mean_Coverage': np.mean(fold_metrics['Coverage']),
            'std_cindex': np.std(fold_metrics['test_cindex']),
            'std_auc_12m': np.std(fold_metrics['test_auc_12m']),
            'std_auc_24m': np.std(fold_metrics['test_auc_24m']),
            'std_auc_36m': np.std(fold_metrics['test_auc_36m']),
            'fold_details': fold_metrics,
            'config_dir': cfg_dir  # 保存配置的目录路径
        }
        all_training_logs[str(cfg)] = cfg_training_logs  # 保存当前配置的所有fold训练日志
    print("\n== Results Summary ===")
    for config, res in sorted(final_results.items(), key=lambda x: -x[1]['mean_cindex']):
        print(
            f"{config[:80]:<80} | "
            f"Loss: {res['mean_train_loss']:.4f} | "
            f"MAE: {res['mean_MAE']:.4f} | "
            f"Coverage: {res['mean_Coverage']:.4f} | "
            f"C-index: {res['mean_cindex']:.3f}\pm{res['std_cindex']:.3f} | "
            f"AUC_12m: {res['mean_auc_12m']:.3f}\pm{res['std_auc_12m']:.3f} | "
            f"AUC_24m: {res['mean_auc_24m']:.3f}\pm{res['std_auc_24m']:.3f} | "
            f"AUC_36m: {res['mean_auc_36m']:.3f}\pm{res['std_auc_36m']:.3f}")
    return final_results, all_training_logs