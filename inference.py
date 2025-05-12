import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import LiverSurvivalDataset
from survival_trainer import SurvivalTrainer
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor, Lambda
import os
from tqdm import tqdm
from glob import glob


class SurvivalPredictor:
    def __init__(self, model_checkpoint_paths, device=None):
        """
        初始化预测器，支持加载多个k-fold模型
        :param model_checkpoint_paths: 模型检查点文件路径列表或包含k-fold模型的目录
        :param device: 使用的设备 (cuda/cpu)，默认为自动选择
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 处理模型路径输入
        if isinstance(model_checkpoint_paths, str):
            # 如果是目录路径，查找所有.pth文件
            if os.path.isdir(model_checkpoint_paths):
                self.model_paths = sorted(glob(os.path.join(model_checkpoint_paths, '*.pth')))
            else:
                # 如果是单个文件路径
                self.model_paths = [model_checkpoint_paths]
        elif isinstance(model_checkpoint_paths, list):
            self.model_paths = model_checkpoint_paths
        else:
            raise ValueError(
                "model_checkpoint_paths should be a directory path, a single model path, or a list of model paths")

        # 加载所有模型
        self.models = []
        for path in self.model_paths:
            model_info = SurvivalTrainer.load_model(path, self.device)
            self.models.append(model_info['model'])

        # 初始化数据预处理
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """获取图像预处理转换"""
        return Compose([
            # 1. 加载图像（强制转为灰度）
            LoadImage(image_only=True, reader='PILReader', dtype=np.float32),
            # 2. 显式转为单通道灰度
            Lambda(self._ensure_grayscale),
            # 3. 强度归一化
            ScaleIntensity(minv=0.0, maxv=1.0),
            # 4. 调整尺寸（2D处理）
            Resize(spatial_size=(224, 224), mode='bilinear'),
            # 5. 转为Tensor
            ToTensor()
        ])

    @staticmethod
    def _ensure_grayscale(x):
        """确保输入图像为单通道灰度格式"""
        if x.ndim > 2:  # 如果是RGB或多通道图像
            return x.mean(axis=0, keepdims=True) if x.shape[0] > 1 else x
        else:  # 如果是单通道灰度 (H,W)
            return x[None, ...]  # 增加通道维度 -> (1,H,W)

    def predict_from_dataset(self, dataset, batch_size=4, clinical_data=None):
        """
        从数据集批量预测，使用所有k-fold模型并取均值
        :param dataset: LiverSurvivalDataset实例或类似结构
        :param batch_size: 批大小
        :param clinical_data: 可选的临床数据DataFrame，如果dataset不包含临床数据
        :return: 包含预测结果的DataFrame
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 存储每个模型的预测结果
        all_results = []

        for model in self.models:
            model.eval()
            results = []
            with torch.no_grad():
                for batch in tqdm(dataloader,
                                  desc=f"Predicting with model {self.models.index(model) + 1}/{len(self.models)}"):
                    # 准备输入数据
                    mr_inputs = batch['mr'].to(self.device)

                    # 处理临床数据
                    if 'clinical' in batch:
                        clinical_input = batch['clinical'].to(self.device)
                    elif clinical_data is not None:
                        # 从外部DataFrame获取临床数据
                        pids = batch['patient_id']
                        clin_data = clinical_data.loc[clinical_data['patient_id'].isin(pids)]
                        clinical_input = torch.FloatTensor(
                            clin_data.drop(['patient_id', 'os', 'event'], axis=1).values
                        ).to(self.device)
                    else:
                        clinical_input = None

                    # 预测
                    quantiles, risk = model(mr_inputs, clinical_input)

                    # 收集结果
                    for i in range(len(batch['patient_id'])):
                        results.append({
                            'patient_id': batch['patient_id'][i].item(),
                            'lower_quantile': quantiles[i, 0].item(),
                            'median_survival': quantiles[i, 1].item(),
                            'upper_quantile': quantiles[i, 2].item(),
                            'risk_score': risk[i].item(),
                            'true_time': batch.get('time', [None] * len(batch['patient_id']))[i].item(),
                            'event': batch.get('event', [None] * len(batch['patient_id']))[i].item()
                        })

            all_results.append(pd.DataFrame(results))

        # 合并所有模型的预测结果
        final_results = self._average_predictions(all_results)
        return final_results

    def _average_predictions(self, all_results):
        """
        对多个模型的预测结果取平均
        :param all_results: 包含所有模型预测结果的DataFrame列表
        :return: 平均后的预测结果DataFrame
        """
        # 确保所有DataFrame有相同的patient_id顺序
        patient_ids = all_results[0]['patient_id']
        for df in all_results[1:]:
            assert (df['patient_id'] == patient_ids).all(), "Patient IDs must match across all predictions"

        # 计算每个指标的均值
        avg_results = []
        for i in range(len(patient_ids)):
            patient_data = {
                'patient_id': patient_ids[i],
                'lower_quantile': np.mean([df.iloc[i]['lower_quantile'] for df in all_results]),
                'median_survival': np.mean([df.iloc[i]['median_survival'] for df in all_results]),
                'upper_quantile': np.mean([df.iloc[i]['upper_quantile'] for df in all_results]),
                'risk_score': np.mean([df.iloc[i]['risk_score'] for df in all_results]),
                'true_time': all_results[0].iloc[i]['true_time'],  # 这些值在所有模型中相同
                'event': all_results[0].iloc[i]['event']  # 这些值在所有模型中相同
            }
            avg_results.append(patient_data)

        return pd.DataFrame(avg_results)

    def predict_single(self, mr_data, clinical_data=None):
        """
        预测单个样本，使用所有k-fold模型并取均值
        :param mr_data: MRI数据 [num_sequences, num_slices, height, width] 或文件路径列表
        :param clinical_data: 临床数据数组 [clinical_dim] 或DataFrame行
        :return: 预测结果字典
        """
        # 预处理MRI数据
        if isinstance(mr_data, list):  # 如果是文件路径
            mr_tensor = self._load_mr_from_files(mr_data)
        else:  # 如果是numpy数组
            mr_tensor = torch.from_numpy(mr_data).float()

        # 增加batch维度 [1, num_sequences, num_slices, 1, 224, 224]
        mr_tensor = mr_tensor.unsqueeze(0).to(self.device)

        # 预处理临床数据
        if clinical_data is not None:
            if isinstance(clinical_data, pd.Series):  # 如果是DataFrame行
                clinical_tensor = torch.FloatTensor(
                    clinical_data.drop(['patient_id', 'os', 'event']).values
                ).unsqueeze(0).to(self.device)
            else:  # 如果是numpy数组
                clinical_tensor = torch.from_numpy(clinical_data).float().unsqueeze(0).to(self.device)
        else:
            clinical_tensor = None

        # 存储所有模型的预测结果
        all_quantiles = []
        all_risks = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                quantiles, risk = model(mr_tensor, clinical_tensor)
                all_quantiles.append(quantiles)
                all_risks.append(risk)

        # 计算均值
        avg_quantiles = torch.mean(torch.stack(all_quantiles), dim=0)
        avg_risk = torch.mean(torch.stack(all_risks), dim=0)

        return {
            'lower_quantile': avg_quantiles[0, 0].item(),
            'median_survival': avg_quantiles[0, 1].item(),
            'upper_quantile': avg_quantiles[0, 2].item(),
            'risk_score': avg_risk[0].item()
        }

    def _load_mr_from_files(self, file_paths):
        """
        从文件路径加载MRI数据
        :param file_paths: 文件路径列表，按[seq1_slice1, seq1_slice2,..., seqN_sliceM]顺序
        :return: 处理后的MRI tensor [num_sequences, num_slices, 1, 224, 224]
        """
        # 重组文件路径为[num_sequences, num_slices]
        num_slices = 3
        seq_files = [file_paths[i:i + num_slices] for i in range(0, len(file_paths), num_slices)]

        mr_data = []
        for seq in seq_files:
            slices = []
            for slice_path in seq:
                img = self.transform(slice_path)
                slices.append(img)
            mr_data.append(torch.stack(slices, dim=0))

        return torch.stack(mr_data)


# 使用示例
if __name__ == "__main__":
    # 1. 初始化预测器 - 加载所有k-fold模型
    predictor = SurvivalPredictor(
        model_checkpoint_paths="./data/ablation_results/"  # 包含所有k-fold模型的目录
    )

    # 2. 准备数据
    test_dataset = LiverSurvivalDataset(
        data_root="./data",
        clinical_excel_path="./data/clinical_info.xlsx",
        mri_sequences=['AP', 'VP', 'DP', 'T2'],
        num_slices=3
    )

    # 批量预测（会自动使用所有k-fold模型并取均值）
    batch_results = predictor.predict_from_dataset(test_dataset, batch_size=8)
    print("\nAverage prediction results from all k-fold models:")
    print(batch_results.head())

    # 保存结果
    batch_results.to_csv("average_predictions.csv", index=False)

    # # 3. 单个样本预测示例
    # # 示例1: 使用numpy数组
    # example_mr = np.random.rand(5, 3, 224, 224)  # [num_sequences, num_slices, height, width]
    # example_clinical = np.random.rand(42)  # [clinical_dim]
    #
    # single_result = predictor.predict_single(example_mr, example_clinical)
    # print("\nSingle prediction result (average from all models):")
    # print(f"Median survival time: {single_result['median_survival']:.2f} days")
    # print(f"Risk score: {single_result['risk_score']:.4f}")