import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensity, Resize, ToTensor, Lambda
)

class LiverSurvivalDataset(Dataset):
    def __init__(self, data_root, clinical_excel_path,
                 mri_sequences=['arterial', 'venous', 'delay', 't2', 'dwi'],
                 num_slices=3, transform=None):
        """
        Args:
            data_root: 包含patientsid(如patients_001)目录的根路径
            clinical_excel_path: 包含临床数据、生存时间和事件的EXCEL文件，假设临床数据已进行过预处理
            mri_sequences: 使用的MRI序列列表
            num_slices: 每个序列使用的切片数量
            transform: 可选的图像增强
        """
        # 加载临床数据
        clinical_df = pd.read_excel(clinical_excel_path)
        self.patient_ids = clinical_df['patient_id'].tolist()
        self.times = torch.FloatTensor(clinical_df['os'].values)
        self.events = torch.FloatTensor(clinical_df['event'].values)
        self.clinical_data = torch.FloatTensor(
            clinical_df.drop(['patient_id', 'os', 'event'], axis=1).values
        )
        # 参数校验
        #self.mri_sequences = [seq.lower() for seq in mri_sequences]  # 统一小写
        self.mri_sequences = [seq for seq in mri_sequences]
        self.num_slices = num_slices

        # 构建MRI文件路径索引
        self.mr_paths = OrderedDict()  # {patient_id: [ [seq1_img1, seq1_img2, seq1_img3], ... ]}
        for pid in self.patient_ids:
            patient_dir = os.path.join(data_root, 'mri_images', str(pid))
            self.mr_paths[pid] = []
            for seq in self.mri_sequences:
                seq_dir = os.path.join(patient_dir, seq)
                if not os.path.exists(seq_dir):
                    raise FileNotFoundError(f"序列目录不存在：{seq_dir}")
                # 按固定顺序加载切片
                slices = sorted([
                    os.path.join(seq_dir, f)
                    for f in os.listdir(seq_dir)
                    if f.endswith('.png')
                ], key=lambda x: int(x.split('slice_')[-1].split('.')[0]))
                if len(slices) < self.num_slices:
                    raise ValueError(f"{pid}/{seq} 切片不足，预期{self.num_slices}张，实际{len(slices)}张")
                self.mr_paths[pid].append(slices[:self.num_slices])

        # 图像预处理


        #todo :调monai库

        self.transform = Compose([
        # 1. 加载图像（强制转为灰度）
        LoadImage(image_only=True, reader='PILReader', dtype=np.float32),

        # 2. 显式转为单通道灰度（如果输入是RGB，取均值；如果已经是灰度则无操作）
        Lambda(_ensure_grayscale),

        # 3. 强度归一化
        ScaleIntensity(minv=0.0, maxv=1.0),

        # 4. 调整尺寸（2D处理）
        Resize(spatial_size=(224, 224), mode='bilinear'),

        # 5. 转为Tensor
        ToTensor()
    ])


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        # 加载MRI数据
        mr_data = []
        for seq_slices in self.mr_paths[pid]:  # 每个序列的切片
            slices = []
            for slice_path in seq_slices:
                img = self.transform(slice_path)
                slices.append(img)
            # 调整维度顺序为 [num_slices, channels, height, width]
            mr_data.append(torch.stack(slices, dim=0))
        # 最终形状: [num_sequences, num_slices, channels, height, width]
        mr_data = torch.stack(mr_data)  # [5, 3, 1, 224, 224]

        return {
            'mr': mr_data.float(),
            'clinical': self.clinical_data[idx],
            'time': self.times[idx],
            'event': self.events[idx],
            'patient_id': pid
        }
def _ensure_grayscale(x):
    """
    确保输入图像为单通道灰度格式
    Args:
        x: 输入图像 (numpy.ndarray 或 torch.Tensor)
            - 如果是 RGB (H,W,3 或 3,H,W): 取均值转为灰度 (1,H,W)
            - 如果是灰度 (H,W) 或 (1,H,W): 直接转为 (1,H,W)
    Returns:
        单通道灰度图像 (1,H,W)
    """
    if x.ndim > 2:  # 如果是RGB或多通道图像
        # 取均值转为灰度 (保持维度)
        return x.mean(axis=0, keepdims=True) if x.shape[0] > 1 else x
    else:  # 如果是单通道灰度 (H,W)
        return x[None, ...]  # 增加通道维度 -> (1,H,W)