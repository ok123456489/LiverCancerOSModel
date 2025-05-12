import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet
from torchvision import models
from model_utils import SNN_Block

class SliceFusion(nn.Module):
    """支持多种切片融合方式的模块"""
    def __init__(self, fusion_type='attention', mr_feat_dim=512, use_snn=True):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'attention':
            self.attn = nn.Sequential(
                nn.Linear(mr_feat_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        elif fusion_type == 'concat':
            self.proj = nn.Linear(mr_feat_dim*3, mr_feat_dim)

    def forward(self, slices):
        """输入: [B, num_slices, mr_feat_dim]"""
        if self.fusion_type == 'attention':
            attn = F.softmax(self.attn(slices), dim=1)
            self.last_attention = attn
            return (attn * slices).sum(dim=1)
        elif self.fusion_type == 'concat':
            return self.proj(slices.flatten(1))
        elif self.fusion_type == 'maxpool':
            return slices.max(dim=1)[0]
        elif self.fusion_type == 'avgpool':
            return slices.mean(dim=1)

class DynamicFusion(nn.Module):
    """基于临床特征的动态权重融合"""
    def __init__(self, clinical_dim, num_seq):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_seq),
            nn.Softmax(dim=1))

    def forward(self, seq_features, clinical_data):
        weights = self.weight_net(clinical_data)
        weighted_feats = torch.stack([
            weights[:,i].unsqueeze(1)*feat for i, feat in enumerate(seq_features)
        ]).sum(0)
        return weighted_feats

class SequenceFusion(nn.Module):
    """支持多种序列融合方式的模块"""
    def __init__(self, fusion_type='concat', num_seq=5, mr_feat_dim=512, clinical_dim=None):
        super().__init__()
        self.fusion_type = fusion_type
        #todo:
        #当fusion_type，下面这个参数是否设置为None
        self.fusion = None
        if fusion_type == 'bilinear':
            self.fusion = nn.Bilinear(mr_feat_dim, mr_feat_dim, mr_feat_dim)
        elif fusion_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=mr_feat_dim, nhead=4)
            self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif fusion_type == 'dynamic' and clinical_dim is not None:
            self.fusion = DynamicFusion(clinical_dim, num_seq)

    def forward(self, seq_features, clinical_data=None):
        """输入: list of [B, nr_feat_dim]"""
        if self.fusion_type == 'concat':
            fused = torch.cat(seq_features, dim=1)
            return fused
        elif self.fusion_type == 'bilinear':
            fused = seq_features[0]
            for feat in seq_features[1:]:
                fused = self.fusion(fused, feat)
            return fused
        elif self.fusion_type == 'transformer':
            stacked = torch.stack(seq_features, dim=1)
            return self.fusion(stacked).mean(dim=1)
        elif self.fusion_type == 'dynamic':
            return self.fusion(seq_features, clinical_data)

class SurvivalModel(nn.Module):
    def __init__(self,
                 slice_fusion_type='attention',
                 seq_fusion_type='concat',
                 use_clinical=True,
                 clinical_dim=41,
                 backbone='resnet18',
                 use_SNN=True,
                 quantiles=[0.025, 0.5, 0.0975]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        self.seq_fusion_type = seq_fusion_type
        self.use_clinical = use_clinical

        #todo:使用monai的resnet

        # 共享基础特征提取器
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            #todo： resnet50和resnet18输出维度不一样
            self.mr_feat_dim = 2048  # ResNet50输出2048维
        else:
            resnet = models.resnet18(pretrained=True)
            self.mr_feat_dim = 512  # ResNet18输出512维
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.slice_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # 消融实验配置
        self.slice_fusion = SliceFusion(slice_fusion_type)
        self.slice_fusion_type = slice_fusion_type
        self.seq_fusion = SequenceFusion(
            fusion_type=self.seq_fusion_type,
            num_seq=5,
            mr_feat_dim=self.mr_feat_dim,
            clinical_dim=clinical_dim if self.seq_fusion_type == 'dynamic' else None)

        if use_SNN:
            self.clinical_net = nn.Sequential(
                SNN_Block(clinical_dim, 256),
                SNN_Block(256, 128))
        else:
            self.clinical_net = nn.Sequential(
                nn.Linear(clinical_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128))

        # 动态计算维度
        seq_out_dim = self.mr_feat_dim * 4 if self.seq_fusion_type == 'concat' else self.mr_feat_dim
        final_in_dim = seq_out_dim + (128 if use_clinical else 0)


        # 预测头
        if use_SNN:
            self.surv_net = nn.Sequential(
                SNN_Block(final_in_dim, 256),
                SNN_Block(256, 128))
        else:
            self.surv_net = nn.Sequential(
                nn.Linear(final_in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128))

        # 输出层
        self.quantile_reg = nn.Linear(128, 3)
        self.risk_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())

    def forward(self, mr_inputs, clinical_data):
        seq_features = []
        for seq_idx in range(mr_inputs.shape[1]):  # 处理每个序列
            slices = mr_inputs[:, seq_idx]  # [B, num_slices, 1, 224, 224]
            # 处理每张切⽚
            slice_feats = []
            for slice_idx in range(slices.shape[1]):  # 对每个序列中的每个切⽚进⾏特征提取
                feat = self.slice_extractor(slices[:,slice_idx]).flatten(1)  # [B, 512]
                slice_feats.append(feat)  # 将变量 feat 的值添加到列表slice_feats的末尾。
            # 切⽚级融合
            seq_feat = self.slice_fusion(torch.stack(slice_feats, dim=1))
            # [B, 512]
            seq_features.append(seq_feat)
        # 序列级融合
        #todo：
        #维度计算有问题
        if isinstance(self.seq_fusion.fusion, DynamicFusion):
            mr_feat = self.seq_fusion(seq_features, clinical_data)
        else:
            mr_feat = self.seq_fusion(seq_features)

        # 多模态融合
        if self.use_clinical and clinical_data is not None:
            clinical_feat = self.clinical_net(clinical_data)

            #简单拼接
            combined = torch.cat([mr_feat, clinical_feat], dim=1)

        else:
            combined = mr_feat

        # ⽣存预测
        features = self.surv_net(combined)  # [B,128]
        quantiles = self.quantile_reg(features)  # [B,3]
        risk = self.risk_head(features).squeeze(1)  # [B]
        return quantiles, risk


    # 以下代码定义了⼀个⽣存分析专⽤的损失函数，⽤于训练⽣存分析模型。它结合了分位数损失和负
    # 对数似然损失，以同时考虑预测分位数的准确性以及⽣存时间与⻛险分数的相关性。它通过以下⽅式
    # 提⾼模型的预测性能：
    # 使⽤分位数损失确保预测的分位数与实际⽣存时间的准确性。
    # 使⽤负对数似然损失确保⻛险分数与⽣存时间的负相关性。
    # 对事件发⽣的样本加强惩罚，提⾼模型对事件发⽣情况的预测能⼒。
    def compute_loss(self, batch, quantile_weights=[0.2,0.6,0.2]):
        """计算⽣存分析专⽤损失函数
        Args:
        batch: 包含'mr', 'clinical', 'time', 'event'的字典
        Returns:
        loss: 综合损失值
        """
        # 1. 模型预测
        quantiles, risk = self(batch['mr'], batch.get('clinical'))
        times = batch['time']
        events = batch['event']

        # 2. 分位数损失 (加强事件样本权重)，将预测的分位数分解为低、中、⾼三个部分。
        lower, median, upper = quantiles[:, 0], quantiles[:, 1],quantiles[:, 2]

        # 将事件标记转换为浮点数格式，对事件发⽣的样本加强惩罚
        event_mask = events.float()

        def quantile_loss(y_true, y_pred, tau):
            error = y_true - y_pred
            return torch.max((tau - 1) * error, tau * error)
        # 各分位数损失计算
        loss_lower = (quantile_loss(times, lower, self.quantiles[0]) * (1
                                                                        + event_mask * 0.5)).mean()
        loss_median = (quantile_loss(times, median, self.quantiles[1]) *
                       (1 + event_mask * 1)).mean()
        loss_upper = (quantile_loss(times, upper, self.quantiles[2]) * (1
                                                                        + event_mask * 0.5)).mean()

        # 顺序惩罚，通过惩罚违反顺序的情况，强制模型学习分位数的单调性
        order_penalty = torch.mean(torch.relu(lower - median) +
                                       torch.relu(median - upper))
        # 根据分位数权重计算加权平均的分位数损失。
        quantile_loss = (quantile_weights[0] * loss_lower +
                         quantile_weights[1] * loss_median +
                         quantile_weights[2] * loss_upper +
                         order_penalty)
        # 3. ⽣存⻛险损失（Cox⽐例⻛险）
        risk_loss = -torch.mean((times - median) * risk * event_mask)
        # 将分位数损失和负对数似然损失加权组合，形成最终的总损失。
        total_loss = quantile_loss.mean() + 0.1 * risk_loss
        return total_loss


