import argparse
import os

from ablation import run_ablation
from visualization import plot_ablation_results


def main():
    # 主程序入口：协调数据加载、模型训练和评估流程
    # 1. 参数解析
    parser = argparse.ArgumentParser(description='肝癌生存预测消融实验')
    parser.add_argument('--train_data_root', default='./data/train', help='训练集数据根目录')
    # parser.add_argument('--test_data_root', default='./data/test', help='测试集数据根目录')
    parser.add_argument('--clinical_file', default='clinical_info.xlsx', help='临床数据文件名')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    # 添加MRI序列相关参数
    parser.add_argument('--mri_sequences', nargs='+', default=['AP', 'VP', 'DP', 'T2'], help='使用的MRI序列列表')
    parser.add_argument('--num_slices', type=int, default=3, help='每个序列使用的切片数量')
    parser.add_argument('--k_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', default='./output', help='输出目录')  # 添加输出目录参数
    args = parser.parse_args()

    # 2. 运行消融实验
    final_results,_ = run_ablation(
        data_root=args.train_data_root,
        clinical_file=args.clinical_file,
        mri_sequences=args.mri_sequences,
        num_slices=args.num_slices,
        batch_size=args.batch_size,
        k_folds=args.k_folds,
        seed=args.seed
    )
    # 3. 可视化结果
    os.makedirs(args.output_dir, exist_ok=True)
    # 绘制消融实验结果
    plot_ablation_results(final_results, save_path=os.path.join(args.output_dir, 'ablation_results.png'))
if __name__ == "__main__":
    main()