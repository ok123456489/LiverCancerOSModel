import os
import numpy as np
import nibabel as nib
import imageio
from pathlib import Path

# 定义序列类型映射（根据您的文档）
SEQUENCE_MAP = {
    'T2': 'T2',
    'DWI': 'DWI',
    'DP': 'DP',  # 延迟期
    'AP': 'AP',  # 动脉期
    'VP': 'VP'  # 静脉期
}


def remove_leading_zeros(s):
    """去除字符串中的前导零"""
    return s.lstrip('0') or '0'  # 如果全部是零，保留一个零


def process_mr_case(patient_root, mask_root, output_root, case_id):
    """
    处理单个病例的MRI数据

    参数:
        patient_root: 患者数据根目录 (xxx/)
        mask_root: mask数据根目录 (xxx/)
        output_root: 输出目录
        case_id: 病例ID (xxx部分)
    """
    # 去除case_id的前导零
    clean_case_id = remove_leading_zeros(case_id)

    # 构建路径
    mri_dir = os.path.join(patient_root, case_id, "baseline", "imaging", "MRI")
    mask_dir = os.path.join(mask_root, case_id, "baseline")
    output_dir = os.path.join(output_root, clean_case_id)  # 使用去除前导零的ID

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 检查目录是否存在
    if not os.path.exists(mri_dir):
        print(f"警告: MRI目录不存在: {mri_dir}")
        return
    if not os.path.exists(mask_dir):
        print(f"警告: mask目录不存在: {mask_dir}")
        return

    # 获取所有NIfTI文件
    mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.nii.gz')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')]

    # 处理每个MRI文件
    for mri_file in mri_files:
        # 确定序列类型
        seq_type = None
        for key in SEQUENCE_MAP:
            if key.lower() in mri_file.lower():
                seq_type = SEQUENCE_MAP[key]
                break

        if not seq_type:
            print(f"警告: 无法确定{mri_file}的序列类型，跳过")
            continue

        # 查找对应的mask文件（假设文件名相同）
        mask_file = mri_file
        if mask_file not in mask_files:
            print(f"警告: 找不到{mri_file}对应的mask文件，跳过")
            continue

        try:
            # 加载数据
            mri_path = os.path.join(mri_dir, mri_file)
            mask_path = os.path.join(mask_dir, mask_file)

            mri_img = nib.load(mri_path).get_fdata()
            mask_img = nib.load(mask_path).get_fdata()

            # 验证尺寸
            if mri_img.shape != mask_img.shape:
                print(f"警告: {mri_path}和{mask_path}尺寸不匹配")
                continue

            # 创建只包含标签5的mask
            mask_label5 = (mask_img == 5).astype(np.uint8)

            # 应用mask
            roi = mri_img * mask_label5

            # 计算每个切片的mask区域（只计算标签5的区域）
            slice_areas = [np.sum(mask_label5[:, :, i] > 0) for i in range(mask_label5.shape[2])]

            # 筛选出面积大于0的切片
            candidate_slices = [i for i, area in enumerate(slice_areas) if area > 0]

            if len(candidate_slices) == 0:
                print(f"警告: {mri_file}没有包含标签5的有效切片")
                continue

            # 按面积排序并选择面积最大的切片
            candidate_slices_sorted = sorted(candidate_slices, key=lambda x: slice_areas[x], reverse=True)
            selected_slice = candidate_slices_sorted[0]  # 选择面积最大的切片

            # 创建序列输出目录
            seq_output_dir = os.path.join(output_dir, seq_type)
            Path(seq_output_dir).mkdir(exist_ok=True)

            # 保存切片
            if slice_areas[selected_slice] > 0:
                # 归一化并保存为PNG
                slice_data = roi[:, :, selected_slice]
                if np.ptp(slice_data) > 0:  # ptp = max - min
                    slice_norm = ((slice_data - slice_data.min()) /
                                  np.ptp(slice_data) * 255).astype(np.uint8)
                else:
                    slice_norm = slice_data.astype(np.uint8)

                output_path = os.path.join(
                    seq_output_dir,
                    f"{clean_case_id}_{seq_type}_slice{selected_slice:03d}.png"  # 使用去除前导零的ID
                )
                imageio.imwrite(output_path, slice_norm)
                print(f"已保存: {output_path}")

        except Exception as e:
            print(f"处理{mri_file}时出错: {str(e)}")

def process_all_cases(patient_root, mask_root, output_root):
    """
    处理所有病例

    参数:
        patient_root: 患者数据根目录 (包含多个xxx文件夹)
        mask_root: mask数据根目录 (包含多个xxx文件夹)
        output_root: 输出根目录
    """
    # 获取所有病例ID (xxx部分)
    case_ids = set()

    # 从patient目录获取
    for entry in os.listdir(patient_root):
        entry_path = os.path.join(patient_root, entry)
        if os.path.isdir(entry_path):
            mri_path = os.path.join(entry_path, "baseline", "imaging", "MRI")
            if os.path.exists(mri_path):
                case_ids.add(entry)

    # 从mask目录获取
    for entry in os.listdir(mask_root):
        entry_path = os.path.join(mask_root, entry)
        if os.path.isdir(entry_path):
            mask_path = os.path.join(entry_path, "baseline")
            if os.path.exists(mask_path):
                case_ids.add(entry)

    print(f"找到{len(case_ids)}个病例需要处理...")

    for case_id in sorted(case_ids):
        print(f"\n处理病例: {case_id}")
        process_mr_case(patient_root, mask_root, output_root, case_id)


if __name__ == "__main__":
    # 配置路径
    PATIENT_ROOT = "/home/jiang_stu/dataset/HCC_Patient_Dataset/patients"  # 包含多个xxx文件夹的根目录
    MASK_ROOT = "/home/jiang_stu/dataset/HCC_Patient_Dataset/masks/liver_organ/TotalSegmentator"  # 包含多个xxx文件夹的根目录
    OUTPUT_ROOT = "../data/mri_images"  # 输出目录

    # 确保输出目录存在
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # 处理所有病例
    process_all_cases(PATIENT_ROOT, MASK_ROOT, OUTPUT_ROOT)

    print("\n处理完成!")