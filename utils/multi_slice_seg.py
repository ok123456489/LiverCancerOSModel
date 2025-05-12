import os
import numpy as np
import nibabel as nib
import imageio
import pandas as pd
from pathlib import Path

# 定义序列类型映射
SEQUENCE_MAP = {
    'T2': 'T2',
    'DWI': 'DWI',
    'DP': 'DP',  # 延迟期
    'AP': 'AP',  # 动脉期
    'VP': 'VP'  # 静脉期
}


def load_slice_info_from_excel(excel_path):
    """
    从Excel文件加载切片信息

    返回:
        dict: {病人编号: {序列: [切片列表]}}
    """
    df = pd.read_excel(excel_path)
    slice_info = {}

    for _, row in df.iterrows():
        patient_id = str(row['病人编号'])
        seq_type = row['序列名称']
        slices_str = row['切片号（癌栓、面积最大、多发或者差异大）']

        # 处理切片字符串 (如 "32,39,16")
        if isinstance(slices_str, str):
            slices = [int(s.strip()) for s in slices_str.split(',')]
        else:
            slices = []

        # 添加到字典
        if patient_id not in slice_info:
            slice_info[patient_id] = {}

        slice_info[patient_id][seq_type] = slices

    return slice_info


def remove_leading_zeros(s):
    """去除字符串中的前导零"""
    return s.lstrip('0') or '0'


def load_nifti_with_orientation(path, orientation='RAS'):
    """
    加载NIfTI文件并调整到指定方向
    """
    img = nib.load(path)
    data = img.get_fdata()
    orig_orientation = nib.aff2axcodes(img.affine)

    if orig_orientation != tuple(orientation):
        ornt_transform = nib.orientations.ornt_transform(
            nib.orientations.axcodes2ornt(orig_orientation),
            nib.orientations.axcodes2ornt(orientation)
        )
        data = nib.orientations.apply_orientation(data, ornt_transform)

    return data


def process_mr_case(patient_root, mask_root, output_root, case_id, slice_info):
    """
    处理单个病例的MRI数据

    参数:
        slice_info: 从Excel加载的切片信息字典
    """
    clean_case_id = remove_leading_zeros(case_id)

    # 检查是否有该病例的切片信息
    if clean_case_id not in slice_info:
        print(f"警告: 病例 {clean_case_id} 没有切片信息，跳过")
        return

    mri_dir = os.path.join(patient_root, case_id, "baseline", "imaging", "MRI")
    mask_dir = os.path.join(mask_root, case_id, "baseline")
    output_dir = os.path.join(output_root, clean_case_id)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(mri_dir):
        print(f"警告: MRI目录不存在: {mri_dir}")
        return
    if not os.path.exists(mask_dir):
        print(f"警告: mask目录不存在: {mask_dir}")
        return

    mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.nii.gz')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')]

    for mri_file in mri_files:
        seq_type = None
        for key in SEQUENCE_MAP:
            if key.lower() in mri_file.lower():
                seq_type = SEQUENCE_MAP[key]
                break

        if not seq_type:
            print(f"警告: 无法确定{mri_file}的序列类型，跳过")
            continue

        # 检查该序列是否有指定的切片
        if seq_type not in slice_info[clean_case_id]:
            print(f"警告: 病例 {clean_case_id} 序列 {seq_type} 没有切片信息，跳过")
            continue

        mask_file = mri_file
        if mask_file not in mask_files:
            print(f"警告: 找不到{mri_file}对应的mask文件，跳过")
            continue

        try:
            mri_path = os.path.join(mri_dir, mri_file)
            mask_path = os.path.join(mask_dir, mask_file)

            mri_img = load_nifti_with_orientation(mri_path)
            mask_img = load_nifti_with_orientation(mask_path)

            if mri_img.shape != mask_img.shape:
                print(f"警告: {mri_path}和{mask_path}尺寸不匹配")
                continue

            mask_label5 = (mask_img == 5).astype(np.uint8)
            roi = mri_img * mask_label5

            # 获取该序列的指定切片
            slice_indices = slice_info[clean_case_id][seq_type]

            # 创建序列输出目录
            seq_output_dir = os.path.join(output_dir, seq_type)
            Path(seq_output_dir).mkdir(exist_ok=True)

            # 保存指定的切片
            for i, idx in enumerate(slice_indices, 1):
                if idx < mri_img.shape[2]:  # 确保切片索引有效
                    slice_data = roi[:, :, idx]
                    if np.ptp(slice_data) > 0:
                        slice_norm = ((slice_data - slice_data.min()) /
                                      np.ptp(slice_data) * 255).astype(np.uint8)
                    else:
                        slice_norm = slice_data.astype(np.uint8)

                    # 逆时针旋转90度再水平翻转
                    rotated = np.rot90(slice_norm, k=1)  # k=1表示逆时针旋转90度
                    flipped = np.fliplr(rotated)  # 水平翻转

                    output_path = os.path.join(
                        seq_output_dir,
                        f"{clean_case_id}_{seq_type}_slice_{i}.png"
                    )
                    imageio.imwrite(output_path, flipped)
                    print(f"已保存: {output_path}")
                else:
                    print(f"警告: 切片索引 {idx} 超出范围 (最大 {mri_img.shape[2] - 1})")

        except Exception as e:
            print(f"处理{mri_file}时出错: {str(e)}")


def process_all_cases(patient_root, mask_root, output_root, excel_path):
    """
    处理所有病例

    参数:
        excel_path: Excel文件路径
    """
    # 从Excel加载切片信息
    slice_info = load_slice_info_from_excel(excel_path)

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
        process_mr_case(patient_root, mask_root, output_root, case_id, slice_info)


if __name__ == "__main__":
    # 配置路径
    PATIENT_ROOT = "/home/jiang_stu/dataset/HCC_Patient_Dataset/patients"  # 包含多个xxx文件夹的根目录
    MASK_ROOT = "/home/jiang_stu/dataset/HCC_Patient_Dataset/masks/liver_organ/TotalSegmentator"  # 包含多个xxx文件夹的根目录
    OUTPUT_ROOT = "../data/mri_images/train"  # 输出目录
    EXCEL_PATH = "slice_info.xlsx"  # Excel文件路径

    # 确保输出目录存在
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # 处理所有病例
    process_all_cases(PATIENT_ROOT, MASK_ROOT, OUTPUT_ROOT, EXCEL_PATH)

    print("\n处理完成!")