# 将train2014文件夹里的前8000张照片提取到train2014_min文件夹，加快训练速度

import os
import shutil

def copy_files(src_dir, dst_dir, num_files):
    # 获取源目录中的所有文件
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # 确保文件数量不超过目录中的文件数量
    files_to_copy = files[:num_files]
    
    # 复制文件到目标目录
    for file in files_to_copy:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        shutil.copy(src_file, dst_file)
        print(f"Copied {file} to {dst_dir}")

# 源目录和目标目录
src_directory = '/home/ustcgmh/picture_style_transform/train2014'
dst_directory = '/home/ustcgmh/picture_style_transform/train2014_min'

# 复制前 8000 个文件
copy_files(src_directory, dst_directory, 8000)