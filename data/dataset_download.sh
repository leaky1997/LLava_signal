export HF_ENDPOINT="https://hf-mirror.com"

git lfs install
# git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain --depth=1
git clone https://www.modelscope.cn/datasets/thomas/LLaVA-Pretrain.git

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
# git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K --depth=1
git clone https://www.modelscope.cn/datasets/thomas/LLaVA-Instruct-150K.git

# 定义数据集的URLs
urls=(
"http://images.cocodataset.org/zips/train2017.zip"
"https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
"https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing"
"https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
"https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
"https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
)

# # 下载和解压每个数据集
# for url in "${urls[@]}"; do
#     if [[ $url == *"drive.google.com"* ]]; then
#         # 使用gdown工具下载Google Drive链接的文件
#         gdown "$url"
#     else
#         # 使用wget命令下载
#         wget "$url"
#     fi

#     # 获取.zip文件的名称
#     file=$(basename "$url")

#     # 如果文件是.zip文件，使用unzip命令解压
#     if [[ $file == *.zip ]]; then
#         unzip "$file"
#     fi
# done

# 下载和解压每个数据集
for url in "${urls[@]}"; do
    # 创建新的文件夹
    folder=$(basename "$url" .zip)
    mkdir "$folder"
    cd "$folder"

    if [[ $url == *"drive.google.com"* ]]; then
        # 使用gdown工具下载Google Drive链接的文件
        gdown "$url"
    else
        # 使用wget命令下载
        wget "$url"
    fi

    # 获取.zip文件的名称
    file=$(basename "$url")

    # 如果文件是.zip文件，使用unzip命令解压
    if [[ $file == *.zip ]]; then
        unzip "$file"
    fi

    # 返回上一级目录
    cd ..
done

# http://images.cocodataset.org/zips/train2017.zip
# https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
# https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing

# #!/bin/bash
# ocr_vqa_path="<your-directory-path>"

# find "$target_dir" -type f | while read file; do
#     extension="${file##*.}"
#     if [ "$extension" != "jpg" ]
#     then
#         cp -- "$file" "${file%.*}.jpg"
#     fi
# done

# https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

# https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip

# https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip