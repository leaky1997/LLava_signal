# XTuner 微调 Llama3 图片理解多模态

随着 XTuner 团队放出了基于 Llama3-8B 的 LLaVA 模型，我们也是第一时间与 XTuner 团队取得了联系，并获得了他们已经预训练好的 Image Projector。接下来，我们将带大家基于 Llama3-8B-Instruct 和 XTuner 团队预训练好的 Image Projector 微调自己的多模态图文理解模型 LLaVA。

- 基于Llama3-8B的模型和XTuner团队的Image Projector，我们希望构建一个针对一维信号的模式识别模型。

## 环境、模型、数据准备

### 配置环境

我们先来配置相关环境。使用如下指令便可以安装好一个 python=3.10 pytorch=2.1.2+cu121 的基础环境了。

```bash
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
接下来我们安装 XTuner。

```bash
cd ~
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install -e .
```

### 模型准备

#### 准备 Llama3 权重

在微调开始前，我们首先来准备 Llama3-8B-Instruct 模型权重。

```bash
/home/user/model_checkpoint/modelscope_cache/LLM-Research/Meta-Llama-3-8B-Instruct/
```

<!-- - InternStudio

```bash
cd ~
ln -s /root/new_models/meta-llama/Meta-Llama-3-8B-Instruct .
```
- 非 InternStudio

我们选择从 OpenXLab 上下载 Meta-Llama-3-8B-Instruct 的权重。

```bash
cd ~
git lfs install
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
``` -->

#### 准备 Visual Encoder 权重

我们接下来准备 Llava 所需要的 openai/clip-vit-large-patch14-336，权重，即 Visual Encoder 权重。

```bash
./clip-vit-large-patch14-336
```

<!-- - InternStudio
  
```bash
cd ~
ln -s /root/new_models/openai/clip-vit-large-patch14-336 .
```

- 非 InternStudio

可以访问 https://huggingface.co/openai/clip-vit-large-patch14-336 以进行下载。 -->


#### 准备 Image Projector 权重

然后我们准备 Llava 将要用到的 Image Projector 部分权重。

- xtuner_llave 训练toturial的权重

```
/home/user/model_checkpoint/z_other/iter_2181.pth or
./Image_Projector/iter_2181.pth 
```

- llava-llama-3-8b 的预训练权重

<!-- - InternStudio

```bash
cd ~
ln -s /root/new_models/xtuner/llama3-llava-iter_2181.pth .
```

- 非 InternStudio

相关权重可以访问：https://huggingface.co/xtuner/llava-llama-3-8b 以及 https://huggingface.co/xtuner/llava-llama-3-8b-v1_1 。（已经过微调，并非 Pretrain 阶段的 Image Projector） -->

### 数据准备

#### 预训练数据


> 参考 https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_prepare.md#llava-dataset
- 构建数据下载脚本，并采用国内model_scope
```bash
./data/dataset_download.sh
```


- 准备好数据 就可以执行
  
```bash
./pretrain.sh
```
- 参考的数据结构为

```
./data/llava_data
├── LLaVA-Pretrain
│   ├── blip_laion_cc_sbu_558k.json
│   ├── blip_laion_cc_sbu_558k_meta.json
│   └── images
├── LLaVA-Instruct-150K
│   └── llava_v1_5_mix665k.json
└── llava_images
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
```


#### 微调数据构建



image_generator.ipynb 步骤生成
<!-- 我们按照 https://github.com/InternLM/Tutorial/blob/camp2/xtuner/llava/xtuner_llava.md 中的教程来准备微调数据。为了让大家可以快速上手，我们选择了使用过拟合的方式快速实现。

可以执行以下代码：



```bash
cd ~
git clone https://github.com/InternLM/tutorial -b camp2
python ~/tutorial/xtuner/llava/llava_data/repeat.py \
  -i ~/tutorial/xtuner/llava/llava_data/unique_data.json \
  -o ~/tutorial/xtuner/llava/llava_data/repeated_data.json \
  -n 200
``` -->

## 微调过程

### 训练启动

我们已经为大家准备好了可以一键启动的配置文件，主要是修改好了模型路径、对话模板以及数据路径。

我们使用如下指令以启动训练：

```bash
cd ~
git clone https://github.com/SmartFlowAI/Llama3-XTuner-CN
mkdir -p ./project/llama3-ft
cd ./project/llama3-ft



export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
xtuner train /home/user/LQ/B_Signal/LLava_signal/Llama3-XTuner-CN/configs/llama3-llava/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
--work-dir ./llava \
--deepspeed deepspeed_zero2

```

> 这段代码是一个深度学习模型训练的配置脚本，主要用于配置模型、数据集、优化器、学习率调度器等训练相关的参数。它主要分为五个部分：

> > 设置（Settings）：这部分定义了模型、数据、优化器和学习率调度器的基本参数，如模型和数据的路径、批次大小、学习率、权重衰减等。

>> 模型 & 分词器 & 图像处理器（Model & Tokenizer & Image Processor）：这部分定义了模型、分词器和图像处理器的具体参数。模型部分包括了预训练模型的路径、是否冻结模型的参数、模型的量化配置等。分词器和图像处理器部分主要是定义了预训练模型的路径。

>> 数据集 & 数据加载器（Dataset & Dataloader）：这部分定义了数据集和数据加载器的参数，如数据路径、图像文件夹、分词器、图像处理器、数据映射函数、模板映射函数、最大长度等。

>> 调度器 & 优化器（Scheduler & Optimizer）：这部分定义了优化器和学习率调度器的参数，如优化器类型、学习率、beta值、权重衰减、梯度裁剪、累积计数、损失缩放、数据类型等。学习率调度器部分定义了学习率的调整策略，如线性学习率和余弦退火学习率。

>> 运行时（Runtime）：这部分定义了训练过程中的一些参数，如自定义钩子、默认钩子、环境配置、可视化器、日志级别、加载的检查点、是否从加载的检查点恢复训练、随机种子、是否禁用确定性、日志处理器等。


训练过程所需显存约为44447 MiB，在单卡A100上训练所需时间为30分钟。

在训练好之后，我们将原始 image projector 和 我们微调得到的 image projector 都转换为 HuggingFace 格式，为了下面的效果体验做准备。

```bash
xtuner convert pth_to_hf ~/Llama3-XTuner-CN/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
  ~/llama3-llava-iter_2181.pth \
  ~/project/llama3-ft/llava/pretrain_iter_2181_hf

xtuner convert pth_to_hf ~/Llama3-XTuner-CN/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
  ~/project/llama3-ft/llava/iter_1200.pth \
  ~/project/llama3-ft/llava/finetune_iter_1200_hf
```

### 效果体验

![image](https://github.com/SmartFlowAI/Llama3-XTuner-CN/assets/75657629/551bfebf-399c-4aec-985b-affa94a5963b)

在转换完成后，我们就可以在命令行简单体验一下微调后模型的效果了。

> 问题1：Describe this image.
> 问题2：What is the equipment in the image?

#### Pretrain 模型

```bash
xtuner chat ~/Meta-Llama-3-8B-Instruct \
  --visual-encoder ~/clip-vit-large-patch14-336 \
  --llava ~/project/llama3-ft/llava/pretrain_iter_2181_hf \
  --prompt-template llama3_chat \
  --image ~/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

![image](https://github.com/SmartFlowAI/Llama3-XTuner-CN/assets/75657629/0ddd6ed1-97d2-46e6-b580-5d6425a15604)

此时可以看到，Pretrain 模型只会为图片打标签，并不能回答问题。

#### Finetune 后 模型

```bash
xtuner chat ~/Meta-Llama-3-8B-Instruct \
  --visual-encoder ~/clip-vit-large-patch14-336 \
  --llava ~/project/llama3-ft/llava/finetune_iter_1200_hf \
  --prompt-template llama3_chat \
  --image ~/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```

![image](https://github.com/SmartFlowAI/Llama3-XTuner-CN/assets/75657629/a8f0f0be-7210-4ecb-9584-0f02c2335246)

经过 Finetune 后，我们可以发现，模型已经可以根据图片回答我们的问题了。
