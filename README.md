# 文档

## 代码结构

- train: 训练相关代码
  - config: 配置文件
  - datasets: 各种属性的image pair，如"age"属性
    - age
      - source: 对应source图片
      - target: 对应target图片，文件名与source一一对应
  - eval_scripts: 量化脚本
  - prompts: 各种prompt文件，做生成的时候可以使用
  - 训练：
    - train_image.py: Pairedit完整的训练代码
    - train_image_script.py: 训练脚本
    - train_image_wocontent.py: 无content的ablation训练代码
    - train_image_wocfg.py: 无cfg的ablation训练代码
    - train_image_fm.py: 无noise schedule，使用flow matching加噪的ablation训练代码
    - train_image_sliders.py: sliders image版本
    - train_image_real.py: 训练real image的content LoRA，用于做real image editing
  - 生成：
    - generate_image.py: 推理生成图片代码
    - gen_image_script.py: 生成用的脚本
    - generate_real_image.py: 用于real image editing
    - gen_real_image_script.py: real image editing脚本
  - utils: 工具类，flux及lora相关工具
- slider_text: slider text的训练代码

## 配置文件

```yaml
device: 1
pretrained_model_name_or_path: "/home/haoguang/model/black-forest-labs/FLUX.1-dev" # 模型地址

python_name: train_image_fm # 用于训练的python文件，如用train_image.py训练则是train_image

datasets_folder: datasets # 数据集文件夹地址
datasets: # 需要训练哪个数据集，可训练多个
  - age
  - smile

source_folder: source # 训练集source
target_folder: target # 训练集target

config_name: base # 配置文件名字，用于标记当前的训练目的，可随意指定

start_step: 0 # 随机选择t进行训练时的起始t
end_step: 27 # 随机选择t进行训练时的终止t
num_inference_steps: 28
guidance_scale: 3.5

cfg_eta: 4 # 对应semantic loss的eta
noise_scales: # 对应加噪时的noise的倍数，可测试不同的noise scale
  - 1
  # - 2
  # - 3

train_iteration: 500
save_per_steps: 200 # 每多少步保存一次
train_start: 1 # 训练的当前版本号，如1则对应lora的"v1"版本
train_count: 1 # 需要训练多少个版本，版本号会累加
output_folder: "output/models/image/"

train_method: full # lora设置

# 生成部分
use_trained_lora_path: 0 # 是否使用已经训练好的lora，0或1
trained_lora_path: "output/models/pretrained/elf.pt" # 如果使用已训练好的lora，则直接加载trained_lora_path

use_prompts_file: 0 # 是否使用prompts_file_gen来生成图片，0或1
prompts_file: 
  - prompts-person.yaml

use_random_seed: 0 # 是否使用随机seed生成图片，如果不使用随机seed,则使用start_seed_gen顺序生成

output_folder_gen: "output/eval/image/"

prompt: # 如果不使用prompts_file_gen来生成图片，则使用该prompt
  - image of a person

skips: # 从第几个t开始使LoRA生效，可测试多个skip
  - 10
start_seed_gen: 6000

scales: 0,0.8,1.0,1.2 # 对应LoRA的scale
count_gen: 20 # 生成的数量
```

## Training

进入train文件夹，使用命令如：

`python train_image_script.py --c=train_image`

则使用config文件夹下`train_image.yaml`配置文件进行训练

会生成模型到output_folder，如：

`output/models/image/pixel/pixel_vtrain_image_4.0eta_3.0ns_full_base_v1/pixel_4.0eta_3.0ns_28inf_500epoch.pt`

## Inference

使用命令如：

`python gen_image_script.py --c=train_image`

则使用config文件夹下`train_image.yaml`配置文件进行图片生成

## Real Image Editing

real image editing的配置文件如`train_image_real.yaml`与正常训练的相比多了以下部分：

```yaml
image_ids: # 用于指定使用哪张图片(使用id对应)来训练出一个用于inversion的LoRA，可训练多个id
  - 610
  - 1743

semantic_lora_path: "output/models/pretrained/elf.pt" # 生成的时候使用哪个semantic LoRA进行编辑
semantic_cfg_scale: 0.75 # 生成的时候编辑用的semantic LoRA的cfg强度
```

先训练出rea image图片对应的inversion的LoRA:

`python train_image_real_script.py --c=train_image_real`

然后编辑real image：

`python gen_image_real_script.py --c=train_image_real`
## 环境
python==3.9.19
cuda: 12.1