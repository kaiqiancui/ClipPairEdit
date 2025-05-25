import sys
sys.path.append('..')
from torch.utils.tensorboard import SummaryWriter
import argparse
import gc
import logging
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import torch
import numpy as np
from torch.optim import AdamW
from contextlib import ExitStack
import random
from transformers import logging
logging.set_verbosity_warning()
from diffusers import logging
logging.set_verbosity_error()
from src.flux_pipeline import FluxPipeline

from PIL import Image
from torchvision.transforms import ToTensor
from src.model import load_clip_model
import open_clip
from transformers import CLIPModel, CLIPProcessor





parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str)
parser.add_argument("--pretrained_model_name_or_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--datasets_folder", type=str)
parser.add_argument("--dataset",type=str)
parser.add_argument("--source_folder",type=str)
parser.add_argument("--target_folder",type=str)
parser.add_argument("--file_name", type=str)
parser.add_argument("--prompts", type=str)
args = parser.parse_args()


device = args.device
pretrained_model_name_or_path = args.pretrained_model_name_or_path
output_dir = args.output_dir
data_set_folder = args.datasets_folder
dataset = args.dataset
source_folder = args.source_folder
target_folder = args.target_folder
file_name = args.file_name
prompts = args.prompts

import os
os.environ["CUDA_VISIBLE_DEVICES"]=device
device = f"cuda:{args.device}"

weight_dtype = torch.bfloat16


# 加载 CLIP 模型
clip_model, _, transform_for_clip = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
clip_model.load_state_dict(torch.load('/disks/sata2/kaiqian/ClipPairEdit/clip_sub/sub_clip_model.pt', weights_only=True))
clip_model.to(device, dtype=weight_dtype).eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')

# 加载 FluxPipeline
pipe = FluxPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=weight_dtype)
pipe.to(device)

# 图像路径
ims_sources = sorted([f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))])
ims_targets = sorted([f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))])

# 读取图像
img1 = Image.open(f'{source_folder}/{ims_sources[0]}').convert("RGB").resize((512, 512))
img2 = Image.open(f'{target_folder}/{ims_targets[0]}').convert("RGB").resize((512, 512))

# 预处理图像并提取特征
img1_tensor = transform_for_clip(img1).unsqueeze(0).to(device, dtype=weight_dtype)
img2_tensor = transform_for_clip(img2).unsqueeze(0).to(device, dtype=weight_dtype)

with torch.no_grad():
    img1_features = clip_model.encode_image(img1_tensor)  # [1, 768]
    img2_features = clip_model.encode_image(img2_tensor)  # [1, 768]

# 计算文本嵌入
text_token = clip_tokenizer([prompts]).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_token)  # [1, 768]


sub_features = img2_features - img1_features + text_features # [1, 768]
# sub_features = text_features  # [1, 768]
sub_features = sub_features/(sub_features.norm(dim=-1, keepdim=True)+1e-6)
sub_features = sub_features.to(device, dtype=weight_dtype)


prompt = "a person" 
prompt2 = ""
# 获取 T5
prompt_tokens = pipe.tokenizer_2(
    prompt2, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
).input_ids.to(device)

# clip_tokens = pipe.tokenizer(
#     prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
# ).input_ids.to(device)

with torch.no_grad():
    prompt_embeds = pipe.text_encoder_2(prompt_tokens).last_hidden_state  # [1, 512, 4096]


# 生成图像
image = pipe(
    prompt=None,
    prompt_embeds = prompt_embeds,
    pooled_prompt_embeds = sub_features,
    height=512,
    width=512,
    num_inference_steps=28,
    guidance_scale=7.0,
    generator=torch.manual_seed(0),
    output_type="pil",
).images[0]

image.save(f'{output_dir}/{file_name}')