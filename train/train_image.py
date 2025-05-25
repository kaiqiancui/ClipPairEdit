import sys
sys.path.append('..')
from torch.utils.tensorboard import SummaryWriter
import argparse

import argparse
import gc
import logging
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast
from utils.compare_utils import *
from utils.flux_utils import *
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from utils.flux_pipeline import FluxPipeline, calculate_shift, retrieve_timesteps
import argparse
import gc
import torch
from tqdm.auto import tqdm
import numpy as np
from torch.optim import AdamW
from contextlib import ExitStack
import random
from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from transformers import logging
logging.set_verbosity_warning()
from diffusers import logging
logging.set_verbosity_error()
import math
#! clip添加的部分
from transformers import CLIPModel, CLIPProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--lora_name", type=str)
parser.add_argument("--version", type=str)
parser.add_argument("--num_inference_steps", type=int)
parser.add_argument("--train_iteration", type=int)
parser.add_argument("--save_per_steps", type=int)
parser.add_argument("--device", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--datasets_folder", type=str)
parser.add_argument("--dataset",type=str)
parser.add_argument("--source_folder",type=str)
parser.add_argument("--target_folder",type=str)
parser.add_argument("--cfg_eta",type=float)
parser.add_argument("--start_step", type=int)
parser.add_argument("--end_step", type=int)
parser.add_argument("--guidance_scale",type=float)
parser.add_argument("--noise_scale",type=float)
parser.add_argument("--pretrained_model_name_or_path",type=str)

args = parser.parse_args()
num_inference_steps = args.num_inference_steps
lora_name = args.lora_name

device = args.device

max_train_steps = args.train_iteration
save_per_steps = args.save_per_steps
output_dir = args.output_dir

dataset = args.dataset

source_folder = args.source_folder
target_folder = args.target_folder

cfg_eta = args.cfg_eta
noise_scale = args.noise_scale

start_step = args.start_step
end_step = args.end_step
guidance_scale = args.guidance_scale

import os
os.environ["CUDA_VISIBLE_DEVICES"]=device
device = f"cuda:{args.device}"

import torch

pretrained_model_name_or_path = args.pretrained_model_name_or_path
weight_dtype = torch.bfloat16



max_sequence_length = 512
height = width = 512 

# timestep weighting
weighting_scheme = 'none' #["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"]
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29
bsz = 1
training_eta = 1
# optimizer params
lr = 0.002

# os.makedirs(output_dir, exist_ok=True)


# lora params
alpha = 1
rank = 16
# train_method = 'xattn'
train_method = 'full'

# training params
batchsize = 1



#! 这里的clip_loss_weight是CLIP loss的权重 
clip_loss_weight_init = 10
clip_loss_final = 0.2 * clip_loss_weight_init


loss2_weight_final = 0.06
loss2_weight_init = 10 * loss2_weight_final


def schedule_weight(step, total_steps, initial_val, final_val):
    """Linearly interpolates a weight from an initial to a final value."""
    if total_steps <= 1:
        return initial_val
    # Calculate the interpolation factor (from 0.0 to 1.0)
    alpha = min(float(step) / float(total_steps - 1), 1.0)
    # Apply the linear interpolation
    return initial_val + (final_val - initial_val) * alpha



def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()


# Load the tokenizers
tokenizer_one = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype, device_map=device)
tokenizer_two = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", torch_dtype=weight_dtype, device_map=device)

# Load scheduler and models
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",torch_dtype=weight_dtype, device_map=device)
# noise_scheduler_copy = copy.deepcopy(noise_scheduler)


# import correct text encoder classes
text_encoder_cls_one = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, device=device)
text_encoder_cls_two = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, subfolder="text_encoder_2", device=device)
# Load the text encoders
text_encoder_one, text_encoder_two = load_text_encoders(pretrained_model_name_or_path, text_encoder_cls_one, text_encoder_cls_two, weight_dtype, device)

# Load VAE
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)

#! 预训练的clip模型
clip_model_name = "openai/clip-vit-large-patch14" # 或者 "openai/clip-vit-base-patch32" 或其他变体
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=weight_dtype).to(device)
clip_model.requires_grad_(False) # 冻结CLIP模型参数
clip_model.eval() # 设置CLIP模型为评估模式


# We only train the additional adapter LoRA layers
transformer.requires_grad_(False)
vae.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

vae.to(device)
transformer.to(device)
text_encoder_one.to(device)
text_encoder_two.to(device)

tokenizers = [tokenizer_one, tokenizer_two]
text_encoders = [text_encoder_one, text_encoder_two]


prompt_embeds_arr = []
pooled_prompt_embeds_arr = []
text_ids_arr = []

# 计算空文本的嵌入作为目标嵌入
prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
    [""], text_encoders, tokenizers, max_sequence_length
)
target_prompt_embeds = prompt_embeds[0]
target_pooled_prompt_embeds = pooled_prompt_embeds[0]

target_text_ids = text_ids[0]


params = []
modules = DEFAULT_TARGET_REPLACE
modules += UNET_TARGET_REPLACE_MODULE_CONV

# 两个lora
network_content = LoRANetwork(
    transformer,
    rank=rank,
    multiplier=0.0,
    alpha=alpha,
    train_method=train_method,
).to(device, dtype=weight_dtype)

network_semantic = LoRANetwork(
    transformer,
    rank=rank,
    multiplier=0.0,
    alpha=alpha,
    train_method=train_method,
).to(device, dtype=weight_dtype)

# 将两个lora添加到参数列表
params.extend(network_content.prepare_optimizer_params() + network_semantic.prepare_optimizer_params())
    
optimizer = AdamW(params, lr=lr)
optimizer.zero_grad()

criteria = torch.nn.MSELoss()

pipe = FluxPipeline(noise_scheduler,
    vae,
    text_encoder_one,
    tokenizer_one,
    text_encoder_two,
    tokenizer_two,
    transformer,
)

# 进度条
pipe.set_progress_bar_config(disable=False)

# 学习率调度
lr_warmup_steps = 200
lr_num_cycles = 1
lr_power = 1.0
lr_scheduler = 'constant' 
#Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
lr_scheduler = get_scheduler(
    lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps,
    num_training_steps=max_train_steps,
    num_cycles=lr_num_cycles,
    power=lr_power,
)
    
progress_bar = tqdm(
    range(0, max_train_steps),
    desc="Steps",
)

losses = {}

# log_dir = "./logs_test"  # 日志保存的目录
# writer = SummaryWriter(log_dir=log_dir)

l1_loss = torch.nn.L1Loss()
criteria = torch.nn.MSELoss()

generator = None

vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
height_model = 2 * (int(height) // pipe.vae_scale_factor)
width_model = 2 * (int(width) // pipe.vae_scale_factor)

# 均匀减小的噪声调度
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

# 图像展平之后的大小
image_seq_len = 1024

# 线性插值计算mu方便后续的timesteps计算（对于不同的seq_len适用）
mu = calculate_shift(
    image_seq_len,
    pipe.scheduler.config.base_image_seq_len,
    pipe.scheduler.config.max_image_seq_len,
    pipe.scheduler.config.base_shift,
    pipe.scheduler.config.max_shift,
)

# 计算时间步
retrieve_timesteps(
    pipe.scheduler,
    num_inference_steps,
    device,
    None,
    sigmas,
    mu=mu,
)


ims_sources = os.listdir(f'{source_folder}/')
ims_sources = [im_ for im_ in ims_sources if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]
ims_sources = sorted(ims_sources)

ims_targets = os.listdir(f'{target_folder}/')
ims_targets = [im_ for im_ in ims_targets if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]
ims_targets = sorted(ims_targets)    

#! clip的tensor操作

clip_image_size = clip_processor.image_processor.size["shortest_edge"] # 通常是 224
clip_mean = torch.tensor(clip_processor.image_processor.image_mean, device=device, dtype=weight_dtype).view(1, 3, 1, 1)
clip_std = torch.tensor(clip_processor.image_processor.image_std, device=device, dtype=weight_dtype).view(1, 3, 1, 1)
# 预处理方法
from torchvision.transforms import Resize, Normalize, Compose, InterpolationMode, ToTensor
transform_for_clip = Compose([
    Resize((clip_image_size, clip_image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
    Normalize(mean=clip_processor.image_processor.image_mean, std=clip_processor.image_processor.image_std)
])


for epoch in range(max_train_steps):
    # u = compute_density_for_timestep_sampling(
    #     weighting_scheme=weighting_scheme,
    #     batch_size=bsz,
    #     logit_mean=logit_mean,
    #     logit_std=logit_std,
    #     mode_scale=mode_scale,
    # )
    # indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    
    # 随机选择一个时间步
    timestep_to_infer = random.randint(start_step, end_step)
    
    # 将时间步转换为张量
    timesteps = torch.tensor([noise_scheduler.timesteps[timestep_to_infer]]).to(device=device)

    with torch.no_grad():
        random_sampler = random.randint(0, len(ims_sources) - 1)

        img1 = Image.open(f'{source_folder}/{ims_sources[random_sampler]}').convert("RGB").resize((512,512))
        img2 = Image.open(f'{target_folder}/{ims_targets[random_sampler]}').convert("RGB").resize((512,512))
        
        seed = random.randint(0,2*15)

        # 对img1加噪声， 获取初始潜在表示、噪声潜在表示和噪声
        # add_nosie scale方法：

        # def add_noise(self, image, step, generator):
        #     vae = self.vae

        #     height = 512
        #     width = 512

        #     height = 2 * (int(height) // self.vae_scale_factor)
        #     width = 2 * (int(width) // self.vae_scale_factor)

        #     batch_size = 1
        #     num_channels_latents = self.transformer.config.in_channels // 4
        #     shape = (batch_size, num_channels_latents, height, width)

        #     device = vae.device
        #     image = self.image_processor.preprocess(image)
        #     image = image.to(device=device, dtype=vae.dtype)

        #     init_latents = vae.encode(image).latent_dist.sample(None)
        #     init_latents = vae.config.scaling_factor * (init_latents - self.vae.config.shift_factor)

        #!    init_latents = self._pack_latents(init_latents, batch_size, num_channels_latents, height, width)

        #     shape = init_latents.shape

        #     noise = torch.randn(shape, generator=generator).to(device)

        #     sigmas = self.scheduler.sigmas.to(device=device)

        #     # add noise
        #     init_latents_add_noise = init_latents + sigmas[step] * noise
        #     # init_latents_add_noise = self.scheduler.scale_noise(init_latents, torch.tensor([self.scheduler.timesteps[step]], device=device), noise)
            
        #     return init_latents, init_latents_add_noise, noise
        # 生成的这些变量，都是打包过的
        init_latents, denoised_latents, noise = pipe.add_noise_rect(img1, timestep_to_infer, torch.Generator().manual_seed(seed))
        
        denoised_latents = denoised_latents.to(device, dtype=weight_dtype)
        noise = noise.to(device, dtype=weight_dtype)

        init_latents2, denoised_latents2, noise2 = pipe.add_noise_rect(img2, timestep_to_infer, torch.Generator().manual_seed(seed))
        denoised_latents2 = denoised_latents2.to(device, dtype=weight_dtype)
        noise2 = noise2.to(device, dtype=weight_dtype)

        
        denoised_latents_unpack = FluxPipeline._unpack_latents(
            denoised_latents,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )

        init_latents = FluxPipeline._unpack_latents(
            init_latents,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )

        denoised_latents_unpack = FluxPipeline._unpack_latents(
            denoised_latents,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )

        init_latents2 = FluxPipeline._unpack_latents(
            init_latents2,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )

    if epoch == 0:
        # 在第一个epoch时，初始化模型的输入，输入的是加噪的img1的潜变量
        model_input = FluxPipeline._unpack_latents(
            denoised_latents,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )

    noise = FluxPipeline._unpack_latents(
        noise,
        height=int(model_input.shape[2] * vae_scale_factor / 2),
        width=int(model_input.shape[3] * vae_scale_factor / 2),
        vae_scale_factor=vae_scale_factor,
    )

    noise2 = FluxPipeline._unpack_latents(
        noise2,
        height=int(model_input.shape[2] * vae_scale_factor / 2),
        width=int(model_input.shape[3] * vae_scale_factor / 2),
        vae_scale_factor=vae_scale_factor,
    )
    
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0], model_input.shape[2], 
        model_input.shape[3], 
        device, weight_dtype,
    )

    # handle guidance
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=device)
        guidance = guidance.expand(denoised_latents.shape[0])
    else:
        guidance = None

    # network_content 用于学习原来的语义，比如说image of a person
    network_content.set_lora_slider(1) # 权重滑块，这里的意思是完全应用lora权重
    with ExitStack() as stack:
        stack.enter_context(network_content)
        #? 在这里临时应用lora权重训练，退出这个部分之后会恢复原始状态?
        model_noise_pred_neg = transformer(
            hidden_states=denoised_latents,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=target_pooled_prompt_embeds,
            encoder_hidden_states=target_prompt_embeds,
            txt_ids=target_text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

    # unpack
    model_pred_neg = FluxPipeline._unpack_latents(
        model_noise_pred_neg,
        height=int(model_input.shape[2] * vae_scale_factor / 2),
        width=int(model_input.shape[3] * vae_scale_factor / 2),
        vae_scale_factor=vae_scale_factor,
    )

    loss1 = criteria(model_pred_neg, noise)
    loss1 = loss1.mean()
    loss1.backward()

    # network_semantic 用于学习增加的语义，比如说old = "image of a person, old" - "image of a person"
    network_content.set_lora_slider(1)
    network_semantic.set_lora_slider(1)
    # 设置network_content不需要梯度，只需要更新network_semantic的梯度
    for lora in network_content.unet_loras:
        for param in lora.parameters():
            param.requires_grad = False

    # 同样预测一个噪声
    with network_semantic:
        with network_content:
            model_noise_pred_pos = transformer(
                hidden_states=denoised_latents,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=target_pooled_prompt_embeds,
                encoder_hidden_states=target_prompt_embeds,
                txt_ids=target_text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]


    model_pred_pos = FluxPipeline._unpack_latents(
        model_noise_pred_pos,
        height=int(model_input.shape[2] * vae_scale_factor / 2),
        width=int(model_input.shape[3] * vae_scale_factor / 2),
        vae_scale_factor=vae_scale_factor,
    )

    loss2 = criteria(model_pred_pos, cfg_eta * (init_latents - init_latents2) + noise)
    loss2 = loss2.mean()
    
    clip_loss_value = torch.tensor(0.0).to(device, dtype = weight_dtype)

    # 确保 current_sigma 是 PyTorch 张量，且 dtype 和 device 正确
    current_sigma = timesteps[0].to(device=device, dtype=weight_dtype)

    # 计算预测的x0 (依然是 Transformer 的原始输出形式，例如 [1, 1024, 64])
    # denoised_latents 在此处已经是 [1, 1024, 64] 形式，与 model_noise_pred_pos 兼容。
    x0_pred_unpacked = denoised_latents - current_sigma * model_noise_pred_pos
    
    #! 重新打包
    latents_for_vae = FluxPipeline._unpack_latents(
        latents=x0_pred_unpacked,
        height=height, # Your target image height, e.g., 512
        width=width,   # Your target image width, e.g., 512
        vae_scale_factor=vae_scale_factor # The VAE scale factor, e.g., 16
    )

    
    vae.eval() # 保持 VAE 在评估模式
    
    # 使用调整后的latents进行数值调整，并确保数据类型和设备正确
    latents_to_decode = latents_for_vae / vae.config.scaling_factor
    latents_to_decode = latents_to_decode.to(dtype=pipe.vae.dtype, device=pipe.vae.device)

    # 解码（梯度流经 VAE 解码器）
    decoded_image_tensor = vae.decode(latents_to_decode).sample
    
    # 目标：将图像张量 [-1, 1] 归一化到 [0, 1]
    decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)
    
    # #! CLIP 的张量操作
    # 预测图像的 CLIP 预处理（使用预定义的 transform_for_clip）
    # decoded_image_tensor 是 [B, 3, H, W]，值在 [0, 1] 范围内
    clip_input_pred_img_tensor = transform_for_clip(decoded_image_tensor)

    # 目标图像 img2 的 CLIP 预处理
    # img2 是 PIL 图像，先转换为张量，再进行与预测图像相同的预处理
    target_image_tensor_raw = ToTensor()(img2).unsqueeze(0).to(device, dtype=decoded_image_tensor.dtype)
    clip_input_target_img_tensor = transform_for_clip(target_image_tensor_raw)

    # 确保 clip_model 在 eval 模式且已冻结
    clip_model.eval()
    clip_model.requires_grad_(False) 

    pred_img_features = clip_model.get_image_features(clip_input_pred_img_tensor)
    
    with torch.no_grad(): 
        target_img_features = clip_model.get_image_features(clip_input_target_img_tensor)
    
    # 归一化特征
    pred_img_features = pred_img_features / pred_img_features.norm(p=2, dim=-1, keepdim=True)
    target_img_features = target_img_features / target_img_features.norm(p=2, dim=-1, keepdim=True)
    

    # 计算余弦相似度损失
    similarity = torch.nn.functional.cosine_similarity(pred_img_features, target_img_features, dim=-1)
    clip_loss_value = (1.0 - similarity).mean()
    
    clip_loss_weight = schedule_weight(epoch, max_train_steps, clip_loss_weight_init, clip_loss_final)
    loss2_weight = schedule_weight(epoch, max_train_steps, loss2_weight_init, loss2_weight_final)
    
    total_loss_for_semantic = clip_loss_weight * clip_loss_value + loss2_weight * loss2
    
    total_loss_for_semantic.backward() 

    logs = {
        "loss2": loss2.item(), 
        "loss1": loss1.item(), 
        "clip_loss": clip_loss_value.item(),
        "lr": lr_scheduler.get_last_lr()[0]
    }
    
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    progress_bar.update(1)
    progress_bar.set_postfix(**logs)

    
    # 恢复network_content的梯度
    for lora in network_content.unet_loras:
        for param in lora.parameters():
            param.requires_grad = True

    # if (epoch) % save_per_steps == 0 and epoch != 0:
    #     # Save the trained LoRA model
    #     save_path = f'{output_dir}'
    #     os.makedirs(save_path, exist_ok=True)

    #     print("Saving...")
    #     output_weight = f"{save_path}/{dataset}_{cfg_eta:.1f}eta_{noise_scale:.1f}ns_{num_inference_steps}inf_{epoch}epoch_0.pt"
    #     network_content.save_weights(
    #         output_weight,
    #         dtype=weight_dtype,
    #     )
    #     network_semantic.save_weights(
    #         f"{save_path}/{dataset}_{cfg_eta:.1f}eta_{noise_scale:.1f}ns_{num_inference_steps}inf_{epoch}epoch.pt",
    #         dtype=weight_dtype,
    #     )


# Save the trained LoRA model
save_path = f'{output_dir}'
os.makedirs(save_path, exist_ok=True)

print("Saving...")
output_weight = f"{save_path}/{dataset}_{cfg_eta:.1f}eta_{noise_scale:.1f}ns_{num_inference_steps}inf_{epoch}epoch_0.pt"
network_content.save_weights(
    output_weight,
    dtype=weight_dtype,
)
network_semantic.save_weights(
    f"{save_path}/{dataset}_{cfg_eta:.1f}eta_{noise_scale:.1f}ns_{num_inference_steps}inf_{epoch}epoch.pt",
    dtype=weight_dtype,
)

  
print('Training Done')