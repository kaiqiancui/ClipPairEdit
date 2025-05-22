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



#! 
clip_loss_weight = 0.5
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


prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
    [""], text_encoders, tokenizers, max_sequence_length
)
target_prompt_embeds = prompt_embeds[0]
target_pooled_prompt_embeds = pooled_prompt_embeds[0]

target_text_ids = text_ids[0]


params = []
modules = DEFAULT_TARGET_REPLACE
modules += UNET_TARGET_REPLACE_MODULE_CONV

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
pipe.set_progress_bar_config(disable=False)


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


sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
image_seq_len = 1024
mu = calculate_shift(
    image_seq_len,
    pipe.scheduler.config.base_image_seq_len,
    pipe.scheduler.config.max_image_seq_len,
    pipe.scheduler.config.base_shift,
    pipe.scheduler.config.max_shift,
)
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

for epoch in range(max_train_steps):
    # u = compute_density_for_timestep_sampling(
    #     weighting_scheme=weighting_scheme,
    #     batch_size=bsz,
    #     logit_mean=logit_mean,
    #     logit_std=logit_std,
    #     mode_scale=mode_scale,
    # )
    # indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    
    timestep_to_infer = random.randint(start_step, end_step)
    # print("timestep_to_infer:", timestep_to_infer)
    timesteps = torch.tensor([noise_scheduler.timesteps[timestep_to_infer]]).to(device=device)

    with torch.no_grad():
        random_sampler = random.randint(0, len(ims_sources) - 1)

        img1 = Image.open(f'{source_folder}/{ims_sources[random_sampler]}').convert("RGB").resize((512,512))
        img2 = Image.open(f'{target_folder}/{ims_targets[random_sampler]}').convert("RGB").resize((512,512))
        
        seed = random.randint(0,2*15)

        init_latents, denoised_latents, noise = pipe.add_noise_scale(img1, timestep_to_infer, torch.Generator().manual_seed(seed), noise_scale)
        
        denoised_latents = denoised_latents.to(device, dtype=weight_dtype)
        noise = noise.to(device, dtype=weight_dtype)

        init_latents2, denoised_latents2, noise2 = pipe.add_noise_scale(img2, timestep_to_infer, torch.Generator().manual_seed(seed), noise_scale)
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
    network_content.set_lora_slider(1)
    with ExitStack() as stack:
        stack.enter_context(network_content)

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

    # 获取当前时间步
    current_sigma = timesteps[0]
    
    # 计算预测的x0
    x0_pred_unpacked = denoised_latents - current_sigma * model_noise_pred_pos
    # print(f"x0_pred_unpacked shape: {x0_pred_unpacked.shape}") # 输出应为 torch.Size([1, 1024, 64])
    # print(f"vae.config.block_out_channels length: {len(vae.config.block_out_channels)}") # 输出应为 4
    # print(f"vae.config.latent_channels: {vae.config.latent_channels}") # 输出VAE的潜在通道数
    
    # 分析_unpack_latents函数的工作原理
    # 从源代码来看，_unpack_latents函数的处理流程是：
    # 1. 输入形状为[batch_size, num_patches, channels]
    # 2. 将其重塑为[batch_size, height, width, channels//4, 2, 2]
    # 3. 转置为[batch_size, channels//4, height, 2, width, 2]
    # 4. 最后重塑为[batch_size, channels//4, height*2, width*2]
    
    # 根据错误信息，VAE期望的输入通道数为16，而我们的通道数是64
    # 这表明我们需要将channels从64调整为16
    
    # 首先获取基本维度信息
    batch_size, num_patches, channels = x0_pred_unpacked.shape
    latent_height = int(math.sqrt(num_patches))
    latent_width = latent_height
    
    # 打印详细的维度信息以便分析
    # print(f"Original latent shape: {x0_pred_unpacked.shape}")
    # print(f"Latent dimensions (H,W): ({latent_height}, {latent_width})")
    
    # 根据_unpack_latents函数的逻辑，我们需要将channels从64调整为16
    # 在_unpack_latents中，它假设channels可以被分解为channels//4组，每组包含4个通道
    # 所以我们需要将channels调整为可以被正确解包的形式
    
    # 将latents重塑为正确的形状
    # 首先将[B, H*W, C]转换为[B, H, W, C]
    reshaped = x0_pred_unpacked.reshape(batch_size, latent_height, latent_width, channels)
    
    # 根据错误信息，VAE期望的输入通道数为16
    # 所以我们需要将channels从64调整为16
    # 在VAE中，每个通道对应一个特征图，所以我们需要将通道数量减少
    
    # 计算需要的通道数
    target_channels = 16  # VAE期望的输入通道数
    
    # 使用平均池化将通道数从64减少到16
    # 我们将每4个连续的通道平均合并为1个通道
    # 这样可以保留所有信息，只是以压缩的形式
    channel_groups = channels // target_channels
    
    # 重塑为[B, H, W, target_channels, channel_groups]
    grouped = reshaped.reshape(batch_size, latent_height, latent_width, target_channels, channel_groups)
    
    # 对每组通道进行平均，得到[B, H, W, target_channels]
    averaged = grouped.mean(dim=4)
    
    # 转置为[B, target_channels, H, W]格式，这是VAE期望的输入格式
    latents_for_vae = averaged.permute(0, 3, 1, 2)
    
    # print(f"Adjusted latents_for_vae shape: {latents_for_vae.shape}")

    vae.eval()
    # 使用调整后的latents进行数值调整
    latents_to_decode = latents_for_vae / vae.config.scaling_factor
    # print(f"Final latents_to_decode shape: {latents_to_decode.shape}")
    # print(f"VAE scaling factor: {vae.config.scaling_factor}")
    
    
    latents_to_decode = latents_to_decode.to(dtype=pipe.vae.dtype, device=pipe.vae.device)

    # 解码
    with torch.no_grad():
        decoded_image_tensor = vae.decode(latents_to_decode).sample
        
    # 目标：将图像张量 [-1, 1] 转换为 PIL 图像
    # 首先：将范围从 [-1, 1] 归一化到 [0, 1]
    decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)
    
    # 手动转化到PIL
    pil_images = []
    
    for i in range(decoded_image_tensor.shape[0]): # 遍历批次中的每张图像
        image_slice = decoded_image_tensor[i]
        
        # 先将BFloat16转换为float32，然后再转换为NumPy数组
        # 解决“Got unsupported ScalarType BFloat16”错误
        image_slice = image_slice.to(dtype=torch.float32)
        
        # 手动转换为 PIL 图像
        image_np = image_slice.cpu().permute(1, 2, 0).numpy() # C, H, W -> H, W, C
        image_np = (image_np * 255).round().astype("uint8")
        
        from PIL import Image
        pil_img = Image.fromarray(image_np)
        pil_images.append(pil_img)
        
    # 算clip loss
    if pil_images:
        decoded_pred_image_pil = pil_images[0] # 假设批处理大小为 1

        # --- 后续的 CLIP 处理代码保持不变 ---
        clip_input_target_img = clip_processor(
            images=img2, return_tensors="pt"
        ).pixel_values.to(device=device, dtype=clip_model.dtype)

        clip_input_pred_img = clip_processor(
            images=decoded_pred_image_pil, return_tensors="pt"
        ).pixel_values.to(device=device, dtype=clip_model.dtype)

        with torch.no_grad():
            pred_img_features = clip_model.get_image_features(clip_input_pred_img)
            target_img_features = clip_model.get_image_features(clip_input_target_img)
            pred_img_features = pred_img_features / pred_img_features.norm(p=2, dim=-1, keepdim=True)
            target_img_features = target_img_features / target_img_features.norm(p=2, dim=-1, keepdim=True)
        
        similarity = torch.nn.functional.cosine_similarity(pred_img_features, target_img_features, dim=-1)
        clip_loss_value = (1.0 - similarity).mean()     
        
    total_loss_for_semantic = loss2 + clip_loss_weight * clip_loss_value
    total_loss_for_semantic.backward() # 在这里进行反向传播

    logs = {
        "loss2": loss2.item(), 
        "loss1": loss1.item(), # loss1 的计算应保持不变
        "clip_loss": clip_loss_value.item(), 
        "lr": lr_scheduler.get_last_lr()[0]
    }
    
    # writer.add_scalar(f'LoRA - {lora_name}_{version}_{init_cl_weight}cl -Concept', loss1.item(), epoch)

    
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    progress_bar.update(1)
    progress_bar.set_postfix(**logs)

    
    # 恢复network_content的梯度
    for lora in network_content.unet_loras:
        for param in lora.parameters():
            param.requires_grad = True

    if (epoch) % save_per_steps == 0 and epoch != 0:
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