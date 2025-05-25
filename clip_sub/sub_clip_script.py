# 用来解析.yaml文件
# 将参数传到sub_clip.py

import sys
sys.path.append('..')
import os
import yaml
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=str, help="config name")

args = parser.parse_args()

config_name = args.c

timestamp = time.time()
time_tuple = time.localtime(timestamp)
formatted_time = time.strftime("%Y%m%d%H%M%S", time_tuple)

config_name = f"config/{config_name}.yaml"

with open(config_name, 'r') as file:
    config = yaml.safe_load(file)

# 获取配置文件中的参数
device = config['device']
pretrained_model_name_or_path = config['pretrained_model_name_or_path']
output_dir = config['output_dir']
datasets_folder = config['datasets_folder']
dataset = config['dataset']
source_folder = config['source_folder']
target_folder = config['target_folder']
file_name = config['file_name']
prompts = config['prompts']

# 可选参数，如果配置文件中有就使用，没有就使用默认值
guidance_scale = config.get('guidance_scale', 3.5)
num_inference_steps = config.get('num_inference_steps', 28)
height = config.get('height', 512)
width = config.get('width', 512)

# 构建源文件夹和目标文件夹的完整路径
source = f"{datasets_folder}/{dataset}/{source_folder}"
target = f"{datasets_folder}/{dataset}/{target_folder}"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 构建命令
command = f'python sub_clip.py '
command += f'--device="{device}" '
command += f'--pretrained_model_name_or_path="{pretrained_model_name_or_path}" '
command += f'--output_dir="{output_dir}" '
command += f'--datasets_folder="{datasets_folder}" '
command += f'--dataset="{dataset}" '
command += f'--source_folder="{source}" '
command += f'--target_folder="{target}" '
command += f'--file_name="{file_name}" '
command += f'--prompts="{prompts}" '

# 添加可选参数
if 'guidance_scale' in config:
    command += f'--guidance_scale={guidance_scale} '
if 'num_inference_steps' in config:
    command += f'--num_inference_steps={num_inference_steps} '
if 'height' in config:
    command += f'--height={height} '
if 'width' in config:
    command += f'--width={width} '

print(command)

# 创建日志目录
os.makedirs("log", exist_ok=True)

# 记录日志
with open("log/sub_clip_log.log", "a") as file:
    file.write(f"执行时间: {formatted_time}\n")
    file.write(f"命令: {command}\n")

# 执行命令
rs = os.system(command)

# 记录执行结果
with open("log/sub_clip_log.log", "a") as file:
    if rs == 0:
        file.write(f"执行完成: {formatted_time}, 结果保存在 {output_dir}/{file_name}\n")
    else:
        file.write(f"执行失败: {formatted_time}, 错误代码 {rs}\n")
    file.write("\n")