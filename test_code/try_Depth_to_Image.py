# !/usr/bin/env Python
# encoding=utf-8
'''
@Project ：6501_Capstone 
@File    ：try_Depth_to_Image.py
@Author  ：Yixi Liang
@Date    ：2023/4/4 16:11
from https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion_2
'''
#%%
import torch
import requests
import PIL
import matplotlib.pyplot as plt
import os

from diffusers import StableDiffusionDepth2ImgPipeline, EulerDiscreteScheduler

euler_scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler")
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
    scheduler=euler_scheduler
).to("cuda")


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# init_image = Image.open(requests.get(url, stream=True).raw)
def preprocess_image(path):
    image = PIL.Image.open(path)
    width, height = image.size
    ratio = width//400
    new_size = (400, height//ratio)
    image = image.resize(new_size)
    return image

cur_path = os.getcwd()
image_path = cur_path + '/real_photo/A/forbidden_city-Landscapes_HD_Wallpaper_1366x768.jpg'
init_image = preprocess_image(image_path)
plt.imshow(init_image)
plt.show()
#%%
prompt = "change image into dunhuang murals"
n_propmt = ""
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.5,
             ).images[0]
plt.imshow(image)
plt.show()