# !/usr/bin/env Python
# encoding=utf-8
'''
@Project ：6501_Capstone 
@File    ：try_InstructPix2Pix.py
@Author  ：Yixi Liang
@Date    ：2023/4/4 13:59
from https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/pix2pix#usage-example
'''
#%%
import PIL

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import matplotlib.pyplot as plt
import os

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
#
#
# def download_image(url):
#     image = PIL.Image.open(requests.get(url, stream=True).raw)
#     image = PIL.ImageOps.exif_transpose(image)
#     image = image.convert("RGB")
#     return image
#
# image = download_image(url)

def preprocess_image(path):
    image = PIL.Image.open(path)
    width, height = image.size
    ratio = width//400
    new_size = (400, height//ratio)
    image = image.resize(new_size)
    return image

cur_path = os.getcwd()
image_path = cur_path + '/real_photo/A/forbidden_city-Landscapes_HD_Wallpaper_1366x768.jpg'
image = preprocess_image(image_path)
plt.imshow(image)
plt.show()
#%%
prompt = "make the whole image change into dunhuang murals"
images = pipe(prompt, image=image, num_inference_steps=100, image_guidance_scale=1.5, guidance_scale=7.5).images
plt.imshow(images[0])
plt.show()

# images[0].save("snowy_mountains.png")
#%%
