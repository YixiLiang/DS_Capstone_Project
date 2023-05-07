# !/usr/bin/env Python
# encoding=utf-8
'''
@Project ：6501_Capstone 
@File    ：try_ stable-diffusion-v1-5.py
@Author  ：Yixi Liang
@Date    ：2023/2/27 17:51 
'''
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "National Mall in Washington D.C. in Monet style"
image = pipe(prompt).images[0]

plt.imshow(image)
plt.show()
image.save("./test_diffusion/monet_nation_mall.png")
print()