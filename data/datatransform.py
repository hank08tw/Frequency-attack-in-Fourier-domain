import numpy as np
from PIL import Image
import PIL
import os
import cv2
from PIL import Image
#import math
import random
Image.LOAD_TRUNCATED_IMAGES = True

"""
for i in range(50):
    r1=random.randint(0,255)
    g1=random.randint(0,255)
    b1=random.randint(0,255)

    r2=random.randint(0,255)
    g2=random.randint(0,255)
    b2=random.randint(0,255)
    generate_square_wave(178,[r1,g1,b1],[r2,g2,b2],218,178)
"""


def compute(thepath):
    img = cv2.imread(thepath)
    return np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
#計算資料夾裡圖片rgb的平均值
origin_names = os.listdir('./check_alexnet_root_resize')
filter_names = os.listdir('./maxsinwave')
for origin_name in origin_names:
    if origin_name== '.DS_Store':continue
    path= './check_alexnet_root_resize/'+origin_name
    for filter_name in filter_names:
        if filter_name == '.DS_Store':continue
        path2 = './maxsinwave/'+filter_name
        im=Image.open(path)
        #print(im)
        R, G, B= compute(path)
        #print(R,G,B)
        im2=Image.open(path2)
        #print(im2)
        im3=Image.blend(im,im2,0.3)
        #im3.save(path2,format="jpeg")
        #print(im3)
        R2, G2, B2= compute(path2)
        #print(R2,G2,B2)
        #print(im3)
        I=np.array(im3)#PIL.Image.open(path2))
        for i in range(I.shape[0]): # for every pixel:
            for j in range(I.shape[1]):
                I[i,j,0] = max(min(I[i,j,0] - (R2-R),255),0)
                I[i,j,1] = max(min(I[i,j,1] - (G2-G),255),0)
                I[i,j,2] = max(min(I[i,j,2] - (B2-B),255),0)
        im4 = PIL.Image.fromarray(np.uint8(I))
        filter_name=filter_name[:-4]
        final_name=filter_name+'+'+origin_name
        im4.save('./maxsinwave_test/'+final_name,format="jpeg")
