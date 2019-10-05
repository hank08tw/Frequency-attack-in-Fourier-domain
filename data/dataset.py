# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
#from torchvision import transforms as T
from . import transforms as T
#import random
from random import randint
import math
count=0
class mydata(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs = sorted(imgs)
        imgs_num = len(imgs)
        self.imgs = imgs

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        
        img_path = self.imgs[index]
        #if ".DS_Store" in img_path:return #????怎麼處理這個 只能手動刪？？？
        if self.test:
            label = 1#int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            # print("Now is train")
            print("img path is {}".format(img_path))
            label = 1#int(img_path.split('/')[-1].split('.')[0])
            #label = 1 if 'female' in img_path.split('/')[-1] else 0
            print("Train label is:{}".format(label))
        data = Image.open(img_path)
        global count
        count=count+4
        print(count)
        #print(data)
        
        crop=T.CenterCrop(110)
        resize=T.Resize(224)
        
        max_color=math.floor(count)#randint(1,count)
        m_choice=[1,2,3,4,5,6,7,8,15,23,38,55,80]
        m_pick=randint(1,10)
        m=m_choice[m_pick]#[m_pick]
        my_filter=T.Filter()
        
        #randomhorizontalflip=T.RandomHorizontalFlip()
        totensor=T.ToTensor()
        normalize=T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        data=crop(data)
        data=resize(data)
        data=my_filter(data,max_color,m)
        #data=randomhorizontalflip(data)
        data=totensor(data)
        data=normalize(data)
        #data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)