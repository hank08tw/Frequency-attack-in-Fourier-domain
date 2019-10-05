import torch.utils.model_zoo as model_zoo
from data.dataset import mydata
from config import opt
import os
import torch as t
import models
from torchvision import models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer as vis
import numpy
from tqdm import tqdm
from torch import nn
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt 


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
all_max_num=[]
all_max_val=[]
all_target_max_val=[]
all_flip=[]
target_max_num=9
target_max_val=18.462112426757812
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        
        for i in range(x.shape[0]):
            max_num=-1
            max_val=-2147483647
            for j in range(x.shape[1]):
                if x[i][j].item()>max_val:
                    max_val=x[i][j].item()
                    max_num=j
            all_max_num.append(max_num)
            all_max_val.append(max_val)
            all_target_max_val.append(x[i][target_max_num].item())
            if max_num==target_max_num:
                all_flip.append(0)#0代表沒有flip
            else:
                all_flip.append(1)#1代表flip
        return x
count=0
flip_num=0
same_num=0
flip_list=[]
max_diff=[]
class AlexNet2(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet2, self).__init__()
        #count=0
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = x.view(-1,1)
        print('********************************')
        global count
        global flip_num
        global same_num
        global flip_list
        global max_diff
        count=count+1
        #for i in range(len(x)):
        #    origin[i].append(x[i][0])
        #print(len(x))
        #print(origin)
        
        max_num=-1
        max_val=-2147483647
        for i in range(len(x)):
            #origin[i].append(x[i][0])
            #print('x[i].data[0].item(): '+str(x[i].data[0].item()))
            if x[i].data[0].item()>max_val:
                max_val=x[i].data[0].item()
                max_num=i
        print(count)
        print('max_num: '+str(max_num))
        print('max_val: '+str(max_val))
        print('origin_num: '+str(origin[-2]))
        print('origin_val: '+str(origin[-1]))
        if max_num==origin[-2]:
            print('same')
            same_num=same_num+1
        else:
            print('flip')
            flip_num=flip_num+1
            flip_list.append(count)
        print('*********************************')
        max_diff.append(max_val-origin[-1])
        
        return x

def test(**kwargs):
    model= AlexNet(1000)
    model.eval()
    #model.eval()
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
    
    print(model)
    #if opt.use_gpu:model.cuda()
    # data
    train_data_origin = mydata(opt.test_root,test=True)
    test_dataloader = DataLoader(train_data_origin,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data,volatile = True)
        #if opt.use_gpu: input = input.cuda()
        score = model(input)
    print('all_max_num: ')
    print(all_max_num)
    print('all_max_val: ')
    print(all_max_val)
    x1=range(0,100) 
    plt.plot(x1,all_target_max_val,label='all_target_max_val',linewidth=3,color='r',marker='o', 
    markerfacecolor='blue',markersize=12) 
    plt.plot(x1,all_flip,label='flip') 
    plt.xlabel('sample number') 
    plt.ylabel('val') 
    plt.title('graph') 
    plt.legend() 
    plt.show() 

    return results
test()
