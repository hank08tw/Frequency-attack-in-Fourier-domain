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
        x = x.view(-1,1)
        print('********************************')
        global flip_num
        global same_num
        global flip_list
        global max_diff
        
        max_num=-1
        max_val=-2147483647

        second_num=-1
        second_val=-2147483647
        for i in range(len(x)):
            if x[i].data[0].item()>max_val:
                max_val=x[i].data[0].item()
                max_num=i
            elif x[i].data[0].item()>second_val:
                second_val=x[i].data[0].item()
                second_num=i
        print(max_num)
        print(max_val)
        print(second_num)
        print(second_val)
        print(max_val-second_val)
        return x
def test(**kwargs):
    model= AlexNet(1000)
    model.eval()
    #model.eval()
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
    
    print(model)
    #if opt.use_gpu:model.cuda()
    # data
    train_data_origin = mydata(opt.check_root,test=True)
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


    return results
test()
