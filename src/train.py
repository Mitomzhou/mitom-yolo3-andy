from __future__ import print_function
import sys

import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import gc

from utils import *
from cfg import parse_cfg
from darknet import DarkNet

datacfg    = '../cfg/voc.data'
cfgfile    = '../cfg/yolo3_voc_train.cfg'
weightfile = '../weight/darknet53.conv.74'

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]
print(data_options)
print(net_options)

use_cuda = torch.cuda.is_available()

#voc.data
trainlist = data_options['train']
testlist = data_options['valid']
backupdir = data_options['backup']
gpus = data_options['gpus']
ngpus = len(gpus.split(','))
num_workers = int(data_options['num_workers'])

# cfg
batch_size = int(net_options['batch'])
max_batches = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum = float(net_options['momentum'])
decay = float(net_options['decay'])
steps = [float(step) for step in net_options['steps'].split(',')]
scales = [float(scale) for scale in net_options['scales'].split(',')]


print(batch_size)
print(max_batches)
print(learning_rate)
print(momentum)
print(decay)
print(steps)
print(scales)

nsamples = file_lines(trainlist)
max_epochs = (max_batches*batch_size)//nsamples+1  #why?
print(nsamples)
print(max_epochs)

seed = int(time.time())
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")
print(device)

model = DarkNet(cfgfile, use_cuda=use_cuda)

if weightfile is not None:
    model.load_weights(weightfile)

#model.print_net()
init_epoch = model.seen//nsamples
print(init_epoch)

loss_layers = model.loss_layers
for l in loss_layers:
    l.seen = model.seen

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
        
params_dict = dict(model.named_parameters())
print(params_dict)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



if __name__ == '__main__':
    print('----------')
