'''
Created on Jul 24, 2019

@author: monky
'''
from utils import convert2cpu

import torch

def load_conv(buf, start, conv_model):

    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    #print("start: {}, num_w: {}, num_b: {}".format(start, num_w, num_b))
    # by ysyun, use .view_as()
    conv_model.bias.data.copy_(torch.Tensor(buf[start:start+num_b]).view_as(conv_model.bias.data));   start = start + num_b
    conv_model.weight.data.copy_(torch.Tensor(buf[start:start+num_w]).view_as(conv_model.weight.data)); start = start + num_w
    return start

def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.Tensor(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.Tensor(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.Tensor(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.Tensor(buf[start:start+num_b]));   start = start + num_b
    #conv_model.weight.data.copy_(torch.Tensor(buf[start:start+num_w])); start = start + num_w
    conv_model.weight.data.copy_(torch.Tensor(buf[start:start + num_w]).view_as(conv_model.weight.data)); start = start + num_w
    return start

def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.Tensor(buf[start:start+num_b]));     start = start + num_b
    fc_model.weight.data.copy_(torch.Tensor(buf[start:start+num_w]));   start = start + num_w 
    return start

def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)