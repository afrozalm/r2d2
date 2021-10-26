import torch
import torch.nn as nn
import cv2
import torchvision
import os, pdb
from PIL import Image
import numpy as np
import torch
from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *

img = cv2.imread('./imgs/boat.png', 0)

def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()

model = load_network("./models/r2d2_WASF_N16.pt")
model.eval()
print(model)
example1 = torch.rand(1, 1, 3, 256, 256)
traced_script_module = torch.jit.trace(model, example1, strict = False)
traced_script_module.save("traced_r2d2_WASF_N16.pt")