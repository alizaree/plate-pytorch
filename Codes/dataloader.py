import sys
 
# setting path
sys.path.append('../')
import argparse
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
#from callbacks import AverageMeter, Logger
from data_utils.data_loader_frames_new import VideoFolder
from utils import save_results
import torchvision
import ffmpeg
from model.model_lib_new import BC_MODEL as VideoModel


