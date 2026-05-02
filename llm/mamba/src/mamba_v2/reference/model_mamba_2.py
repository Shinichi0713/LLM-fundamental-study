import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import AutoTokenizer

from causal_conv1d import CausalConv1d  # 因果性を保証するconv1d
from einops import rearrange  # 形状操作を簡潔にする


