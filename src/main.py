import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
#from mnas_seesaw_w import MnasNetw
#from mnas_seesaw_3version import MnasNet
#from cv3_half import seesawnet_half
#from seesaw_sfv2_img import ShuffleNetV2
#from mnas_seesaw_kernel import seesawnet
from model import MobileFaceNet, Backbone
#from seesaw_model import seesawFaceNet
from model_seesaw import seesawFaceNet
from model_seesaw_alls import seesawFaceNet_alls
from model_seesaw_alls_d import seesawFaceNet_alls_d
from model_seesaw_large import seesawFaceNet_alls_large
from model_seesaw_alls_sepro import seesawFaceNet_alls_d_sepro
from model_seesaw_alls_w import seesawFaceNet_alls_w

#from hyperface import hyperFaceNet
#from hyperattentionface import hyperattentionface
#from model_preluhyper import hyperFaceNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#model = MobileFaceNet().to(device)
'''
# Params:    1,200.51K
# Mult-Adds: 221.18M
'''

#model = seesawFaceNet_alls().to(device)
'''
# Params:    1,370.18K
# Mult-Adds: 145.77M
sigmoid not included
'''

#model = seesawFaceNet_alls_d().to(device)
'''
# Params:    2,073.41K
# Mult-Adds: 236.92M
'''

#model = seesawFaceNet_alls_d_sepro().to(device)
'''
# Params:    2,080.93K
# Mult-Adds: 390.31M
'''

#model = Backbone(100, 0.2).to(device)
'''
# Params:    65,129.79K
# Mult-Adds: 12,076.80M
'''
model = seesawFaceNet_alls_w().to(device)
'''
# Params:    2,073.41K
# Mult-Adds: 236.92M
'''

model.eval()
summary(model, torch.zeros((1, 3, 112, 112)))