import os 
import torch
import torch.nn as nn
import numpy as np
#from IPython import embed
import random
from e2evc.Utils import utils_nn  # use integer-convolutions like in TemporalResampling

# disable all non-deterministic behaviour
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

def setup_seed(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False    
        torch.backends.cudnn.deterministic = True
setup_seed()

class ECCVGeneratorYCbCr(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGeneratorYCbCr, self).__init__()


        self.model0 = nn.Sequential(
            utils_nn.IntConv2d(1, 128, kernel_size=1), nn.ReLU(),  
            utils_nn.IntConv2d(128, 64, kernel_size=1), nn.ReLU(),
            utils_nn.IntConv2d(64, 32, kernel_size=1), nn.ReLU(),
            utils_nn.IntConv2d(32, 1, kernel_size=1) )
              

        model1=[utils_nn.IntConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[utils_nn.IntConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[utils_nn.IntConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[utils_nn.IntConv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[utils_nn.IntConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[utils_nn.IntConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[utils_nn.IntConv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[utils_nn.IntConv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[utils_nn.IntConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[utils_nn.IntConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[utils_nn.IntConv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[utils_nn.IntConv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[utils_nn.IntConv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[utils_nn.IntConv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[utils_nn.IntConv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[utils_nn.IntConv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[utils_nn.IntConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[utils_nn.IntConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[utils_nn.IntConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[utils_nn.IntTransposedConv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[utils_nn.IntConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[utils_nn.IntConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[utils_nn.IntConv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model11 = nn.Sequential(
            utils_nn.IntConv2d(3, 128, kernel_size=1), nn.ReLU(),  
            utils_nn.IntConv2d(128, 64, kernel_size=1), nn.ReLU(),
            utils_nn.IntConv2d(64, 32, kernel_size=1), nn.ReLU(),
            utils_nn.IntConv2d(32, 3, kernel_size=1) )

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = utils_nn.IntConv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):

        conv0 = self.model0( input_l )
        conv1_2 = self.model1(conv0)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))
        ups = self.upsample4(out_reg)
        conv11 = self.model11( torch.cat((conv0, ups), dim=1) )
        return conv11[:,1:3,::]

def eccv16ycbcr(pretrained=True):
    model = ECCVGeneratorYCbCr()
    if(pretrained):
        #import torch.utils.model_zoo as model_zoo        
        #model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
        import torch
        current_directory  = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join( current_directory, "..", "models", "eccv16_ycbcr.pth" )
        model.load_state_dict(torch.load(model_path,map_location='cpu'))
    return model
