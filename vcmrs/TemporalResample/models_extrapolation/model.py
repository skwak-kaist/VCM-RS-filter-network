from .EFNet import *
import torch
import numpy as np
from e2evc.Utils import ctx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, load_path=None):
        self.EFNet = EFNet()
        self.device()
        self.load_model(load_path,-1)

    def load_model(self, load_path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v.double()
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if torch.cuda.is_available():
            state_dict = torch.load(load_path)
        else:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        if rank <= 0:
            state_dict = convert(state_dict)
        model_state_dict = self.EFNet.state_dict()
        try:
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict[k]
        except Exception as e:
            print(e)
        self.EFNet.load_state_dict(model_state_dict)

    def eval(self):
        self.EFNet.eval()

    @ctx.int_conv()
    def predict_frame(self, img_0, img_1, padding, scale_list=[4, 4, 4, 2, 2, 2, 1, 1, 1]):
        if torch.cuda.is_available():
            img_0 = img_0.to('cuda')
            img_1 = img_1.to('cuda')
        else:
            img_0 = img_0.to('cpu')
            img_1 = img_1.to('cpu')
        img = torch.cat((img_0, img_1), 1)

        self.EFNet.eval()
        with ctx.int_conv():
            pred = self.EFNet(img, scale=scale_list, training=False)  # 1, 3, H, W
        # pred = np.array(pred.cpu().squeeze() * 255).transpose(1, 2, 0)
        return pred

    def device(self):
        self.EFNet.to(device)