from data_loader import RescaleT, ToTensorLab
from model import U2NET, U2NETP

import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image


class SOD(object):
    def __init__(self, model_name):
        self.model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

        if model_name == 'u2net':
            print("...load U2NET---173.6 MB")
            net = U2NET(3, 1)
        elif model_name == 'u2netp':
            print("...load U2NEP---4.7 MB")
            net = U2NETP(3, 1)

        net.load_state_dict(torch.load(self.model_dir, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        self.net = net

    @staticmethod
    def normPRED(d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    def process_image(self, image):
        """
        :param image: image of type numpy array
        :return: processed image
        """
        transformer = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])
        image = transformer(image)
        image = image.unsqueeze(0)
        image = image.type(torch.FloatTensor)
        image = Variable(image)
        return image

    def get_mask(self, input_data):
        # convert input to tensor
        input_data = self.process_image(input_data)

        d1, d2, d3, d4, d5, d6, d7 = self.net(input_data)
        pred = d1[:, 0, :, :]
        pred = SOD.normPRED(pred)
        pred = pred.squeeze()
        x = pred.cpu().data.numpy()
        mask = Image.fromarray(x * 255).convert('RGB')
        return mask
