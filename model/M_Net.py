"""
    Matting network : M-Net

Author: Zhengwei Li
Date  : 2018/12/24
"""

import torch
import torch.nn as nn


class M_net(nn.Module):
    '''
        encoder + decoder
    '''

    def __init__(self, classes=2):

        super(M_net, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
        # 1/2
        self.en_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(6, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # 1/4
        self.en_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        # 1/8
        self.en_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # 1/16
        self.en_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        self.en_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        # -----------------------------------------------------------------
        # decoder  
        # ---------------------
        # 1/8
        self.de_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.deconv_1 = nn.ConvTranspose2d(128, 128, 5, 2, 2, 1, bias=False)

        # 1/4
        self.de_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 5, 2, 2, 1, bias=False)

        # 1/2
        self.de_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.deconv_3 = nn.ConvTranspose2d(32, 32, 5, 2, 2, 1, bias=False)

        # 1/1
        self.de_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.deconv_4 = nn.ConvTranspose2d(16, 16, 5, 2, 2, 1, bias=False)


        self.conv = nn.Conv2d(16, 1, 5, 1, 2, bias=False)


    def forward(self, input):

        # ----------------
        # encoder
        # --------
        x = self.en_conv_bn_relu_1(input)
        x = self.max_pooling_1(x)

        x = self.en_conv_bn_relu_2(x)
        x = self.max_pooling_2(x)

        x = self.en_conv_bn_relu_3(x)
        x = self.max_pooling_3(x)

        x = self.en_conv_bn_relu_4(x)
        x = self.max_pooling_4(x)
        # ----------------
        # decoder
        # --------
        x = self.de_conv_bn_relu_1(x)
        x = self.deconv_1(x)
        x = self.de_conv_bn_relu_2(x)
        x = self.deconv_2(x)

        x = self.de_conv_bn_relu_3(x)
        x = self.deconv_3(x)

        x = self.de_conv_bn_relu_4(x)
        x = self.deconv_4(x)

        # raw alpha pred
        out = self.conv(x)

        return out 

# --------------------------------------------------------------------------------

class M_tiny_net(nn.Module):
    '''
        encoder + decoder
    '''

    def __init__(self):

        super(M_tiny_net, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
        # 1/2
        self.en_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(6, 8, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(8),
                                       nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # 1/4
        self.en_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(8, 12, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(12),
                                       nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        # 1/8
        self.en_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(12, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # 1/16
        self.en_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(16, 24, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(24),
                                       nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        self.en_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(24),
                                       nn.ReLU())
        # -----------------------------------------------------------------
        # decoder  
        # ---------------------
        # 1/8
        self.de_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(24),
                                       nn.ReLU())
        # self.deconv_1 = nn.ConvTranspose2d(24, 24, 3, 2, 1, 1, bias=False)
        self.deconv_1 = nn.Upsample(scale_factor=2, mode='bilinear')

        # 1/4
        self.de_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(24, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        # self.deconv_2 = nn.ConvTranspose2d(16, 16, 3, 2, 1, 1, bias=False)
        self.deconv_2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # 1/2
        self.de_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(16, 12, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(12),
                                       nn.ReLU())
        # self.deconv_3 = nn.ConvTranspose2d(12, 12, 3, 2, 1, 1, bias=False)
        self.deconv_3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # 1/1
        self.de_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(12, 8, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(8),
                                       nn.ReLU())
        # self.deconv_4 = nn.ConvTranspose2d(8, 8, 3, 2, 1, 1, bias=False)
        self.deconv_4 = nn.Upsample(scale_factor=2, mode='bilinear')


        self.conv = nn.Conv2d(8, 1, 5, 1, 2, bias=False)


        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, input):

        # ----------------
        # encoder
        # --------
        x = self.en_conv_bn_relu_1(input)
        x = self.max_pooling_1(x)

        x = self.en_conv_bn_relu_2(x)
        x = self.max_pooling_2(x)

        x = self.en_conv_bn_relu_3(x)
        x = self.max_pooling_3(x)

        x = self.en_conv_bn_relu_4(x)
        x = self.max_pooling_4(x)
        # ----------------
        # decoder
        # --------
        x = self.de_conv_bn_relu_1(x)
        x = self.deconv_1(x)
        x = self.de_conv_bn_relu_2(x)
        x = self.deconv_2(x)

        x = self.de_conv_bn_relu_3(x)
        x = self.deconv_3(x)

        x = self.de_conv_bn_relu_4(x)
        x = self.deconv_4(x)

        # raw alpha pred
        out = self.conv(x)

        return out 






