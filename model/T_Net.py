"""
    Trimap generation : T-Net

Author: Zhengwei Li
Date  : 2018/12/24
"""


import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class mobilenet_v2(nn.Module):
    def __init__(self, nInputChannels=3):
        super(mobilenet_v2, self).__init__()
        # 1/2
        self.head_conv = nn.Sequential(nn.Conv2d(nInputChannels, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        # 1/2
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 
        self.block_2 = nn.Sequential( 
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 
        self.block_3 = nn.Sequential( 
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/16 
        self.block_4 = nn.Sequential( 
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)            
            )
        # 1/16
        self.block_5 = nn.Sequential( 
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)          
            )
        # 1/32 
        self.block_6 = nn.Sequential( 
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)          
            )
        # 1/32
        self.block_7 = InvertedResidual(160, 320, 1, 6)

    def forward(self, x):
        x = self.head_conv(x)
        # 1/2
        s1 = self.block_1(x)
        # 1/4 
        s2 = self.block_2(s1)
        # 1/8
        s3 = self.block_3(s2)
        # 1/16
        s4 = self.block_4(s3)
        s4 = self.block_5(s4)
        # 1/32
        s5 = self.block_6(s4)
        s5 = self.block_7(s5)

        return s1, s2, s3, s4, s5


class T_mv2_unet(nn.Module):
    '''
        mmobilenet v2 + unet 

    '''

    def __init__(self, classes=3):

        super(T_mv2_unet, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
        self.feature = mobilenet_v2()

        # -----------------------------------------------------------------
        # decoder 
        # ---------------------

        self.s5_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(320, 96, 3, 1, 1),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU())
        self.s4_fusion = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1),
                                       nn.BatchNorm2d(96))

        self.s4_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(96, 32, 3, 1, 1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        self.s3_fusion = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                       nn.BatchNorm2d(32))

        self.s3_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(32, 24, 3, 1, 1),
                                        nn.BatchNorm2d(24),
                                        nn.ReLU())
        self.s2_fusion = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1),
                                       nn.BatchNorm2d(24))

        self.s2_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(24, 16, 3, 1, 1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())
        self.s1_fusion = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1),
                                       nn.BatchNorm2d(16))

        self.last_conv = nn.Conv2d(16, classes, 3, 1, 1)
        self.last_up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input):

        # -----------------------------------------------
        # encoder 
        # ---------------------
        s1, s2, s3, s4, s5 = self.feature(input)
        # -----------------------------------------------
        # decoder
        # ---------------------
        s4_ = self.s5_up_conv(s5)
        s4_ = s4_ + s4
        s4 = self.s4_fusion(s4_)

        s3_ = self.s4_up_conv(s4)
        s3_ = s3_ + s3
        s3 = self.s3_fusion(s3_)

        s2_ = self.s3_up_conv(s3)
        s2_ = s2_ + s2
        s2 = self.s2_fusion(s2_)

        s1_ = self.s2_up_conv(s2)
        s1_ = s1_ + s1
        s1 = self.s1_fusion(s1_)

        out = self.last_conv(s1)

        return out


# --------------------------------------------------------------------------------

def conv_bn_act(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )
def bn_act(inp):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        nn.PReLU(inp)
    )
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(make_dense, self).__init__()
        
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(growthRate)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x_ = self.bn(self.conv(x))
        out = self.act(x_)
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, reset_channel=False):
        super(DenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out

# ResidualDenseBlock
class ResidualDenseBlock(nn.Module):
    def __init__(self, nIn, s=4, add=True):

        super(ResidualDenseBlock, self).__init__()

        n = int(nIn//s) 

        self.conv =  nn.Conv2d(nIn, n, 1, stride=1, padding=0, bias=False)
        self.dense_block = DenseBlock(n, nDenselayer=(s-1), growthRate=n)

        self.bn = nn.BatchNorm2d(nIn)
        self.act = nn.PReLU(nIn)

        self.add = add

    def forward(self, input):

        # reduce
        inter = self.conv(input)
        combine =self.dense_block(inter)

        # if residual version
        if self.add:
            combine = input + combine

        output = self.act(self.bn(combine))
        return output

class RD_FPNnet(nn.Module):

    def __init__(self, classes=3):

        super(RD_FPNnet, self).__init__()

        # -----------------------------------------------------------------
        # encoder 
        # ---------------------
        # input cascade
        self.cascade = nn.AvgPool2d(3, stride=2, padding=1)
        # 1/2
        self.head_conv = conv_bn_act(3, 24, kernel_size=3, stride=2, padding=1)
        self.stage_0 = ResidualDenseBlock(24, s=3, add=True)

        # 1/4
        self.ba_1 = bn_act(24+3)
        self.down_1 = conv_bn_act(24+3, 24, kernel_size=3, stride=2, padding=1)
        self.stage_1 = nn.Sequential(ResidualDenseBlock(24, s=3, add=True),
                                     ResidualDenseBlock(24, s=3, add=True))
                                     # ResidualDenseBlock(24, s=2, add=True))
        # 1/8
        self.ba_2 = bn_act(48+3)
        self.down_2 = conv_bn_act(48+3, 48, kernel_size=3, stride=2, padding=1)
        self.stage_2 = nn.Sequential(ResidualDenseBlock(48, s=4, add=True),
                                     ResidualDenseBlock(48, s=4, add=True))
                                     # ResidualDenseBlock(48, s=3, add=True),
                                     # ResidualDenseBlock(48, s=3, add=True))s
        # 1/16
        self.ba_3 = bn_act(96+3)
        self.down_3 = conv_bn_act(96+3, 96, kernel_size=3, stride=2, padding=1)
        self.stage_3 = nn.Sequential(ResidualDenseBlock(96, s=4, add=True),
                                     ResidualDenseBlock(96, s=4, add=True))
                                     # ResidualDenseBlock(96, s=3, add=True),
                                     # ResidualDenseBlock(96, s=3, add=True))
        # 1/32
        self.ba_4 = bn_act(192+3)
        self.down_4 = conv_bn_act(192+3, 192, kernel_size=3, stride=2, padding=1)
        self.stage_4 = nn.Sequential(ResidualDenseBlock(192, s=6, add=True),
                                     ResidualDenseBlock(192, s=6, add=True))
                                     # ResidualDenseBlock(192, s=3, add=True),
                                     # ResidualDenseBlock(192, s=3, add=True)) 

        # -----------------------------------------------------------------
        # heatmap 
        # ---------------------
        self.classifier = nn.Conv2d(192, classes, 1, stride=1, padding=0, bias=True)

        # -----------------------------------------------------------------
        # decoder 
        # ---------------------

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.stage3_down = nn.Conv2d(96, classes, kernel_size=1, stride=1, padding=0)
        self.conv_3 = nn.Conv2d(classes, classes, 3, 1, 1, bias=True)                           
                                
        self.stage2_down = nn.Conv2d(48, classes, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(classes, classes, 3, 1, 1, bias=True)
 
        self.stage1_down = nn.Conv2d(24, classes, kernel_size=1, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(classes, classes, 3, 1, 1, bias=True)


        self.stage0_down = nn.Conv2d(24, classes, kernel_size=1, stride=1, padding=0)
        self.conv_0 = nn.Conv2d(classes, classes, 3, 1, 1, bias=True)
            
        self.last_up = nn.Upsample(scale_factor=2, mode='bilinear')


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

        input_cascade1 = self.cascade(input)
        input_cascade2 = self.cascade(input_cascade1)
        input_cascade3 = self.cascade(input_cascade2)
        input_cascade4 = self.cascade(input_cascade3)
        x = self.head_conv(input)
        # 1/2
        s0 = self.stage_0(x)

        # ---------------
        s1_0 = self.down_1(self.ba_1(torch.cat((input_cascade1, s0),1)))
        s1 = self.stage_1(s1_0)

        # ---------------
        s2_0 = self.down_2(self.ba_2(torch.cat((input_cascade2, s1_0, s1),1)))
        s2 = self.stage_2(s2_0)

        # ---------------
        s3_0 = self.down_3(self.ba_3(torch.cat((input_cascade3, s2_0, s2),1)))
        s3 = self.stage_3(s3_0)

        # ---------------
        s4_0 = self.down_4(self.ba_4(torch.cat((input_cascade4, s3_0, s3),1)))
        s4 = self.stage_4(s4_0)


        # -------------------------------------------------------

        heatmap = self.classifier(s4)
        # -------------------------------------------------------


        p3 = self.up(heatmap) + self.stage3_down(s3)
        p3 = self.conv_3(p3)
        p2 = self.up(p3) + self.stage2_down(s2)
        p2 = self.conv_2(p2)
        p1 = self.up(p2) + self.stage1_down(s1)
        p1 = self.conv_1(p1)
        p0 = self.up(p1) + self.stage0_down(s0)
        p0 = self.conv_0(p0)
       
        out = self.last_up(p0)  


        return out
