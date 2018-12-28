"""
    end to end network

Author: Zhengwei Li
Date  : 2018/12/24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.M_Net import M_net
from model.T_Net import T_mv2_unet


T_net = T_mv2_unet


class net(nn.Module):
    '''
		end to end net 
    '''

    def __init__(self):

        super(net, self).__init__()

        self.t_net = T_net()
        self.m_net = M_net()



    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        trimap_softmax = F.softmax(trimap, dim=1)

        # paper: bs, fs, us
        bg, fg, unsure = torch.split(trimap_softmax, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p







