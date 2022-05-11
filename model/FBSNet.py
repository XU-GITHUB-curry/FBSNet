"FBSNet: A Fast Bilateral Symmetrical Network for Real-Time Semantic Segmentation"
""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from IPython import embed

from torch.autograd import Variable


__all__ = ["FBSNet"]



class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)




class BRUModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):  #
        super().__init__()
        #
        self.bn_relu_1 = BNPReLU(nIn)  #

        self.conv1x1_init = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=True)  #
        self.ca0 = eca_layer(nIn // 2)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)

        self.dconv1x3_l = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.dconv3x1_l = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)

        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3_r = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1_r = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.ca11 = eca_layer(nIn // 2)
        self.ca22 = eca_layer(nIn // 2)
        self.ca = eca_layer(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle_end = ShuffleBlock(groups=nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_init(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        b1 = self.ca11(br1)
        br1 = self.dconv1x3_l(b1)
        br1 = self.dconv3x1_l(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        b2 = self.ca22(br2)
        br2 = self.ddconv1x3_r(b2)
        br2 = self.ddconv3x1_r(br2)


        output = br1 + br2 + self.ca0(output )+ b1 + b2

        output = self.bn_relu_2(output)

        output = self.conv1x1(output)
        output = self.ca(output)
        out = self.shuffle_end(output + input)
        return out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool],
                               1)

        output = self.bn_prelu(output)

        return output



class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace= True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class FBSNet(nn.Module):
    def __init__(self, classes=11, block_1=5, block_2=5, block_3 = 16, block_4 = 3, block_5 = 3):
        super().__init__()

        # ---------- Encoder -------------#
        self.init_conv = nn.Sequential(
            Conv(3, 16, 3, 2, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
        )
        # 1/2
        self.bn_prelu_1 = BNPReLU(16)

        # Branch 1
        # Attention 1
        self.attention1_1 = eca_layer(16)

        # BRU Block 1
        dilation_block_1 = [1, 1, 1, 1, 1]
        self.BRU_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.BRU_Block_1.add_module("BRU_Module_1_" + str(i) ,BRUModule(16, d=dilation_block_1[i]))
        self.bn_prelu_2 = BNPReLU(16)
        # Attention 2
        self.attention2_1 = eca_layer(16)



        # Down 1  1/4
        self.downsample_1 = DownSamplingBlock(16, 64)
        # BRU Block 2
        dilation_block_2 = [1, 2, 5, 9, 17]
        self.BRU_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.BRU_Block_2.add_module("BRU_Module_2_" + str(i) ,BRUModule(64, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(64)
        # Attention 3
        self.attention3_1 = eca_layer(64)


        # Down 2  1/8
        self.downsample_2 = DownSamplingBlock(64, 128)
        # BRU Block 3
        dilation_block_3 = [1, 2, 5, 9, 1, 2, 5, 9,       2, 5, 9, 17, 2, 5, 9, 17]
        self.BRU_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.BRU_Block_3.add_module("BRU_Module_3_" + str(i), BRUModule(128, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(128)
        # Attention 4
        self.attention4_1 = eca_layer(128)





        # --------------Decoder   ----------------- #
        # Up 1 1/4
        self.upsample_1 = UpsamplerBlock(128, 64)

        # BRU Block 4
        dilation_block_4 = [1, 1, 1]
        self.BRU_Block_4 = nn.Sequential()
        for i in range(0, block_4):
            self.BRU_Block_4.add_module("BRU_Module_4_" + str(i), BRUModule(64, d=dilation_block_4[i]))
        self.bn_prelu_5 = BNPReLU(64)
        self.attention5_1 = eca_layer(64)
        # self.attention5_1 = CoordAtt(64,64)



        # Up 2 1/2
        self.upsample_2 = UpsamplerBlock(64, 32)
        # BRU Block 5
        dilation_block_5 = [1, 1, 1]
        self.BRU_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.BRU_Block_5.add_module("BRU_Module_5_" + str(i), BRUModule(32, d=dilation_block_5[i]))
        self.bn_prelu_6 = BNPReLU(32)
        self.attention6_1 = eca_layer(32)




        # Branch 2
        self.conv_sipath1 = Conv(16, 32, 3, 1, 1, bn_acti=True)
        self.conv_sipath2 = Conv(32, 128, 3, 1, 1, bn_acti=True)
        self.conv_sipath3 = Conv(128, 32, 3, 1, 1, bn_acti=True)

        self.atten_sipath = SpatialAttention()
        self.bn_prelu_8 = BNPReLU(32)
        self.bn_prelu_9 = BNPReLU(32)

        self.endatten = CoordAtt(32, 32)

        self.output_conv = nn.ConvTranspose2d(32, classes, 2, stride=2, padding=0, output_padding=0, bias=True)




    def forward(self, input):

        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        # Branch1
        output1 = self.attention1_1(output0)

        # block1
        output1 = self.BRU_Block_1(output1)
        output1 = self.bn_prelu_2(output1)
        output1 = self.attention2_1(output1)

        # down1
        output1 = self.downsample_1(output1)

        # block2
        output1 = self.BRU_Block_2(output1)
        output1 = self.bn_prelu_3(output1)
        output1 = self.attention3_1(output1)

        # down2
        output1 = self.downsample_2(output1)

        # block3
        output2 = self.BRU_Block_3(output1)
        output2 = self.bn_prelu_4(output2)
        output2 = self.attention4_1(output2)


        # ---------- Decoder ----------------
        # up1
        output = self.upsample_1(output2)

        # block4
        output = self.BRU_Block_4(output)
        output = self.bn_prelu_5(output)
        output = self.attention5_1(output)

        # up2
        output = self.upsample_2(output)

        # block5
        output = self.BRU_Block_5(output)
        output = self.bn_prelu_6(output)
        output = self.attention6_1(output)


        # Detail Branch
        output_sipath = self.conv_sipath1(output0)
        output_sipath = self.conv_sipath2(output_sipath)
        output_sipath = self.conv_sipath3(output_sipath)
        output_sipath = self.atten_sipath(output_sipath)

        # Feature Fusion Module
        output = self.bn_prelu_8(output + output_sipath)

        # Feature Augment Module
        output = self.endatten(output)

        # output projection
        out = self.output_conv(output)

        return out




"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FBSNet(classes=19).to(device)
    summary(model, (3, 512, 1024))
