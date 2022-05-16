import torch
import torch.nn.functional as F
from torch import nn
import netCDF4
import numpy as np

class bm_PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')  # 字典 pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值
        else:
            self.multi_channel = False
        self.return_mask = True

        super(bm_PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:  # 如果为多通道图片
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        # 滑动窗口尺寸
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]  # 卷积核大小

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask=None):
        # data取出tensor的数据，舍弃其他额外信息，并且改变会回传给tensor
        if mask is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])  # last_szie=输入图片的尺寸
            with torch.no_grad():  # 被该语句 wrap 起来的部分将不会track梯度
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)  # .to()将数据强制转换类型

                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)
                # groups=1,为全连接层
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)  # mask比率在更新后变化，除法会除以tensor中的每一个数字
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)  # 将输入张量每个元素的夹紧到区间[min,max]，并返回结果到一个新张量
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)  # mul为对应元素相乘，mm为矩阵相乘
        # 加入类型转换
        if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
            self.update_mask.to(input)
            self.mask_ratio.to(input)

        #提取模式多年月平均数据
        model_data = netCDF4.Dataset('/home/yaoziqiang/E3SM/model_mean_1.nc')
        m_data = model_data['tas'][...]
        m_data = np.array(m_data[-101::-1, :])
        m_data = torch.from_numpy(m_data).float()  # 将数据转换为tensor
        m_data = m_data.unsqueeze(0)
        m_data = torch.repeat_interleave(m_data, repeats=3, dim=0)  # 格式为C，H，W
        model_data.close()
        m_data=m_data.cuda()
        # 2D卷积计算输出(可能也要改，会影响初值)
        raw_out = super(bm_PartialConv2d, self).forward(
            (torch.mul(input, mask) + (1-mask)*m_data) if mask is not None else input)  # python3可以直接写super().xxx

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view  # 加上偏差
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
