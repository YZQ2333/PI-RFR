import torch
import torch.nn as nn
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from modules.RFRNet_temp import RFRNet_t, VGG16FeatureExtractor
import os
import time
import h5py
import numpy as np

class Temp_RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0

    # 模型参数初始化，并尝试接续训练
    def initialize_model(self, path=None, train=True):
        self.G = RFRNet_t()  # （1）先入这个网络
        self.optm_G = optim.Adam(self.G.parameters(), lr=1e-4)  # parameters()返回所有的参数，以迭代器的方式
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:  # 尝试接续训练
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr=1e-4)  # lr = 6e-4
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:  # 如果没有已经训练好的模型，则从头开始训练
            print('No trained model, from start')
            self.iter = 0

    # 如果有GPU，则将模型搬到GPU上
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")

    def train(self, train_loader, save_path, finetune=False, iters=80001):
        #   writer = SummaryWriter(log_dir="log_info")
        self.G.train(finetune=finetune)
        # 微调
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=6e-5)
            # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象
            # lambda 一个匿名函数，：后面为函数体
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        while self.iter < iters:
            for items in train_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                self.forward(masked_images, masks, gt_images)  # 重点查看函数中是否可以对二维数据进行计算，得到填补后的结果
                self.update_parameters()  # 更新参数
                self.iter += 1
                if self.iter % 50 == 0:  # 每隔50轮输出一次误差
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, loss:%.4f, time_taken:%.2f" % (self.iter, self.l1_loss_val / 50, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0

                if self.iter % 2000 == 0:  # 40000轮保存一次模型
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    # 保存模型
                    save_ckpt('{:s}/(new)temp_70_{:d}.pth'.format(save_path, self.iter), [('generator', self.G)],
                              [('optimizer_G', self.optm_G)], self.iter)
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            # 保存最终的模型
            save_ckpt('{:s}/(new)temp_70_{:s}.pth'.format(save_path, "final"), [('generator', self.G)],
                      [('optimizer_G', self.optm_G)], self.iter)

    # 测试部分
    def test(self, test_loader, result_save_path, modelth):
        self.G.eval()  # 测试时固定住BN和DropOut
        for para in self.G.parameters():  # 所有参数不用求梯度
            para.requires_grad = False
        count = 0
        dname = ['time', 'lat', 'lon']
        # save_paht = '/gpfs/home/wl_yzq/py_project/RFR_Inpainting/results/mask30/test_result/'
        for items in test_loader:
            gt_images, masks = self.__cuda__(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks] * 3, dim=1)
            fake_B, mask = self.G(masked_images, masks)
            # print(fake_B.shape(),mask.shape())
            comp_B = fake_B * (1 - masks) + gt_images * masks  # mask部分+真实部分
            n = comp_B.size(0)  # ==1
            # 计算RMSE
            loss_fn = nn.MSELoss(reduction='mean')
            rmse_loss = loss_fn(gt_images, comp_B) ** 0.5

            fake_B = fake_B.cpu().numpy()
            comp_B = comp_B.cpu().numpy()
            # print(comp_B.shape)
            if not os.path.exists('{:s}/results'.format(result_save_path)):  # 创建结果的保存路径
                os.makedirs('{:s}/results'.format(result_save_path))
            count += 1
            ################################保存南极数据####################################
            lats = np.arange(-60, -91.5, -1.5)
            lons = np.arange(0, 360, 1.5)

            h5 = h5py.File(result_save_path + '/results/station_70_{:d}_({:d})_bm.h5'.format(count, modelth),'w')
            h5.create_dataset('skt', data=comp_B[:, 0, ...])
            h5.create_dataset('lat', data=lats)
            h5.create_dataset('lon', data=lons)
            for dim in range(3):
                h5['skt'].dims[dim].label = dname[dim]
            h5.close()

            # h5_out = h5py.File(result_save_path + '/results/S_b273_30_{:d}_output({:d}).h5'.format(count, modelth), 'w')
            # h5_out.create_dataset('skt', data=fake_B[:, 0, ...])
            # h5_out.create_dataset('lat', data=lats)
            # h5_out.create_dataset('lon', data=lons)
            # for dim in range(3):
            #     h5_out['skt'].dims[dim].label = dname[dim]
            # h5_out.close()
            ##############################################################################
            print(rmse_loss)

            # 将rmse存入txt文件中
            # 写入文件
            rmse_path = './rmse_70/station_rmse_70({:d})_bm.txt'.format(modelth)
            f = open(rmse_path, 'a+')
            f.write('un:' + str(rmse_loss) + '\r\n')
            f.close()

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)  # 神经网络的输出,self.G = RFRNet()
        self.fake_B = fake_B  # 输出值
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask

    def update_parameters(self):
        self.update_G()
        self.update_D()

    def update_G(self):
        self.optm_G.zero_grad()  # 将梯度初始化为0
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()

    def update_D(self):
        return

    # 计算误差的函数
    def get_g_loss(self):
        real_B = self.real_B  # 真实数据
        fake_B = self.fake_B  # 输出
        comp_B = self.comp_B  # 填补之后

        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        tv_loss = self.TV_loss(comp_B * (1 - self.mask))  # （一致）
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)  # （一致）
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats,
                                                                                                  comp_B_feats)  # （一致）
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)  # 其余部分的误差（一致）
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))  # 缺失部分的误差（一致）
        # 加入RMSE
        loss_fn = nn.MSELoss(reduction='mean')
        rmse_loss = loss_fn(real_B, comp_B) ** 0.5

        loss_G = (tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6
                  + rmse_loss * 100
                  )

        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()  # .detach()可以阻止反向传播和梯度的计算
        return loss_G

    def l1_loss(self, f1, f2, mask=1):
        return torch.mean(torch.abs(f1 - f2) * mask)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))  # transpose(),互换指定的维度
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
