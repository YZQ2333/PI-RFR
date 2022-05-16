import argparse
import os
from model_temp import Temp_RFRNetModel
from dataset_temp import Tem_Dataset
from torch.utils.data import DataLoader

def run():
    parser = argparse.ArgumentParser()
    #训练数据的位置
    parser.add_argument('--data_root', type=str, default='/home/yaoziqiang/py_project/train&test_data/train_mavg')
    parser.add_argument('--mask_root', type=str, default='/home/yaoziqiang/py_project/mask_level/70_360')
    ################# 测试数据路径#########################
    # data
    parser.add_argument('--test_data', type=str, default='/home/yaoziqiang/climate_data/station_data/test_data/data')
    #/home/yaoziqiang/py_project/train&test_data/test_mavg
    #/home/yaoziqiang/climate_data/station_data/test_data/data
    # mask
    parser.add_argument('--test_mask', type=str, default='/home/yaoziqiang/climate_data/station_data/test_data/mask')
    #/home/yaoziqiang/py_project/mask_level/70
    #/home/yaoziqiang/climate_data/station_data/test_data/mask
    #####################################################
    #模型保存位置
    parser.add_argument('--model_save_path', type=str, default='/home/yaoziqiang/py_project/RFR/results/mask70/bavg_ckpt')
    #测试结果保存位置
    parser.add_argument('--result_save_path', type=str, default='/home/yaoziqiang/py_project/RFR/results/mask70/station_data')

    parser.add_argument('--num_iters', type=int, default=80001)#训练迭代次数450000
    #训练时的模型位置
    parser.add_argument('--model_path', type=str, default="/home/yaoziqiang/py_project/RFR/results/mask70/bavg_ckpt/(new)temp_70_76000ft.pth")#可能要改

    #测试时的模型位置
    parser.add_argument('--test_model_path', type=str,default="/home/yaoziqiang/py_project/RFR/results/mask70/bavg_ckpt/")  # 可能要改
    #/home/yaoziqiang/py_project/RFR/results/mask70/mdata_ckpt/

    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id#指定要使用的GPU
    model = Temp_RFRNetModel()

    if args.test:
        dataloader = DataLoader(
            Tem_Dataset(args.test_data, args.test_mask,training=False))#默认不打乱数据
        #测试某个模型
        model.initialize_model(args.test_model_path + '(new)temp_70_80000.pth', False)
        model.cuda()
        model.test(dataloader, args.result_save_path, 80000)
        print('{:d}th model was tested'.format(80000))
        for i in range(76000, 162000, 2000):
            model.initialize_model(args.test_model_path+'temp_70_{:d}ft.pth'.format(i), False)
            model.cuda()
            model.test(dataloader, args.result_save_path, i)
            print('{:d}th model was tested'.format(i))
        for j in range(76000, 162000, 2000):
            path = './rmse_70/b254_rmse_70({:d})ft.txt'.format(j)
            f = open(path,'r')
            sum = 0.0
            for line in f.readlines():
                sum += float(line[10:16])#
            avg = sum/120
            avgrmse_path = './avg_b254_70ft.txt'
            f_w = open(avgrmse_path, 'a+')
            f_w.write('avgrmse:' + str(avg) + '\r\n')
    else:
        model.initialize_model(args.model_path, True)
        model.cuda()
        #打开训练数据，并加载如dataloader
        dataloader = DataLoader(Tem_Dataset(args.data_root, args.mask_root),
                                batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        model.train(dataloader, args.model_save_path, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()
