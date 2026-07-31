import os
import numpy as np
import argparse
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss

import matplotlib.pyplot as plt

from lib.networks2 import PVT_CASCADE
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import random  # Python内置随机库
import os      # 操作系统接口库
import torch   # PyTorch深度学习框架

def set_seed(seed):
    """设置所有可能用到的随机种子以确保可复现性"""
    random.seed(seed)  # Python内置的随机库
    np.random.seed(seed)  # Numpy库
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希随机化，影响Python的哈希基础设施
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的，如果不设置这个选项，可能因为cudnn的优化而使得计算结果不可复现
    torch.backends.cudnn.benchmark = False  # 当网络输入数据维度或类型上变化不大时，设置为True可以增加运行效率

seed = 42  # 或者其他你喜欢的数字
set_seed(seed)



def structure_loss(pred, mask):
    # 计算权重矩阵，用于强调边缘区域
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # 计算加权二元交叉熵损失（未降维）
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    # 对每个样本的空间维度进行加权平均
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    # 计算加权交集的面积
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    # 计算加权交并比损失（1 - 平滑后的IoU）
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # 返回批次平均后的总损失（BCE + IoU）
    return (wbce + wiou).mean()


def test(model, path, dataset):
    # 构建数据集路径：将输入路径与数据集名称拼接
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)  # 图像文件夹路径
    gt_root = '{}/masks/'.format(data_path)  # 标注文件（真值）文件夹路径
    # 设置模型为评估模式
    model.eval()
    # 通过统计标注文件数量获取测试样本总数
    num1 = len(os.listdir(gt_root))
    # 初始化测试数据集加载器
    test_loader = test_dataset(image_root, gt_root, opt.img_size)
    # 初始化Dice相似系数累计值
    DSC = 0.0
    # 遍历所有测试样本
    for i in range(num1):
        # 加载数据：返回图像张量、ground truth和文件名
        image, gt, name = test_loader.load_data()
        # 将ground truth转换为float32类型数组
        gt = np.asarray(gt, np.float32)
        # 归一化ground truth到[0,1]范围（避免除以0）
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # 模型前向传播，获取多尺度输出

        res1, res2, res3, res4 = model(image)  # forward
        # 多尺度预测融合：
        # 1. 将四个结果相加（加法融合策略）
        # 2. 使用双线性插值上采样到原始标注尺寸
        # 3. align_corners=False保持OpenCV兼容的插值方式

        res = F.upsample(res1 + res2 + res3 + res4, size=gt.shape, mode='bilinear',
                         align_corners=False)  # additive aggregation and upsampling
        res = res.sigmoid().data.cpu().numpy().squeeze()  # apply sigmoid aggregation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # 归一化到[0,1]

        # eval Dice
        input = res  # 模型预测结果
        target = np.array(gt)  # 真值标注
        N = gt.shape  # 获取维度信息（未实际使用）
        smooth = 1  # 平滑系数防止除零
        # 展平为向量计算
        input_flat = np.reshape(input, (-1))  # 预测结果展平
        target_flat = np.reshape(target, (-1))  # 真值展平
        # 计算Dice公式分子分母
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        # 格式化输出并累加
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice
        # 返回平均 Dice 和样本数

    return DSC / num1, num1


def train(train_loader, model, optimizer, epoch, test_path, model_name='PVT-CASCADE'):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- 数据准备 ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            P1, P2, P3, P4 = model(images)

            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)

            alpha, beta, gamma, zeta = 1., 1., 1., 1.

            loss = alpha * loss_P1 + beta * loss_P2 + gamma * loss_P3 + zeta * loss_P4  # current setting is for additive aggregation.

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        # ---- 训练可视化 ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
    # save model 是在每个epoch结束后保存的“last”模型，应该是最新一次的模型状态，用于恢复训练或后续评估
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        #  最后一次迭代的模型
    torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')
    # choose the best model

    global dict_plot

    if (epoch + 1) % 1 == 0:
        total_dice = 0
        total_images = 0
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            # 调用测试函数获取当前数据集的平均Dice和样本数
            dataset_dice, n_images = test(model, test_path, dataset)
            # 累计加权Dice总分（按样本量加权）
            total_dice += (n_images * dataset_dice)
            # 累计总测试样本数
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
            # 计算所有数据集的加权平均Dice（按样本量加权）
        meandice = total_dice / total_images
        # 记录整体平均Dice到test键中
        dict_plot['test'].append(meandice)
        print('Validation dice score: {}'.format(meandice))
        logging.info('Validation dice score: {}'.format(meandice))
        if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            # 路径1：保存当前最佳模型（固定文件名）
            torch.save(model.state_dict(), save_path + '' + model_name + '.pth')
            # 路径2：保存带epoch编号的历史最佳模型（备份）
            torch.save(model.state_dict(), save_path + str(epoch) + '' + model_name + '-best.pth')

        



if __name__ == '__main__':
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'PolypPVT-CASCADE'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    '''parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')'''
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./data/polyp/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./data/polyp/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth2/' + model_name + '/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_2' + model_name + '.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(2)  # set your gpu device
    model = PVT_CASCADE()
    model.cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, model_name=model_name)

