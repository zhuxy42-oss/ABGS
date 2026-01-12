import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.layers.mesh_conv import MeshConv
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
from util.calculate_gradient import hausdorff_distance
import numpy as np
from torch_geometric.nn import GCNConv, GINEConv, BatchNorm, global_mean_pool, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dropout_adj

###############################################################################
# Helper Functions
###############################################################################

class BinaryClassAccuracyLoss(nn.Module):
    def __init__(self, pos_weight=0.5, neg_weight=0.5, eps=1e-8):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.eps = eps
        
    def forward(self, pred, label):
        """
        Args:
            pred: 预测概率 [batch_size, edge_num] (0~1之间)
            target: 真实标签 [batch_size, edge_num] (0或1)
        Returns:
            组合损失值
        """
        total_correct = 0
        total_ones_match = 0
        total_zeros_match = 0
        total_ones = 0
        total_zeros = 0
        total_elements = 0
        batch = len(pred)
        # for i in range(batch):
        #     predict = pred[i:i+1]
        #     target = label[i]

        #     edge_len = 3000
        #     # Convert predictions to binary (0 or 1)
        #     predict_binary = (predict > 0.5).float()
            
        #     # Calculate correct predictions
        #     correct = (predict_binary == target).float()
        #     total_correct += correct.sum().item()
            
        #     # Calculate where both are 1
        #     ones_match = (predict_binary * target).sum().item()
        #     total_ones_match += ones_match
        #     total_ones += target.sum().item()
            
        #     # Calculate where both are 0
        #     zeros_match = ((1 - predict_binary) * (1 - target)).sum().item()
        #     total_zeros_match += zeros_match
        #     total_zeros += (1 - target).sum().item()

            # total_elements += edge_len
        
        predict_binary = (pred > 0.5).float()
            
        # Calculate correct predictions
        correct = (predict_binary == label).float()
        total_correct = correct.sum().item()
            
        # Calculate where both are 1
        ones_match = (predict_binary * label).sum().item()
        total_ones_match += ones_match
        total_ones = label.sum().item()
            
        # Calculate where both are 0
        zeros_match = ((1 - predict_binary) * (1 - label)).sum().item()
        total_zeros_match += zeros_match
        total_zeros = (1 - label).sum().item()
        # Avoid division by zero
        ones_prob = total_ones_match / total_ones if total_ones > 0 else 0
        zeros_prob = total_zeros_match / total_zeros if total_zeros > 0 else 0
            
        # 计算损失 (1 - 加权准确率)
        loss = torch.tensor([1 - (self.pos_weight * ones_prob + self.neg_weight * zeros_prob)], device='cuda').requires_grad_(True)
        return loss

class SimplificationLoss(nn.Module):
    def __init__(self):
        super(SimplificationLoss, self).__init__()
    
    def forward(self, origin_mesh, result_mesh, target_mesh):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss = torch.zeros(size=(len(origin_mesh), 1)).to(device)
        for i in range(len(origin_mesh)):
            # print(origin_mesh[i].vertices.requires_grad)
            # print(result_mesh[i].vertices.requires_grad)
            # print(target_mesh[i].vertices.requires_grad)
            loss1 = hausdorff_distance(result_mesh[i], target_mesh[i])
            # loss2 = target_mesh[i].get_gradinet() - origin_mesh[i].get_gradinet()
            loss[i] += (loss1)
        sum_loss = loss.mean()
        return sum_loss

class EdgeCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(EdgeCrossEntropyLoss, self).__init__()
    def forward(self, edge_prob, mesh, label):
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.L1Loss()
        Loss = []
        for i in range(len(mesh)):
            # true_prob = edge_prob[i, 0:mesh[i].edges_count]
            Loss.append(criterion(edge_prob, label[i]))
        return torch.mean(torch.stack(Loss))
class M2MRegressionLoss(nn.Module):
    def __init__(self):
        super(M2MRegressionLoss, self).__init__()
    def forward(self, pred, label):
        criterion = nn.MSELoss()
        Loss = []
        batch = pred.size(0)
        # pred = F.sigmoid(pred)
        for i in range(batch):
            edge_len = label[i].size(1)
            true_pred = pred[i, 0:edge_len].unsqueeze(0)
            true_label = label[i].requires_grad_(True)
            loss = criterion(true_pred, true_label)

            Loss.append(loss)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)
    
class M2MRegressionLoss1(nn.Module):
    def __init__(self):
        super(M2MRegressionLoss1, self).__init__()
    def forward(self, pred, label, mask):
        Loss = []
        batch = pred.size(0)
        for i in range(batch):
            edge_len = mask[i].size(1)
            true_pred = pred[i:i+1, 0:edge_len]
            true_label = label[i:i+1, 0:edge_len]
            lossing = torch.mean(torch.abs(true_label - true_pred) * (1 + 5 * mask[i]))    
            Loss.append(lossing)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)
    
class M2MClassifyLoss(nn.Module):
    def __init__(self):
        super(M2MClassifyLoss, self).__init__()
    def forward(self, pred, label):
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device='cuda'))    #pos_weight=torch.tensor([4.0], device=device)
        criterion = nn.BCELoss()
        Loss = []
        batch = len(pred)
        for i in range(batch):
            edge_len = label[i].size(1)
            true_pred = pred[i:i+1, 0:edge_len]
            true_label = label[i]
            loss = criterion(true_pred, true_label)
            Loss.append(loss)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)

class M2MRankLoss(nn.Module):
    def __init__(self, temperature=1.0, eps=1e-10):
        super(M2MRankLoss, self).__init__()
        self.temp = temperature  # 温度参数控制分布尖锐程度
        self.eps = eps
    def forward(self, pred, label):
        Loss = []
        for i in range(pred.size(0)):
            target_batch = label[i].requires_grad_(True)
            edge_len = target_batch.size(1)
            pred_batch = pred[i, 0:edge_len].unsqueeze(0)
            loss = -torch.sum(target_batch * torch.log(pred_batch + self.eps), dim=-1)
            Loss.append(loss)
        Lossing = torch.stack(Loss)
        return torch.mean(Lossing).clone().detach().requires_grad_(True)

class M2MRankLoss1(nn.Module):
    def __init__(self, margin=0.0001):
        super(M2MRankLoss1, self).__init__()
    def forward(self, pred, label):
        batch_size, _ = pred.shape
        loss = []
        for i in range(batch_size):
            edge_len = len(label[i])
            pred_batch = pred[i, 0:edge_len]
            target_batch = label[i].squeeze(0)

            target_diff = target_batch.unsqueeze(1) - target_batch.unsqueeze(0)
            pred_diff = pred_batch.unsqueeze(1) - pred_batch.unsqueeze(0)
            mask = (target_diff > 0).float()
            batch_loss = torch.mean(mask * torch.relu(pred_diff + self.margin))

            # 累加损失
            loss.append(batch_loss)

        return torch.mean(torch.stack(loss))


class EdgeRankLoss(nn.Module):
    def __init__(self):
        super(EdgeRankLoss, self).__init__()

    def forward(self, pred, target, margin):
        batch_size, edge = pred.shape

        # 初始化损失
        loss = []
        # 遍历每个 batch
        for i in range(batch_size):
            # 提取当前 batch 的预测值和目标值
            pred_batch = pred[i]
            target_batch = target[i]

            # 计算所有元素对的差异
            target_diff = target_batch.unsqueeze(1) - target_batch.unsqueeze(0)
            pred_diff = pred_batch.unsqueeze(1) - pred_batch.unsqueeze(0)

            # 仅保留 target[i] > target[j] 的元素对
            mask = (target_diff > 0).float()
            batch_loss = torch.mean(mask * torch.relu(pred_diff + margin))

            # 累加损失
            loss.append(batch_loss)
            

        return torch.mean(torch.stack(loss))

class ListNetLoss(nn.Module):
    def __init__(self, temperature=1.0, eps=1e-10):
        super().__init__()
        self.temp = temperature  # 温度参数控制分布尖锐程度
        self.eps = eps

    def forward(self, pred, target):
        """
        参数:
            pred: 预测分数 (batch_size, num_edges)
            target: 真实分数 (batch_size, num_edges)
        """
        # 生成概率分布（带温度参数的softmax）
        pred_p = pred / self.temp
        target_p = target / self.temp
        
        # 计算交叉熵损失（KL散度）
        loss = -torch.sum(target_p * torch.log(pred_p + self.eps), dim=-1)
        return torch.mean(loss)

class SpearmanLoss(nn.Module):
    def __init__(self, regularization='l2', reg_strength=1.0):
        super().__init__()
        self.reg = regularization
        self.reg_strength = reg_strength
        
    def differentiable_ranking(self, scores):
        """ 使用神经排序近似实现可微排序 """
        # 使用NeuralSort算法（需安装torchsort）
        from torchsort import soft_rank
        return soft_rank(scores, regularization=self.reg, regularization_strength=self.reg_strength)

    def forward(self, pred, target):
        """
        参数:
            pred: 预测分数 (batch_size, num_edges)
            target: 真实分数 (batch_size, num_edges)
        """
        # 计算可微排序
        pred_rank = self.differentiable_ranking(pred)
        target_rank = self.differentiable_ranking(target)
        
        # 计算Spearman相关系数
        pred_centered = pred_rank - pred_rank.mean(dim=-1, keepdim=True)
        target_centered = target_rank - target_rank.mean(dim=-1, keepdim=True)
        
        covariance = (pred_centered * target_centered).sum(dim=-1)
        pred_std = torch.sqrt((pred_centered**2).sum(dim=-1))
        target_std = torch.sqrt((target_centered**2).sum(dim=-1))
        
        spearman = covariance / (pred_std * target_std + 1e-8)
        
        # 将相关系数转换为损失（最大化相关系数 → 最小化 1 - rho）
        return 1.0 - torch.mean(spearman)

def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm2d':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args

class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)

    # net = net.to(torch.device('cpu'))

    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)

    if arch == 'mconvnet':
        net = MeshConvNet(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    elif arch == 'meshunet':
        down_convs = [input_nc] + ncf
        up_convs = ncf[::-1] + [nclasses]
        pool_res = [ninput_edges] + opt.pool_res
        net = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                                 transfer_data=True)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_loss(opt):
    if opt.dataset_mode == 'classification':
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif opt.dataset_mode == 'simplification':
        loss = SimplificationLoss()
    return loss




##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################

# class MeshConvNet(nn.Module):
#     """Network for learning a global shape descriptor (classification)
#     """
#     def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
#                  nresblocks=3):
#         super(MeshConvNet, self).__init__()
#         self.k = [nf0] + conv_res
#         self.res = [input_res] + pool_res
#         norm_args = get_norm_args(norm_layer, self.k[1:])

#         for i, ki in enumerate(self.k[:-1]):
#             setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
#             setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
#             setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))


#         # self.gp = torch.nn.AvgPool1d(self.res[-1])
#         # self.gp = torch.nn.MaxPool1d(self.res[-1])
#         self.fc1 = nn.Linear(self.k[-1] * self.res[-1], fc_n)
#         self.fc2 = nn.Linear(fc_n, self.res[-1])
#         self.activate = nn.Sigmoid()
#         # self.activate = nn.ReLU()

#     def forward(self, x, mesh):

#         for i in range(len(self.k) - 1):
#             x = getattr(self, 'conv{}'.format(i))(x, mesh)
#             x = F.relu(getattr(self, 'norm{}'.format(i))(x))
#             # x = getattr(self, 'pool{}'.format(i))(x, mesh)

#         # x = self.gp(x)
#         # x = x.view(-1, self.k[-1])

#         # x = self.fc(x.transpose(1, 2))
#         # x = self.activate(x.squeeze(-1))

#         x = x.squeeze(-1).view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = 0.5 * self.activate(x) + 0.5
#         return x

# class MeshConvNet(nn.Module):
#     """Network for learning a global shape descriptor (classification)
#     """
#     def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
#                  nresblocks=3):
#         super(MeshConvNet, self).__init__()
#         self.k = [nf0] + conv_res
#         self.res = [input_res] + pool_res
#         norm_args = get_norm_args(norm_layer, self.k[1:])

#         for i, ki in enumerate(self.k[:-1]):
#             setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
#             setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
#             c


#         self.gp = torch.nn.AvgPool1d(self.res[-1])
#         # self.gp = torch.nn.MaxPool1d(self.res[-1])
#         self.fc1 = nn.Linear(self.k[-1], fc_n)
#         self.fc2 = nn.Linear(fc_n, nclasses)

#     def forward(self, x, mesh):

#         for i in range(len(self.k) - 1):
#             x = getattr(self, 'conv{}'.format(i))(x, mesh)
#             x = F.relu(getattr(self, 'norm{}'.format(i))(x))
#             x = getattr(self, 'pool{}'.format(i))(x, mesh)

#         x = self.gp(x)
#         x = x.view(-1, self.k[-1])

#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

    def forward(self, x, mesh):
        mid = []
        for i in range(len(self.k) - 1):
            mid.append(x)
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

            # if i > 3 and i != 5:
            #     x = x + mid[5 - i]
        return F.sigmoid(x.squeeze(-1).squeeze(1))

class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))
            # setattr(self, 'dropout{}'.format(i + 1), nn.Dropout(0.5))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data)
        # self.linear = nn.Linear(in_features=2 * pools[0], out_features=pools[0])
        # self.activate = nn.ReLU()
        # self.activate = nn.Sigmoid()
        # self.activate = nn.Tanh()

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        # fe = fe.squeeze(1)
        # return self.activate(fe)
        return fe.squeeze(1)

    def __call__(self, x, meshes):
        return self.forward(x, meshes)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0):
        super(DownConv, self).__init__()
        self.bn = []
        self.pool = None
        self.conv1 = MeshConv(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = None
        if self.pool:
            before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = MeshConv(in_channels, out_channels)
        if transfer_data:
            self.conv1 = MeshConv(2 * out_channels, out_channels)
        else:
            self.conv1 = MeshConv(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
            # self.convs.append(nn.Dropout(0.5))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
            # self.up_convs.append(nn.Dropout(0.5))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

def reset_params(model): # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class EdgeClassificationGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeClassificationGNN, self).__init__()
        # 节点特征提取
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征提取
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, hidden_dim]

        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 边分类logits
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        return edge_logits  # 压缩为[num_edges]以适应BCEWithLogitsLoss

class EdgeClassificationGNN1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeClassificationGNN1, self).__init__()
        # 节点特征提取 - 使用GINEConv替代GCNConv
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # # 添加自环边（可选，但通常有助于GINEConv）
        # edge_index, edge_attr = add_self_loops(
        #     edge_index, 
        #     edge_attr=edge_attr,
        #     fill_value=0.0 if edge_attr is None else edge_attr.mean(dim=0),
        #     num_nodes=x.size(0)
        # )
        
        # 节点特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)  # [num_nodes, hidden_dim]

        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 边分类logits
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        return edge_logits
    
# class EdgeClassificationGNN2(nn.Module):
#     def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
#         super(EdgeClassificationGNN2, self).__init__()
        
#         # 1. 边特征预处理层
#         self.edge_encoder = nn.Sequential(
#             GCNConv(node_feat_dim, hidden_dim),
#             nn.ReLU(),
#             GCNConv(hidden_dim, hidden_dim)
#         ) if edge_feat_dim > 0 else None
        
#         # 2. 节点特征提取 - 使用带BatchNorm的GINEConv
#         self.conv1 = GINEConv(
#             nn=nn.Sequential(
#                 nn.Linear(node_feat_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#             ),
#             edge_dim=hidden_dim if edge_feat_dim > 0 else None
#         )
        
#         self.conv2 = GINEConv(
#             nn=nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#             ),
#             edge_dim=hidden_dim if edge_feat_dim > 0 else None
#         )
        
#         # 3. 更稳健的边分类器
#         self.edge_classifier = nn.Sequential(
#             nn.Linear(2 * hidden_dim + (hidden_dim if edge_feat_dim > 0 else 0), hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Tanh(),
#             nn.Dropout(p=0.6),  # 增加dropout
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.LayerNorm(hidden_dim//2),
#             nn.Tanh(),
#             nn.Dropout(p=0.5),
#             nn.Linear(hidden_dim//2, 1)
#         )

#         # 4. 初始化权重
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, data, edge_dropout_rate=0.2):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
#         # 1. 边特征编码
#         if edge_attr is not None and self.edge_encoder is not None:
#             edge_attr = self.edge_encoder(edge_attr)
        
#         # 4. 图卷积层
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv1(x, edge_index, edge_attr)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_attr)
        
#         # 5. 边表示构建
#         src, dst = edge_index
#         h_src = x[src]
#         h_dst = x[dst]
#         edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
#         # 6. 边分类logits
#         edge_logits = self.edge_classifier(edge_repr)
#         return edge_logits

class EdgeClassificationGNN2(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeClassificationGNN2, self).__init__()
        # 节点特征提取
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 边特征提取
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if edge_feat_dim > 0 else None
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # # 验证输入
        # assert edge_index.max() < x.size(0), "Edge index contains invalid node indices"
        # assert x.size(1) == self.conv1.in_channels, "Input feature dimension mismatch"
        # if self.edge_encoder is not None and edge_attr is not None:
        #     assert edge_attr.size(1) == self.edge_feat_dim, "Edge feature dimension mismatch"

        # print(edge_index.max().item(), x.size(0))

        # 节点特征提取
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, hidden_dim]

        edge_feature = self.edge_encoder(edge_attr)

        # 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_feature], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 边分类logits
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        return edge_logits  # 压缩为[num_edges]以适应BCEWithLogitsLoss
    
class EdgeRegressionGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(EdgeRegressionGNN, self).__init__()
        
        # 特征编码器（添加输入标准化）
        self.node_encoder = nn.Sequential(
            nn.BatchNorm1d(node_feat_dim),  # 输入标准化
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Swish激活函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.BatchNorm1d(edge_feat_dim),  # 输入标准化
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 图卷积模块（使用带边缘信息的GINEConv）
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ), eps=0.1, edge_dim=hidden_dim)
        
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ), eps=0.1, edge_dim=hidden_dim)
        
        # 非负输出处理模块
        self.output_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 保证输出初始值为1附近的偏置初始化
        self.output_head[-1].bias.data.fill_(0.1)  # 初始输出≈exp(0.1)≈1.1
        
        # 特征交互门控机制
        self.feature_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Sigmoid()  # 输出[0,1]作为权重
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 特征编码
        h_nodes = self.node_encoder(x)
        h_edges = self.edge_encoder(edge_attr)
        
        # 图卷积（融合边特征）
        x1 = self.conv1(h_nodes, edge_index, h_edges)
        x2 = self.conv2(F.silu(x1), edge_index, h_edges)
        x = x1 + x2  # 残差连接
        
        # 构建边表示
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], h_edges], dim=-1)
        
        # 门控特征交互
        # 方案2：分通道处理
        gate = edge_repr.chunk(3, dim=1)  # 分成3个[N,64]
        enhanced_repr = torch.cat([
            x[src] * gate[0],
            x[dst] * gate[1],
            h_edges * gate[2]
        ], dim=1)
        
        # 保证正输出的两种方案（任选其一）：
        # 方案1：Softplus激活（推荐）
        raw_output = self.output_head(enhanced_repr).squeeze(-1)
        # output = F.softplus(raw_output) + 0.5  # 确保输出≥1
        
        # 方案2：对数域变换（适用于大范围数值）
        # output = torch.exp(raw_output) + 1.0  # exp(x)+1 ≥1
        
        return raw_output
    
class MultiTaskEdgeGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(MultiTaskEdgeGNN, self).__init__()
        # 共享的节点特征提取层
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 边分类器（输出单个logit值）
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维logits
        )
        
        # 边回归器（使用分类结果作为注意力）
        self.edge_regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)  # 输出1维回归值
        )
        
        # 注意力权重生成器
        self.attention_generator = nn.Sequential(
            nn.Linear(1, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),
            nn.ReLU()  # 输出0-1之间的注意力权重
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 1. 共享节点特征提取
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, hidden_dim]

        # 2. 边表示构建
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_src, h_dst], dim=1)
        
        # 3. 边分类任务
        edge_logits = self.edge_classifier(edge_repr)  # [num_edges, 1]
        
        # # 4. 基于分类结果生成注意力权重（仅对正样本关注）
        # with torch.no_grad():
        #     # 使用sigmoid将logits转换为概率
        #     class_probs = torch.sigmoid(edge_logits)
        #     # 生成注意力权重（正样本概率越高，注意力权重越大）
        #     attention_weights = self.attention_generator(class_probs)
        class_probs = torch.sigmoid(edge_logits)
        attention_weights = self.attention_generator(class_probs)
        
        # 5. 边回归任务（使用注意力权重加权）
        reg_values = self.edge_regressor(edge_repr)  # [num_edges, 1]
        weighted_reg_values = reg_values * attention_weights  # 应用注意力
        
        return {
            'classification': edge_logits,  # [num_edges, 1]
            'regression': weighted_reg_values,  # [num_edges, 1]
            'attention_weights': attention_weights  # [num_edges, 1]
        }

    def predict(self, data):
        """推理模式，返回分类概率和回归值"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
            probs = torch.sigmoid(outputs['classification'])
            return {
                'probabilities': probs,
                'regression_values': outputs['regression'],
                'attention_weights': outputs['attention_weights']
            }
        
class EdgeRankingGNN(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=3, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        优化版的 Edge Ranking GNN 模型
        """
        super(EdgeRankingGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 节点特征编码器 - 使用更高效的架构
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),  # inplace 节省内存
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # GCN卷积层 - 预分配所有层
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 全局池化后的处理
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边评分预测器 - 优化结构
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        """
        优化版前向传播 - 批量处理边特征
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. 编码节点特征
        x = self.node_encoder(x)
        
        # 2. 编码边特征
        edge_features = self.edge_encoder(edge_attr)
        
        # 3. 应用GCN层
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x, inplace=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 4. 全局池化
        global_features = global_mean_pool(x, batch)
        global_features = self.global_processor(global_features)
        
        # 5. 批量构建边特征 - 主要优化点
        edge_scores = self._batch_compute_edge_scores(x, edge_index, edge_features, global_features, batch)
        
        return edge_scores
    
    def _batch_compute_edge_scores(self, x, edge_index, edge_features, global_features, batch):
        """
        批量计算边分数 - 避免for循环
        """
        # 获取源节点和目标节点特征 [num_edges, hidden_dim]
        src_features = x[edge_index[0]]  # 源节点特征
        dst_features = x[edge_index[1]]  # 目标节点特征
        
        # 获取对应的全局特征 [num_edges, hidden_dim]
        batch_indices = batch[edge_index[0]]  # 源节点的batch索引
        global_feats = global_features[batch_indices]
        
        # 批量拼接所有特征 [num_edges, hidden_dim * 4]
        combined_features = torch.cat([
            src_features, 
            dst_features, 
            global_feats, 
            edge_features
        ], dim=-1)
        
        # 批量预测边分数
        edge_scores = self.edge_predictor(combined_features)
        
        return edge_scores
    
class EdgeRankingGNN1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64):
        super(EdgeRankingGNN1, self).__init__()
        
        # 节点特征编码器（添加输入Dropout）
        self.node_encoder = nn.Sequential(
            nn.Dropout(0.2),  # 输入层Dropout
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # 隐藏层Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边特征编码器（添加双重Dropout）
        self.edge_encoder = nn.Sequential(
            nn.Dropout(0.2),  # 输入层Dropout
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # 边特征通常更稀疏，使用更高Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 图卷积层（添加层间Dropout）
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv_dropout = nn.Dropout(0.5)  # 专门用于图卷积的Dropout
        
        # 边评分模块（深度Dropout）
        self.edge_scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.4),  # 第一层后Dropout
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.3),  # 第二层后Dropout
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 全局图特征提取（添加Dropout）
        self.global_pool = global_mean_pool
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # 上下文编码Dropout
            nn.LayerNorm(hidden_dim)
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.01)  # 避免dead neurons
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. 特征编码（保持Dropout）
        h_nodes = self.node_encoder(x)
        h_edges = self.edge_encoder(edge_attr)
        
        # 2. 图卷积（增强版Dropout）
        h_nodes = F.leaky_relu(self.conv1(h_nodes, edge_index), 0.1)
        h_nodes = self.conv_dropout(h_nodes)  # 专用图卷积Dropout
        h_nodes = self.conv2(h_nodes, edge_index)
        h_nodes = self.conv_dropout(h_nodes)  # 二次Dropout
        
        # 3. 获取全局图上下文（保持Dropout）
        global_context = self.global_pool(h_nodes, batch)
        global_context = self.context_encoder(global_context)
        
        # 4. 构建边表示（添加特征级Dropout）
        src, dst = edge_index
        edge_repr = torch.cat([
            F.dropout(h_nodes[src], p=0.2),  # 节点特征Dropout
            F.dropout(h_nodes[dst], p=0.2),
            F.dropout(h_edges, p=0.3),       # 边特征Dropout
            global_context[batch[src]]       # 全局特征不加Dropout
        ], dim=1)
        
        # 5. 计算边评分（保持模型内Dropout）
        edge_scores = self.edge_scorer(edge_repr)
        
        return edge_scores

class EdgeRankingGNN2(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=3, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        【完整修改版】Edge Ranking GNN
        使用 GINEConv 替代 GCNConv，强制模型在消息传递中融合边特征。
        """
        super(EdgeRankingGNN2, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 1. 节点特征编码器
        # 将输入节点特征映射到 hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. 边特征编码器
        # 【关键】GINEConv 要求边特征维度必须与节点特征维度一致，以便进行 sum 操作
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. GNN 卷积层 - 使用 GINEConv
        # 原理: x_i' = MLP( (1+eps)x_i + sum(ReLU(x_j + e_ij)) )
        # 这里的 e_ij 就是边特征，它被加到了邻居节点特征上，直接影响聚合结果
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            # GINEConv 需要一个内部的 MLP (多层感知机) 来处理聚合后的特征
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) # 加个 LayerNorm 训练更稳定
            )
            # train_eps=True 允许模型学习中心节点自身的权重
            self.gcn_layers.append(GINEConv(gin_mlp, train_eps=True))
        
        # 4. 全局池化后的处理
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 5. 边评分预测器
        # 输入维度: 源节点(h) + 目标节点(h) + 全局图(h) + 原始边特征(h) = 4h
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        """
        前向传播逻辑
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # -----------------------------------------------------------
        # Step 1: 特征编码 (Embedding)
        # -----------------------------------------------------------
        x = self.node_encoder(x)
        
        # 【关键】必须先将物理边特征编码为 hidden_dim
        # 这样才能在 GINEConv 中与节点特征相加
        edge_features = self.edge_encoder(edge_attr)
        
        # -----------------------------------------------------------
        # Step 2: 图卷积 (Message Passing)
        # -----------------------------------------------------------
        for i, gcn_layer in enumerate(self.gcn_layers):
            # 【关键修改】显式传入 edge_attr=edge_features
            # 此时，如果 edge_features 被打乱，x 的更新结果会剧烈变化
            x = gcn_layer(x, edge_index, edge_attr=edge_features)
            
            # 层间激活与 Dropout (GINE 内部已有 ReLU，这里是层间的非线性)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x, inplace=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # -----------------------------------------------------------
        # Step 3: 全局信息提取 (Readout)
        # -----------------------------------------------------------
        global_features = global_mean_pool(x, batch)
        global_features = self.global_processor(global_features)
        
        # -----------------------------------------------------------
        # Step 4: 边评分预测 (Link Prediction)
        # -----------------------------------------------------------
        # 注意：这里的 x 已经是包含了图结构和边信息的深层特征了
        edge_scores = self._batch_compute_edge_scores(x, edge_index, edge_features, global_features, batch)
        
        return edge_scores
    
    def _batch_compute_edge_scores(self, x, edge_index, edge_features, global_features, batch):
        """
        批量计算所有边的分数
        """
        # 1. 提取每条边的源节点和目标节点特征
        # x: [num_nodes, hidden_dim] -> src/dst: [num_edges, hidden_dim]
        src_features = x[edge_index[0]]
        dst_features = x[edge_index[1]]
        
        # 2. 提取每条边所属图的全局特征
        # batch: [num_nodes] -> batch_indices: [num_edges]
        # 注意：这里用 edge_index[0] (源节点) 的 batch 索引即可
        batch_indices = batch[edge_index[0]]
        global_feats = global_features[batch_indices]
        
        # 3. 拼接特征
        # 结合了：深层节点信息 + 上下文全局信息 + 原始边物理信息
        combined_features = torch.cat([
            src_features, 
            dst_features, 
            global_feats, 
            edge_features
        ], dim=-1)
        torch.concatenate
        # 4. 预测分数
        edge_scores = self.edge_predictor(combined_features)
        
        return edge_scores

class EdgeRankingGNN_Ablation_0109(nn.Module):
    def __init__(self, 
                 node_in_dim=2, 
                 edge_in_dim=8, 
                 hidden_dim=64, 
                 num_layers=2, 
                 dropout=0.2,
                 # --- 消融实验控制参数 ---
                 use_gine=True,           # 1. 算子选择: True=GINEConv, False=GCNConv
                 use_global=True,         # 2. 全局信息: True=使用全局池化, False=不使用
                 node_encoder_type='mlp', # 3. 节点编码器: 'mlp'=深层编码, 'linear'=简单线性投影
                 edge_encoder_type='mlp'  # 4. 边编码器: 'mlp'=深层编码, 'linear'=简单线性投影
                 ):
        """
        支持消融实验的 Edge Ranking GNN
        """
        super(EdgeRankingGNN_Ablation_0109, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_gine = use_gine
        self.use_global = use_global
        
        # =====================================================================
        # 1. 节点特征编码器 (验证 Vertex Encoder 必要性)
        # =====================================================================
        if node_encoder_type == 'mlp':
            self.node_encoder = nn.Sequential(
                nn.Linear(node_in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        else: # 'linear' (Baseline)
            self.node_encoder = nn.Sequential(
                nn.Linear(node_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) # 仅做维度对齐
            )
        
        # =====================================================================
        # 2. 边特征编码器 (验证 Edge Encoder 必要性)
        # =====================================================================
        if edge_encoder_type == 'mlp':
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        else: # 'linear' (Baseline)
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        # =====================================================================
        # 3. GNN 卷积层 (验证 GINE vs GCN)
        # =====================================================================
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.use_gine:
                # [Ours] GINE: 消息传递中融合边特征
                gin_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
                self.gcn_layers.append(GINEConv(gin_mlp, train_eps=True))
            else:
                # [Ablation] GCN: 仅利用拓扑结构，忽略边特征参与聚合
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # =====================================================================
        # 4. 全局池化后的处理
        # =====================================================================
        if self.use_global:
            self.global_processor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
        
        # =====================================================================
        # 5. 边评分预测器 (根据是否使用全局信息调整输入维度)
        # =====================================================================
        # 基础维度: 源节点(h) + 目标节点(h) + 原始边特征(h) = 3h
        # 如果有全局: + 全局图(h) = 4h
        input_dim = hidden_dim * 4 if self.use_global else hidden_dim * 3
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # -----------------------------------------------------------
        # Step 1: 特征编码 (受 encoder_type 控制)
        # -----------------------------------------------------------
        x = self.node_encoder(x)
        # 无论 GNN 是否使用边特征，最后拼接都需要边特征，所以这里必须编码
        edge_features = self.edge_encoder(edge_attr)
        
        # -----------------------------------------------------------
        # Step 2: 图卷积 (受 use_gine 控制)
        # -----------------------------------------------------------
        for i, gcn_layer in enumerate(self.gcn_layers):
            if self.use_gine:
                # GINE: 显式传入 edge_attr
                x = gcn_layer(x, edge_index, edge_attr=edge_features)
            else:
                # GCN: 不传入 edge_attr，只利用拓扑结构聚合
                x = gcn_layer(x, edge_index)
            
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x, inplace=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # -----------------------------------------------------------
        # Step 3: 全局信息提取 (受 use_global 控制)
        # -----------------------------------------------------------
        global_features = None
        if self.use_global:
            # global_mean_pool 将 [num_nodes, h] -> [batch_size, h]
            global_graph = global_mean_pool(x, batch)
            global_features = self.global_processor(global_graph)
        
        # -----------------------------------------------------------
        # Step 4: 边评分预测
        # -----------------------------------------------------------
        edge_scores = self._batch_compute_edge_scores(
            x, edge_index, edge_features, global_features, batch
        )
        
        return edge_scores
    
    def _batch_compute_edge_scores(self, x, edge_index, edge_features, global_features, batch):
        # 1. 提取源/目标节点特征
        src_features = x[edge_index[0]]
        dst_features = x[edge_index[1]]
        
        concat_list = [src_features, dst_features]
        
        # 2. 如果启用了全局信息，提取并加入
        if global_features is not None:
            batch_indices = batch[edge_index[0]] # 获取每条边属于哪个图
            global_feats_expanded = global_features[batch_indices]
            concat_list.append(global_feats_expanded)
            
        # 3. 加入边特征 (Local Edge Context)
        concat_list.append(edge_features)
        
        # 4. 拼接
        combined_features = torch.cat(concat_list, dim=-1)
        
        # 5. 预测
        edge_scores = self.edge_predictor(combined_features)
        
        return edge_scores

class EdgeRankingGNN2_Ablation(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=3, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        【完整修改版】Edge Ranking GNN
        使用 GINEConv 替代 GCNConv，强制模型在消息传递中融合边特征。
        """
        super(EdgeRankingGNN2_Ablation, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # ------------------------------------------------------------------
        # 1. 特征编码器 (Encoders)
        # ------------------------------------------------------------------
        
        # 节点特征编码器: [node_in] -> [hidden]
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边特征编码器: [edge_in] -> [hidden]
        # 注意: GINEConv 要求边特征维度必须与节点特征维度一致
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # ------------------------------------------------------------------
        # 2. 图卷积层 (GINEConv - Deep Fusion)
        # ------------------------------------------------------------------
        # 原理: x_i' = MLP( (1+eps)x_i + sum(ReLU(x_j + e_ij)) )
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            # GINEConv 内部需要的 MLP
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) # LayerNorm 有助于深层 GNN 训练
            )
            # train_eps=True 允许模型自适应学习中心节点的权重
            self.gcn_layers.append(GINEConv(gin_mlp, train_eps=True))
        
        # ------------------------------------------------------------------
        # 3. 后处理与预测头
        # ------------------------------------------------------------------
        
        # 全局特征处理
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边评分预测器 (Concat: Src + Dst + Global + Edge)
        # 输入维度 = 4 * hidden_dim
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Tanh(), # 使用 Tanh 增加非线性表达
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # 输出 [0, 1] 概率
        )
        
    def forward(self, data, ablation=None):
        """
        前向传播
        Args:
            data: PyG Data 对象
            ablation (str, optional): 消融实验模式. 
                - None: 正常模式 (Model A)
                - 'no_edge': 将边特征置零 (Model B)
                - 'no_node': 将点特征置零 (Model C)
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # ==========================
        # 1. 消融实验逻辑 (Masking)
        # ==========================
        if ablation == 'no_edge':
            # 将物理边特征全置为 0，模拟无边特征输入
            edge_attr = torch.zeros_like(edge_attr)
        elif ablation == 'no_node':
            # 将物理点特征全置为 0，模拟无点特征输入
            x = torch.zeros_like(x)
            
        # ==========================
        # 2. 特征编码 (Embedding)
        # ==========================
        x_emb = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)
        
        # ==========================
        # 3. 图卷积 (Message Passing)
        # ==========================
        # x_curr 记录当前层的节点特征
        x_curr = x_emb
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            # 【核心修改】：显式传入 edge_attr=edge_emb
            # 此时 edge_emb 参与了邻居聚合：sum(ReLU(x_j + e_ij))
            x_curr = gcn_layer(x_curr, edge_index, edge_attr=edge_emb)
            
            # 层间激活与 Dropout (最后一层通常也保留，防止过拟合)
            if i < len(self.gcn_layers) - 1:
                x_curr = F.relu(x_curr, inplace=True)
                x_curr = F.dropout(x_curr, p=self.dropout, training=self.training)
        
        # ==========================
        # 4. 全局池化 (Readout)
        # ==========================
        # 使用卷积后的节点特征进行池化
        global_features = global_mean_pool(x_curr, batch)
        global_features = self.global_processor(global_features)
        
        # ==========================
        # 5. 边评分预测 (Link Prediction)
        # ==========================
        edge_scores = self._batch_compute_edge_scores(x_curr, edge_index, edge_emb, global_features, batch)
        
        return edge_scores
    
    def _batch_compute_edge_scores(self, x, edge_index, edge_features, global_features, batch):
        """
        批量计算边分数
        """
        # 1. 提取源节点和目标节点特征
        src_features = x[edge_index[0]]
        dst_features = x[edge_index[1]]
        
        # 2. 提取对应的全局特征
        batch_indices = batch[edge_index[0]]
        global_feats = global_features[batch_indices]
        
        # 3. 特征拼接 (Fusion)
        # 融合了: 深层节点信息 + 全局上下文 + 原始边信息(编码后)
        combined_features = torch.cat([
            src_features, 
            dst_features, 
            global_feats, 
            edge_features
        ], dim=-1)
        
        # 4. 预测
        edge_scores = self.edge_predictor(combined_features)
        
        return edge_scores

class EdgeRankingGNN2_Ablation1(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=3, hidden_dim=64, num_layers=2, dropout=0.2, ablation_mode='none'):
        """
        支持消融实验的 Edge Ranking GNN
        
        Args:
            ablation_mode (str): 消融模式选择
                - 'none':       完整模型 (Full Model: GINE + Global Context)
                - 'no_gine':    移除边条件卷积 (Replace GINE with GCN, edge features ignored in MP)
                - 'no_global':  移除全局上下文 (Remove Global Context Branch)
                - 'no_mp':      移除图消息传递 (Remove GNN Layers, acts as MLP Baseline)
        """
        super(EdgeRankingGNN2_Ablation1, self).__init__()
        
        self.ablation_mode = ablation_mode
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # -----------------------------------------------------------
        # 1. 基础特征编码器 (所有变体通用)
        # -----------------------------------------------------------
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # -----------------------------------------------------------
        # 2. GNN 层 (根据 ablation_mode 动态构建)
        # -----------------------------------------------------------
        self.gcn_layers = nn.ModuleList()
        
        if self.ablation_mode != 'no_mp':  # 如果不是 MLP Baseline，则构建 GNN 层
            for _ in range(num_layers):
                if self.ablation_mode == 'no_gine':
                    # 【消融变体 A】w/o GINE: 使用普通 GCN，无法融合边特征
                    self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
                else:
                    # 【完整模型 / w/o Global】使用 GINE，融合边特征
                    gin_mlp = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                    self.gcn_layers.append(GINEConv(gin_mlp, train_eps=True))
        
        # -----------------------------------------------------------
        # 3. 全局处理器 (根据 ablation_mode 动态构建)
        # -----------------------------------------------------------
        if self.ablation_mode != 'no_global':
            self.global_processor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
        
        # -----------------------------------------------------------
        # 4. 预测头输入维度计算
        # -----------------------------------------------------------
        # 基础维度: Source节点(h) + Target节点(h) + 边特征(h) = 3h
        input_dim = hidden_dim * 3
        
        # 如果包含全局上下文，则 + Global(h) = 4h
        if self.ablation_mode != 'no_global':
            input_dim += hidden_dim
            
        self.edge_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Step 1: 编码
        x = self.node_encoder(x)
        edge_features = self.edge_encoder(edge_attr) # 边特征始终需要编码，用于最后拼接
        
        # Step 2: 消息传递 (根据模式选择)
        if self.ablation_mode != 'no_mp':
            for i, gcn_layer in enumerate(self.gcn_layers):
                if self.ablation_mode == 'no_gine':
                    # w/o GINE: GCNConv 只接受节点特征和邻接关系，忽略 edge_attr
                    x = gcn_layer(x, edge_index)
                else:
                    # Full / w/o Global: GINEConv 显式利用 edge_attr
                    x = gcn_layer(x, edge_index, edge_attr=edge_features)
                
                # 激活与正则化
                if i < len(self.gcn_layers) - 1:
                    x = F.relu(x, inplace=True)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Step 3: 全局池化 (根据模式选择)
        global_features = None
        if self.ablation_mode != 'no_global':
            if batch is None:
                 batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            global_pool = global_mean_pool(x, batch)
            global_features = self.global_processor(global_pool)
        
        # Step 4: 预测
        return self._batch_compute_edge_scores(x, edge_index, edge_features, global_features, batch)
    
    def _batch_compute_edge_scores(self, x, edge_index, edge_features, global_features, batch):
        # 提取源节点和目标节点特征
        src_features = x[edge_index[0]]
        dst_features = x[edge_index[1]]
        
        # 动态构建特征列表
        features_list = [src_features, dst_features]
        
        # 如果启用了全局上下文，则加入
        if global_features is not None:
            if batch is None:
                 batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            batch_indices = batch[edge_index[0]]
            global_feats_expanded = global_features[batch_indices]
            features_list.append(global_feats_expanded)
        
        # 始终加入边特征 (即使是 w/o GINE，边特征在最后分类时也是必须的)
        features_list.append(edge_features)
        
        # 拼接
        combined_features = torch.cat(features_list, dim=-1)
        
        # 预测
        return self.edge_predictor(combined_features)

class EnhancedEdgeRankingGNN(nn.Module):
    """增强版本的Edge Ranking GNN，包含更复杂的特征处理"""
    
    def __init__(self, node_in_dim=3, edge_in_dim=3, hidden_dim=128, dropout=0.3):
        super(EnhancedEdgeRankingGNN, self).__init__()
        
        self.dropout = dropout
        
        # 更深的节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 更深的边编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 双GCN层
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # 全局特征处理器
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边评分预测器（更复杂）
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 拼接4个特征
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 编码特征
        x = self.node_encoder(x)
        edge_features = self.edge_encoder(edge_attr)
        
        # GCN层
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        
        # 全局特征
        global_features = global_mean_pool(x, batch)
        global_features = self.global_processor(global_features)
        
        # 边评分
        edge_scores = []
        for edge_idx in range(edge_index.size(1)):
            src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]
            
            src_feat = x[src]
            dst_feat = x[dst]
            batch_idx = batch[src]
            global_feat = global_features[batch_idx]
            
            # 拼接4个特征
            combined_feat = torch.cat([
                src_feat, dst_feat, global_feat, edge_features[edge_idx]
            ], dim=-1)
            
            edge_score = self.edge_predictor(combined_feat)
            edge_scores.append(edge_score)
        
        return torch.cat(edge_scores, dim=0)

# 特征计算函数
def compute_mesh_features(vertices, faces, edges):
    """
    计算网格特征
    
    参数:
    vertices: 顶点坐标 [N, 3]
    faces: 面索引 [F, 3]
    edges: 边索引 [E, 2]
    
    返回:
    node_features: 节点特征 [N, 3] (二面角、边长比、法向量角度)
    edge_features: 边特征 [E, 3]
    """
    import numpy as np
    import torch
    
    # 这里需要实现具体的特征计算逻辑
    # 由于实现较复杂，这里提供框架
    
    # 计算顶点法向量
    def compute_vertex_normals(vertices, faces):
        # 实现顶点法向量计算
        pass
    
    # 计算二面角
    def compute_dihedral_angles(vertices, faces, edges):
        # 实现二面角计算
        pass
    
    # 计算边长比例特征
    def compute_edge_length_ratios(vertices, edges):
        # 实现边长比例计算
        pass
    
    # 实际应用中需要填充这些函数的具体实现
    
    # 返回示例特征（实际使用时应替换为真实计算）
    num_nodes = vertices.shape[0]
    num_edges = edges.shape[0]
    
    node_features = torch.randn(num_nodes, 3)
    edge_features = torch.randn(num_edges, 3)
    
    return node_features, edge_features

# 使用示例
def create_sample_data():
    """创建示例数据"""
    num_nodes = 10
    num_edges = 15
    
    # 随机节点特征
    node_features = torch.randn(num_nodes, 3)
    
    # 随机边索引
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 随机边特征
    edge_attr = torch.randn(num_edges, 3)
    
    # 批处理索引（假设所有节点属于同一个图）
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    return data

# 测试
if __name__ == "__main__":
    # 创建模型
    model = EdgeRankingGNN(node_in_dim=3, edge_in_dim=3, hidden_dim=64)
    
    # 创建示例数据
    data = create_sample_data()
    
    # 前向传播
    edge_scores = model(data)
    print(f"Edge scores shape: {edge_scores.shape}")
    print(f"Edge scores: {edge_scores[:5]}")