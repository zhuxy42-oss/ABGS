import meshio
import time
import numpy as np
from data.classification_data import ClassificationData
from options.base_options import BaseOptions
from options.train_options import TrainOptions
# from data import DataLoader
from torch_geometric.loader import DataLoader
from models import create_model
from util.writer import Writer
from util.util import make_dataset
from util.nlp_smooth import smooth_sizing_function
from test import run_test
import torch
import torch.nn as nn
from models.networks import SimplificationLoss, EdgeCrossEntropyLoss, ListNetLoss, SpearmanLoss, M2MRegressionLoss, EdgeRankingGNN,EdgeRankingGNN1, EdgeRankingGNN2, EdgeClassificationGNN1, EdgeRankingGNN2_Ablation, EdgeRankingGNN_Ablation_0109
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from models.layers.mesh import Mesh
from torch_geometric.data import Data
from data.simplification_data import gather_graph, gather_graph_normal, gather_graph_normal1
import os, pickle
import matplotlib.pyplot as plt
import pandas as pd
from models.layers.CustomMesh import CustomMesh
from testing_gcn import test_traditional_classify, test_traditional_method

def plot_convergence_curve(epochs, train_losses, val_losses, learning_rates):
    """绘制训练收敛曲线图"""
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制loss曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Validation Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化
    
    plt.tight_layout()
    
    # 保存图片
    save_path = "/home/zhuxunyang/coding/simply/checkpoints/debug/convergence_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {save_path}")
    
    # 显示图片（如果在有图形界面的环境中）
    # plt.show()
    
    # 同时保存loss数据到CSV文件
    save_loss_data(epochs, train_losses, val_losses, learning_rates)

def save_loss_data(epochs, train_losses, val_losses, learning_rates):
    """保存loss数据到CSV文件"""
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Learning_Rate': learning_rates
    })
    
    csv_path = "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/loss_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Loss数据已保存至: {csv_path}")
    
    # 打印最终训练结果摘要
    print("\n=== 训练结果摘要 ===")
    print(f"总训练轮数: {len(epochs)}")
    print(f"最佳训练loss: {min(train_losses):.8f} (Epoch {epochs[train_losses.index(min(train_losses))]})")
    print(f"最佳验证loss: {min(val_losses):.8f} (Epoch {epochs[val_losses.index(min(val_losses))]})")
    print(f"最终学习率: {learning_rates[-1]:.8f}")

def plot_realtime_convergence(epochs, train_losses, val_losses, learning_rates, current_epoch):
    """实时更新收敛曲线（可选功能）"""
    if current_epoch % 10 == 0:  # 每10个epoch更新一次
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, learning_rates, 'g-', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"/home/zhuxunyang/coding/simply/checkpoints/debug/convergence_epoch_{current_epoch}.png", dpi=300)
        plt.close()

def load_para():
    node_file = '/home/zhuxunyang/coding/simply/para/node_stats.pkl'
    edge_file = '/home/zhuxunyang/coding/simply/para/edge_stats.pkl'

    # 尝试加载已有参数
    if os.path.exists(node_file) and os.path.exists(edge_file):
        with open(node_file, 'rb') as f:
            node_mean, node_std = pickle.load(f)
        with open(edge_file, 'rb') as f:
            edge_mean, edge_std = pickle.load(f)
        return node_mean, node_std, edge_mean, edge_std

def train_regression():
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630", opt, device, 20, load_para())
    val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/sf0613/val", opt, device, 1, load_para())

    model = EdgeRankingGNN(
        node_feat_dim=2,
        edge_feat_dim=8,
        hidden_dim=64
    ).to(device=device)
    total_steps = 0

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)    #, weight_decay=0.0001

    # scheduler = MultiStepLR(optimizer, milestones=[opt.niter, opt.niter_decay], gamma=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    
    max_loss = 100000

    print(model)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_sum_regression = 0
        loss_sum_rank = 0
        val_loss_sum_regression = 0
        val_loss_sum_rank = 0
        for batch in dataset:
            label_regression = batch.y
            out = model(batch)
            loss1 = criterion(out, label_regression)
            # loss2 = rank(out, label_regression)
            loss_sum_regression += loss1
            # loss_sum_rank += loss2
            loss = loss1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()

        for val_batch in val_dataset:
            with torch.no_grad():
                val_label_regression = val_batch.y
                val_out = model(val_batch)
                val_loss1 = criterion(val_out, val_label_regression)
                # val_loss2 = rank(val_out, val_label_regression)
                val_loss_sum_regression += val_loss1
                # val_loss_sum_rank += val_loss2
                val_loss = val_loss1

        if (loss_sum_regression.item() < max_loss):
            max_loss = loss_sum_regression.item()
            torch.save(model.state_dict(), "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression_dis0Newsize1.pth")

        # print(f"Epoch {epoch}  MSE: {loss_sum_regression.item():.8f} Rank: {loss_sum_rank.item():.8f}  Val MSE: {val_loss_sum_regression.item():.8f} Val Rank: {val_loss_sum_rank.item():.8f} Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Epoch {epoch}  MSE: {loss_sum_regression.item():.8f} Val MSE: {val_loss_sum_regression.item():.8f} Learning rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.00001:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001
    
def train_rank():
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, _ = gather_graph_normal("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630", opt, device, 20, load_para())
    val_dataset, _ = gather_graph_normal("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/sf0613/val", opt, device, 1, load_para())

    model = EdgeRankingGNN(
        node_feat_dim=1,
        edge_feat_dim=5,
        hidden_dim=64
    ).to(device=device)
    total_steps = 0


    criterion = nn.MSELoss()
    Classify = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)    #, weight_decay=0.0001

    # scheduler = MultiStepLR(optimizer, milestones=[opt.niter], gamma=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.999)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=100)
    
    max_loss = 100000

    print(model)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_sum_regression = 0
        loss_sum_rank = 0
        val_loss_sum_regression = 0
        val_loss_sum_rank = 0
        loss_sum_classify = 0
        val_loss_sum_classify = 0
        for batch in dataset:
            label_regression = batch.y
            label_classify = batch.y1
            edge_lens = batch.length
            out = model(batch)
            loss1 = criterion(out, label_regression)
            # loss2 = rank(out, label_regression, edge_lens)
            loss3 = Classify(out, label_classify)
            loss_sum_regression += loss1
            # loss_sum_rank += loss2
            loss_sum_classify += loss3
            loss = 1 * loss1 + 0 * loss3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()

        for val_batch in val_dataset:
            with torch.no_grad():
                val_label_regression = val_batch.y
                val_label_classify = val_batch.y1
                val_edge_lens = val_batch.length
                val_out = model(val_batch)
                val_loss1 = criterion(val_out, val_label_regression)
                # val_loss2 = rank(val_out, val_label_regression, val_edge_lens)
                val_loss3 = Classify(val_out, val_label_classify)
                val_loss_sum_regression += val_loss1
                # val_loss_sum_rank += val_loss2
                val_loss_sum_classify += val_loss3
                # val_loss = val_loss2

        if (loss_sum_rank.item() < max_loss):
            max_loss = loss_sum_rank.item()
            torch.save(model.state_dict(), "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression3_1.pth")

        print(f"Epoch {epoch}  MSE: {loss_sum_regression.item():.8f} Rank: {loss_sum_rank.item():.8f} BCE: {loss_sum_classify.item():.8f}  Val MSE: {val_loss_sum_regression.item():.8f} Val Rank: {val_loss_sum_rank.item():.8f} Val Classify: {val_loss_sum_classify.item():.8f} Learning rate: {optimizer.param_groups[0]['lr']}")
        # print(f"Epoch {epoch}  MSE: {loss_sum_regression.item():.8f} Val MSE: {val_loss_sum_regression.item():.8f} Learning rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step(loss_sum_regression)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.00001:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

def train_classify():
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630", opt, device, 20, load_para())
    val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/sf0613/val", opt, device, 1, load_para())

    model = EdgeClassificationGNN1(
        node_feat_dim=2,
        edge_feat_dim=8,
        hidden_dim=64
    ).to(device=device)
    # model = EdgeRankingGNN(
    #     node_feat_dim=2,
    #     edge_feat_dim=8,
    #     hidden_dim=64
    # ).to(device=device)

    total_steps = 0

    Classify = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0], device=device))

    optimizer = optim.Adam(model.parameters(), lr=0.001)    #, weight_decay=0.0001

    # scheduler = MultiStepLR(optimizer, milestones=[opt.niter], gamma=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    
    max_loss = 100000

    print(model)

    train_losses = []
    val_losses = []
    epochs_list = []
    learning_rates = []

    batch_count = 0
    val_batch_count = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_sum_classify = 0
        loss_sum_rank = 0
        val_loss_sum_classify = 0
        val_loss_sum_rank = 0
        for batch in dataset:
            label_classify = batch.y1
            out = model(batch)
            loss1 = Classify(out, label_classify)
            loss_sum_classify += loss1
            loss = loss1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()
            batch_count += 1

        # 计算平均训练loss
        avg_train_loss = loss_sum_classify / batch_count if batch_count > 0 else 0

        for val_batch in val_dataset:
            with torch.no_grad():
                val_label_classify = val_batch.y1
                val_out = model(val_batch)
                val_loss1 = Classify(val_out, val_label_classify)
                val_loss_sum_classify += loss1
                val_loss = val_loss1
                val_batch_count += 1


        avg_val_loss = val_label_classify / val_batch_count if val_batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        if (loss_sum_classify.item() < max_loss):
            max_loss = loss_sum_classify.item()
            torch.save(model.state_dict(), "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Classify1_dis1Newsize1.pth")

        print(f"Epoch {epoch} BCE: {loss_sum_classify.item():.8f} Val BCE: {val_loss_sum_classify.item():.8f} Learning rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.00001:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001

    epochs_list_cpu = [epoch.cpu().item() if hasattr(epoch, 'cpu') else epoch for epoch in epochs_list]
    train_losses_cpu = [loss.cpu().item() if hasattr(loss, 'cpu') else loss for loss in train_losses]
    val_losses_cpu = [loss.cpu().item() if hasattr(loss, 'cpu') else loss for loss in val_losses]
    learning_rates_cpu = [lr.cpu().item() if hasattr(lr, 'cpu') else lr for lr in learning_rates]
    
    plot_convergence_curve(epochs_list_cpu, train_losses_cpu, val_losses_cpu, learning_rates_cpu)


def train_regression1(mode):
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/Bkgm0630", opt, device, 20, load_para())       #"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630"
    val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/sf0613/val", opt, device, 1, load_para())

    # dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/Bkgm0630", opt, device, 20)       #"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630"
    # val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/sf0613/val", opt, device, 1)

    # model = EdgeRankingGNN1(
    #     node_feat_dim=2,
    #     edge_feat_dim=8,
    #     hidden_dim=64
    # ).to(device=device)
    # model = EdgeRankingGNN(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    # model = EdgeRankingGNN2(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    if mode == "nogine":
        model = EdgeRankingGNN_Ablation_0109(use_gine=False).to(device=device)
    elif mode == "noglobal":
        model = EdgeRankingGNN_Ablation_0109(use_global=False).to(device=device)
    elif mode == "noencoder_v":
        model = EdgeRankingGNN_Ablation_0109(node_encoder_type='linear').to(device=device)
    elif mode == "noencoder_e":
        model = EdgeRankingGNN_Ablation_0109(edge_encoder_type='linear').to(device=device)
    else:
        model = EdgeRankingGNN2(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    total_steps = 0

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    
    max_loss = 100000

    print(model)

    # 创建记录loss的列表
    train_losses = []
    val_losses = []
    epochs_list = []
    learning_rates = []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_sum_regression = 0
        loss_sum_rank = 0
        val_loss_sum_regression = 0
        val_loss_sum_rank = 0
        
        # 训练阶段
        model.train()
        batch_count = 0
        for batch in dataset:
            label_regression = batch.y
            out = model(batch)
            loss1 = criterion(out, label_regression)
            loss_sum_regression += loss1.item()
            loss = loss1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()
            batch_count += 1
        
        # 计算平均训练loss
        avg_train_loss = loss_sum_regression / batch_count if batch_count > 0 else 0
        
        # 验证阶段
        model.eval()
        val_batch_count = 0
        for val_batch in val_dataset:
            with torch.no_grad():
                val_label_regression = val_batch.y
                val_out = model(val_batch)
                val_loss1 = criterion(val_out, val_label_regression)
                val_loss_sum_regression += val_loss1.item()
                val_batch_count += 1
        
        # 计算平均验证loss
        avg_val_loss = val_loss_sum_regression / val_batch_count if val_batch_count > 0 else 0
        
        # 记录loss和epoch信息
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 保存最佳模型
        if avg_val_loss < max_loss:
            max_loss = avg_val_loss
            if mode is not None:
                torch.save(model.state_dict(), f"/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_{mode}.pth")
            else:
                torch.save(model.state_dict(), "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth")
            print(f"保存最佳模型，验证loss: {avg_train_loss:.8f}")

        # 打印训练信息
        print(f"Epoch {epoch}/{opt.niter + opt.niter_decay} | "
              f"Train MSE: {avg_train_loss:.8f} | "
              f"Val MSE: {avg_val_loss:.8f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {time.time() - epoch_start_time:.2f}s")

        scheduler.step()

        # 学习率下限
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.00001:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

    # 训练完成后绘制收敛曲线
    # plot_convergence_curve(epochs_list, train_losses, val_losses, learning_rates)

def train_regression_ablation1():
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/Bkgm0630", opt, device, 20, load_para())       #"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630"
    val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/sf0613/val", opt, device, 1, load_para())

    # dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/Bkgm0630", opt, device, 20)       #"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630"
    # val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/sf0613/val", opt, device, 1)

    # model = EdgeRankingGNN1(
    #     node_feat_dim=2,
    #     edge_feat_dim=8,
    #     hidden_dim=64
    # ).to(device=device)
    # model = EdgeRankingGNN(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    # model = EdgeRankingGNN2(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    model = EdgeRankingGNN2_Ablation(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    total_steps = 0

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    
    max_loss = 100000

    print(model)

    # 创建记录loss的列表
    train_losses = []
    val_losses = []
    epochs_list = []
    learning_rates = []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_sum_regression = 0
        loss_sum_rank = 0
        val_loss_sum_regression = 0
        val_loss_sum_rank = 0
        
        # 训练阶段
        model.train()
        batch_count = 0
        for batch in dataset:
            label_regression = batch.y
            out = model(batch, ablation = 'no_edge')
            loss1 = criterion(out, label_regression)
            loss_sum_regression += loss1.item()
            loss = loss1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()
            batch_count += 1
        
        # 计算平均训练loss
        avg_train_loss = loss_sum_regression / batch_count if batch_count > 0 else 0
        
        # 验证阶段
        model.eval()
        val_batch_count = 0
        for val_batch in val_dataset:
            with torch.no_grad():
                val_label_regression = val_batch.y
                val_out = model(val_batch, ablation = 'no_edge')
                val_loss1 = criterion(val_out, val_label_regression)
                val_loss_sum_regression += val_loss1.item()
                val_batch_count += 1
        
        # 计算平均验证loss
        avg_val_loss = val_loss_sum_regression / val_batch_count if val_batch_count > 0 else 0
        
        # 记录loss和epoch信息
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 保存最佳模型
        if avg_val_loss < max_loss:
            max_loss = avg_val_loss
            torch.save(model.state_dict(), "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_noedge_dis1Newsize5_1228.pth")
            print(f"保存最佳模型，验证loss: {avg_train_loss:.8f}")

        # 打印训练信息
        print(f"Epoch {epoch}/{opt.niter + opt.niter_decay} | "
              f"Train MSE: {avg_train_loss:.8f} | "
              f"Val MSE: {avg_val_loss:.8f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {time.time() - epoch_start_time:.2f}s")

        scheduler.step()

        # 学习率下限
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.00001:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

def train_regression_ablation2():
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/Bkgm0630", opt, device, 20, load_para())       #"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630"
    val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/sf0613/val", opt, device, 1, load_para())

    # dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/Bkgm0630", opt, device, 20)       #"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Bkgm0630"
    # val_dataset, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/sf0613/val", opt, device, 1)

    # model = EdgeRankingGNN1(
    #     node_feat_dim=2,
    #     edge_feat_dim=8,
    #     hidden_dim=64
    # ).to(device=device)
    # model = EdgeRankingGNN(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    # model = EdgeRankingGNN2(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    model = EdgeRankingGNN2_Ablation(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
    total_steps = 0

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    
    max_loss = 100000

    print(model)

    # 创建记录loss的列表
    train_losses = []
    val_losses = []
    epochs_list = []
    learning_rates = []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_sum_regression = 0
        loss_sum_rank = 0
        val_loss_sum_regression = 0
        val_loss_sum_rank = 0
        
        # 训练阶段
        model.train()
        batch_count = 0
        for batch in dataset:
            label_regression = batch.y
            out = model(batch, ablation = 'no_node')
            loss1 = criterion(out, label_regression)
            loss_sum_regression += loss1.item()
            loss = loss1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()
            batch_count += 1
        
        # 计算平均训练loss
        avg_train_loss = loss_sum_regression / batch_count if batch_count > 0 else 0
        
        # 验证阶段
        model.eval()
        val_batch_count = 0
        for val_batch in val_dataset:
            with torch.no_grad():
                val_label_regression = val_batch.y
                val_out = model(val_batch, ablation = 'no_node')
                val_loss1 = criterion(val_out, val_label_regression)
                val_loss_sum_regression += val_loss1.item()
                val_batch_count += 1
        
        # 计算平均验证loss
        avg_val_loss = val_loss_sum_regression / val_batch_count if val_batch_count > 0 else 0
        
        # 记录loss和epoch信息
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 保存最佳模型
        if avg_val_loss < max_loss:
            max_loss = avg_val_loss
            torch.save(model.state_dict(), "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_nonode_dis1Newsize5_1228.pth")
            print(f"保存最佳模型，验证loss: {avg_train_loss:.8f}")

        # 打印训练信息
        print(f"Epoch {epoch}/{opt.niter + opt.niter_decay} | "
              f"Train MSE: {avg_train_loss:.8f} | "
              f"Val MSE: {avg_val_loss:.8f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {time.time() - epoch_start_time:.2f}s")

        scheduler.step()

        # 学习率下限
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 0.00001:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

if __name__ == "__main__":
    train_regression1(mode="nogine")
    train_regression1(mode="noglobal")
    train_regression1(mode="noencoder_v")
    train_regression1(mode="noencoder_e")
    # train_regression1()
    # train_regression_ablation1()
    # train_regression_ablation2()
    # train_classify()
    # try:
    #     train_regression1()
    #     # train_classify()
    # except:
    #     print("End")

    # try:
    #     test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", 1.475, 10, 10, "/home/zhuxunyang/coding/simply/result", '1e-3')
    # except:
    #     print("end")
    # try:
    #     test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", 2.0, 200, 20, "/home/zhuxunyang/coding/simply/result1", '1e-4')
    # except:
    #     print("end")
    # try:
    #     test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", 1.5, 20, 20, "/home/zhuxunyang/coding/simply/result2", '1e-4')
    # except:
    #     print("end")
    # try:
    #     test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", 2.5, 0.5, 10, "/home/zhuxunyang/coding/simply/result3", '5e-3')
    # except:
    #     print("end")
    # try:
    #     test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", 2.5, 0.5, 10, "/home/zhuxunyang/coding/simply/result4", '5e-3')
    # except:
    #     print("end")
    # try:
    #     test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/232_0.vtk", 1.8, 0.5, 10, "/home/zhuxunyang/coding/simply/result5", '5e-3')
    # except:
    #     print("end")

    # mesh4 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/268_circuit_1/268_1.vtk")
    # mesh5 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/256_circuit_2/256_0.vtk")
    # start_time4 = time.time()
    # try:
    #     mesh4.sizing_values = torch.tensor(smooth_sizing_function(mesh4)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time4}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time4}s")
    # start_time5 = time.time()
    # try:
    #     mesh5.sizing_values = torch.tensor(smooth_sizing_function(mesh5, beta=1.5, tol=1e-4)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time5}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time5}s")

    # test_traditional_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/221_0.vtk", 1.5, 10, 20, "/home/zhuxunyang/coding/bkgm_simplification/result")
    # test_traditional_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/268_1.vtk", 2.5, 0.5, 10, "/home/zhuxunyang/coding/bkgm_simplification/result1")
    # test_traditional_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/256_0.vtk", 2.5, 0.5, 10, "/home/zhuxunyang/coding/bkgm_simplification/result")

    # start_time5 = time.time()
    # try:
    #     mesh5.sizing_values = torch.tensor(smooth_sizing_function(mesh5, beta=1.3)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time5}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time5}s")

    