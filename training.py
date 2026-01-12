import meshio
import time
import numpy as np
from data.classification_data import ClassificationData
from options.base_options import BaseOptions
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test
import torch
import torch.nn as nn
from models.networks import M2MRankLoss, M2MRegressionLoss, EdgeRankLoss, M2MRegressionLoss1, M2MClassifyLoss, BinaryClassAccuracyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt

def banding_accuracy(pred, label):
    batch = len(pred)
    total_correct = 0
    total_ones_match = 0
    total_zeros_match = 0
    total_ones = 0
    total_zeros = 0
    total_elements = 0
    
    for i in range(batch):
        predict = pred[i:i+1]
        target = label[i]
        
        edge_len = target.size(1)
        # Convert predictions to binary (0 or 1)
        predict_binary = (predict[:, 0:edge_len] > 0.5).float()
        
        # Calculate correct predictions
        correct = (predict_binary == target).float()
        total_correct += correct.sum().item()
        
        # Calculate where both are 1
        ones_match = (predict_binary * target).sum().item()
        total_ones_match += ones_match
        total_ones += target.sum().item()
        
        # Calculate where both are 0
        zeros_match = ((1 - predict_binary) * (1 - target)).sum().item()
        total_zeros_match += zeros_match
        total_zeros += (1 - target).sum().item()

        total_elements += edge_len
    
    # Calculate probabilities
    accuracy = total_correct / total_elements
    
    # Avoid division by zero
    ones_prob = total_ones_match / total_ones if total_ones > 0 else 0
    zeros_prob = total_zeros_match / total_zeros if total_zeros > 0 else 0

    return accuracy, ones_prob, zeros_prob
    
    # return {
    #     'accuracy': accuracy,
    #     'ones_accuracy': ones_prob,
    #     'zeros_accuracy': zeros_prob
    # }


def main():
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)

    dataset_size = len(dataset)

    print("Size:", dataset_size)

    model = create_model(opt)
    # print(model.net)
    total_steps = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = M2MRegressionLoss()
    weight_criterion = M2MRegressionLoss1()
    MSE = nn.MSELoss()
    Classify_inner = M2MClassifyLoss()
    Classify = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))    #pos_weight=torch.tensor([4.0], device=device)
    # Classify = nn.BCELoss()
    zero_one_accu = BinaryClassAccuracyLoss()

    optimizer = optim.Adam(model.net.parameters(), lr=0.001)    #, weight_decay=0.0001

    scheduler = MultiStepLR(optimizer, milestones=[opt.niter, opt.niter_decay], gamma=0.1)

    threshold = 0.5
    # threshold = torch.tensor(1.4, dtype=torch.float32)
    # print("before training loss.grad:", model.loss)

    # for param in model.net.parameters():
    #     print(param.requires_grad)

    train_losses1 = []
    train_losses2 = []
    classify_loss1 = []
    classify_loss2 = []
    val_train_losses1 = []
    val_train_losses2 = []
    val_classify_loss1 = []
    val_classify_loss2 = []
    epochs = []

    print(model.net)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        loss_regression = 0
        loss_regression_inner = 0
        loss_classify = 0
        loss_classify_inner = 0
        loss_one_zero = 0
        val_loss_regression = 0
        val_loss_regression_inner = 0
        val_loss_classify = 0
        val_loss_classify_inner = 0
        val_loss_one_zero = 0
        
        # for parms in model.net.parameters():
	    #     print('-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),' -->grad_value:', parms.grad, ' -->grad_fn:', param.grad_fn)
        loss = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            input_edge_features = torch.from_numpy(data['edge_features']).float()

            edge_features = input_edge_features.to(device).requires_grad_(True)
            mesh = data['mesh']
            label = torch.cat(data['target'], dim=0)
            nopad_label = data['no_pad_target']
            classify_label = torch.cat(data['classify_target'], dim=0)
            # classify_label = data['classify_target']
            out = model.net(edge_features, mesh)
            # no_pad_classify_label = []
            # for tensor in nopad_label:
            #     tensor = tensor.to(device)
            #     classified = (tensor > 0.5).float()
            #     no_pad_classify_label.append(classified)
            # no_pad_classify_label = (out > 0.5).float()

            loss1 = MSE(out, label)
            loss2 = criterion(out, nopad_label)

            loss3 = Classify(out, classify_label)
            # loss4 = Classify_inner(out, no_pad_classify_label)
            # loss_weight = weight_criterion(out, label, no_pad_classify_label)

            # accuracy, one_accu, zero_accu = banding_accuracy(out, no_pad_classify_label)
            # loss5 = zero_one_accu(out, classify_label)

            loss_regression += loss1
            loss_regression_inner += loss2
            loss_classify += loss3
            # loss_classify_inner += loss4
            # loss_one_zero += loss5
            loss = 0 * loss1 + 0 * loss2 + 1 * loss3

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        for i, data in enumerate(dataset):
            with torch.no_grad():
                input_val_edge_features = torch.from_numpy(data['val_edge_features']).float()

                val_edge_features = input_val_edge_features.to(device).requires_grad_(True)
                val_mesh = data['val_mesh']
                val_label = torch.cat(data['val_target'], dim=0)
                val_nopad_label = data['no_pad_val_target']
                val_classify_label = torch.cat(data['val_classify_target'], dim=0)
                # val_classify_label = data['val_classify_target']
                val_out = model.net(val_edge_features, val_mesh).squeeze(1)
                # val_no_pad_classify_labelout_classify = []
                # for tensor in val_nopad_label:
                #     tensor = tensor.to(device)
                #     classified = (tensor > 0.5).float()
                #     val_no_pad_classify_labelout_classify.append(classified)

                val_loss1 = MSE(val_out, val_label)
                val_loss2 = criterion(val_out, val_nopad_label)
                val_loss3 = Classify(val_out, val_classify_label)
                # val_loss4 = Classify_inner(val_out, val_no_pad_classify_labelout_classify)
                val_loss_regression += val_loss1
                val_loss_regression_inner += val_loss2
                val_loss_classify += val_loss3
                # val_loss_classify_inner += val_loss4
                val_loss = 0 * val_loss1 + 0 * val_loss2 + 1 * val_loss3

            iter_data_time = time.time()

        # 计算平均 loss 并记录
        avg_train_loss1 = loss_regression.item()
        avg_train_loss2 = loss_regression_inner.item()
        avg_classify_loss = loss_classify.item()
        # avg_classify_loss2 = loss_classify_inner.item()
        train_losses1.append(avg_train_loss1)
        train_losses2.append(avg_train_loss2)
        classify_loss1.append(avg_classify_loss)
        # classify_loss2.append(avg_classify_loss2)

        avg_val_train_loss1 = val_loss_regression.item()
        avg_val_train_loss2 = val_loss_regression_inner.item()
        avg_val_classify_loss = val_loss_classify.item()
        # avg_val_classify_loss2 = val_loss_classify_inner.item()
        val_train_losses1.append(avg_val_train_loss1)
        val_train_losses2.append(avg_val_train_loss2)
        val_classify_loss1.append(avg_val_classify_loss)
        # val_classify_loss2.append(avg_val_classify_loss2)
        epochs.append(epoch)

        if epoch % opt.save_epoch_freq == 0:
            print("MSE:", loss_regression.item(), "MSE_inner:", loss_regression_inner.item(), "BCE:", loss_classify.item(), "val_MSE:", val_loss_regression.item(), "val_MSE_inner:", val_loss_regression_inner.item(), "val BCE:", val_loss_classify.item())
            # print("Classify:", loss_classify.item(), "val_Classify:", val_loss_classify.item())

        if epoch % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
            # model.save_network('latest_ConvENDE_mse+part+BCE+part_01')
            model.save_network('latest_EncoderDecoder_mse+loss5_dp0.5_q1.2')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        print('Learning rate:', optimizer.param_groups[0]['lr'])
        # model.update_learning_rate()
        scheduler.step()

        # # 每 30 个 epoch 绘制并保存 loss 曲线
        # if epoch % 5 == 0 or epoch == opt.niter + opt.niter_decay:
        #     plt.figure(figsize=(10, 6))
        #     lines = [
        #         plt.plot(epochs, train_losses1, 'b-', label='Train main Loss')[0],
        #         plt.plot(epochs, train_losses2, 'r-', label='Train inner Loss')[0],
        #         plt.plot(epochs, classify_loss1, 'y-', label='Train classify Loss')[0],
        #         plt.plot(epochs, classify_loss2, 'g-', label='Train classify inner Loss')[0]
        #     ]

        #     for line in lines:
        #         x_data = line.get_xdata()
        #         y_data = line.get_ydata()
            
        #         plt.scatter(x_data[0], y_data[0], color=line.get_color(), 
        #                 s=100, zorder=5, edgecolors='black')
        #         plt.text(x_data[0], y_data[0], f'{y_data[0]:.2f}', 
        #                 ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
                
        #         plt.scatter(x_data[-1], y_data[-1], color=line.get_color(), 
        #                 s=100, zorder=5, edgecolors='black')
        #         plt.text(x_data[-1], y_data[-1], f'{y_data[-1]:.2f}', 
        #                 ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        #     plt.xlabel('Epoch')
        #     plt.ylabel('Loss')
        #     plt.title('Loss over Epochs')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f'/home/zhuxunyang/coding/banding_detect/result/latest_EncoderDecoder_mse+loss5_dp0.5_epoch_{epoch}.png')
        #     plt.close()

        # # 每 30 个 epoch 绘制并保存 loss 曲线
        # if epoch % 5 == 0 or epoch == opt.niter + opt.niter_decay:
        #     plt.figure(figsize=(10, 6))
        #     lines=[
        #         plt.plot(epochs, val_train_losses1, 'b-', label='Train val main Loss')[0],
        #         plt.plot(epochs, val_train_losses2, 'r-', label='Train val inner Loss')[0],
        #         plt.plot(epochs, val_classify_loss1, 'y-', label='Train val classify Loss')[0],
        #         plt.plot(epochs, val_classify_loss2, 'g-', label='Train val classify inner Loss')[0]
        #     ]
        #     for line in lines:
        #         x_data = line.get_xdata()
        #         y_data = line.get_ydata()
                
        #         plt.scatter(x_data[0], y_data[0], color=line.get_color(), 
        #                 s=100, zorder=5, edgecolors='black')
        #         plt.text(x_data[0], y_data[0], f'{y_data[0]:.2f}', 
        #                 ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        #         plt.scatter(x_data[-1], y_data[-1], color=line.get_color(), 
        #                 s=100, zorder=5, edgecolors='black')
        #         plt.text(x_data[-1], y_data[-1], f'{y_data[-1]:.2f}', 
        #                 ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Loss')
        #     plt.title('Loss over Epochs')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f'/home/zhuxunyang/coding/banding_detect/result/latest_EncoderDecoder_val_mse+loss5_dp0.5_epoch_{epoch}.png')
        #     plt.close()

if __name__ == "__main__":
    main()