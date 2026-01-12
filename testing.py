from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import torch
import subprocess
from models.layers import CustomMesh
import numpy as np
from label_mesh import point_to_triangle_projection
from models.networks import SimplificationLoss, EdgeCrossEntropyLoss, EdgeRankLoss, ListNetLoss, SpearmanLoss
from models.networks import M2MRankLoss, M2MRegressionLoss, EdgeRankLoss, M2MClassifyLoss
from models.layers.mesh import Mesh
from util.util import pad
import subprocess
import torch.nn as nn
import os
import meshio

def visual_edge(mesh, pred, i):
    red = [1, 0, 0]
    blue = [0, 0, 1]
    colors = []
    threshold = (2 * (2 / torch.pi) * torch.arctan(torch.tensor(1.4, dtype=torch.float32))) - 1
    for r in pred:
        colors.append(red if r > threshold.item() else blue)
    cells = [("line", mesh.edges)]
    cell_data = {"Color": [colors], "banding_condition":list(pred)}
    meshio_mesh = meshio.Mesh(
        points=mesh.vs,
        cells=cells,
        cell_data=cell_data
    )
    meshio_mesh.write(os.path.join("/home/zhuxunyang/coding/banding_detect", f"colored_edge_banding_{i}.vtk"))

def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    criterion = M2MRegressionLoss()
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()
    classify = nn.CrossEntropyLoss()
    # test
    # writer.reset_counter()
    print("Dataset length:", len(dataset))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sum_edge = 0
    sum_true = 0
    sum_pred_banding = 0
    sum_label_banding = 0
    sum_loss = 0
    for n, data in enumerate(dataset):
        with torch.no_grad():

            input_edge_features = torch.from_numpy(data['val_edge_features']).float()
            edge_features = input_edge_features.to(device).requires_grad_(True)
            mesh = data['val_mesh']
            # original_mesh = data['val_original_mesh'][0]
            label = data['val_target'][0]
            classify_label = data['val_classify_target'][0]
            
            edge_count = len(mesh[0].edges)

            out = model.net(edge_features.float().to(device), mesh)
            edge_len = len(mesh[0].edges)
            true_predict = out[:, 0:edge_len]
            true_label = label[:, 0:edge_len]
            one_loss = MSE(true_predict, true_label)
            sum_loss += one_loss
            threshold = 0.5
            
            label_banding_edge = 0
            label_no_banding_edge = 0
            for i in range(edge_count):
                if (true_label[0][i] > threshold):
                    label_banding_edge += 1
                else:
                    label_no_banding_edge += 1

            true_count = 0
            pred_true_banding_edge = 0
            pred_true_no_banding_edge = 0
            pred_false_no_banding = 0
            pred_false_banding = 0

            for i in range(edge_count):
                rito_pred = true_predict[0][i]
                rito_label = true_label[0][i]
                print("pred:", rito_pred, "label:", rito_label)
                if (rito_label > threshold and rito_pred > threshold):
                    pred_true_banding_edge += 1
                    true_count +=1
                    sum_pred_banding += 1
                    sum_label_banding += 1
                elif (rito_label < threshold and rito_pred < threshold):
                    pred_true_no_banding_edge += 1
                    true_count +=1
                    # label_no_banding_edge += 1
                elif(rito_label < threshold and rito_pred > threshold):
                    pred_false_no_banding += 1
                    # label_banding_edge += 1
                elif(rito_label > threshold and rito_pred < threshold):
                    pred_false_banding += 1
                    sum_label_banding += 1
                    # label_no_banding_edge += 1
            print(f"{n:<5} Accuracy: {true_count/edge_count:>8.4f}  Correct Banding: {pred_true_banding_edge:<4}/{label_banding_edge:<4} Correct No-banding: {pred_true_no_banding_edge:<4}/{label_no_banding_edge:<4} Error banding: {pred_false_banding:<4} Error No-banding: {pred_false_no_banding:<4} Sum-edge: {edge_count:<4}")
            print(" ")
            sum_true += true_count
            sum_edge += edge_count

            # visual_edge(mesh[0], out[0, 0:edge_len], n)
    print("Sum accuracy:", sum_true / sum_edge)
    print("Sum banding accuracry:", sum_pred_banding / sum_label_banding, "=", sum_pred_banding, "/",sum_label_banding)
    print("Sum Loss:", sum_loss.item())


def run_classify():
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    criterion = M2MRegressionLoss()
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()
    classify = nn.BCEWithLogitsLoss()
    Classify = M2MClassifyLoss()
    # test
    # writer.reset_counter()
    print("Dataset length:", len(dataset))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sum_edge = 0
    sum_true = 0
    sum_pred_banding = 0
    sum_label_banding = 0
    sum_loss = 0
    for n, data in enumerate(dataset):
        with torch.no_grad():
            input_edge_features = torch.from_numpy(data['val_edge_features']).float()
            edge_features = input_edge_features.to(device).requires_grad_(True)
            mesh = data['val_mesh']
            label = data['val_target'][0]
            classify_label = data['val_classify_target'][0]
            val_nopad_label = data['no_pad_val_target']
            val_no_pad_classify_labelout_classify = []
            for tensor in val_nopad_label:
                    tensor = tensor.to(device)
                    classified = (tensor > 0.5).float()
                    val_no_pad_classify_labelout_classify.append(classified)
            
            edge_count = len(mesh[0].edges)
            out = model.net(edge_features.float().to(device), mesh).squeeze(1)
            edge_len = len(mesh[0].edges)
            true_predict = out[:, 0:edge_len]
            true_label = classify_label[:, 0:edge_len]

            # print("MSE:", MSE(out, label).item(),"BCE:", classify(out, classify_label).item(), "part:", Classify(out, val_no_pad_classify_labelout_classify).item())
            sum_loss += MSE(out, label)

            label_banding_edge = 0
            label_no_banding_edge = 0
            for i in range(edge_count):
                if (true_label[0][i] == 1):
                    label_banding_edge += 1
                else:
                    label_no_banding_edge += 1

            true_count = 0
            pred_true_banding_edge = 0
            pred_true_no_banding_edge = 0
            pred_false_no_banding = 0
            pred_false_banding = 0

            for i in range(edge_count):
                rito_pred0 = true_predict[0][i]
                rito_label = true_label[0][i]
                if (rito_pred0 > 0.5 and rito_label == 1):
                    pred_true_banding_edge += 1
                    true_count +=1
                    sum_pred_banding += 1
                elif (rito_pred0 <= 0.5 and rito_label == 0):
                    pred_true_no_banding_edge += 1
                    true_count +=1
                elif(rito_pred0 <= 0.5 and rito_label == 1):
                    pred_false_no_banding += 1
                elif(rito_pred0 > 0.5 and rito_label == 0):
                    pred_false_banding += 1
            print(f"{n:<5} Accuracy: {true_count/edge_count:>8.4f}  Correct Banding: {pred_true_banding_edge:<4}/{label_banding_edge:<4} Correct No-banding: {pred_true_no_banding_edge:<4}/{label_no_banding_edge:<4} Error banding: {pred_false_banding:<4} Error No-banding: {pred_false_no_banding:<4} Sum-edge: {edge_count:<4}")
            print(" ")
            sum_true += true_count
            sum_edge += edge_count
            sum_label_banding += label_banding_edge
        
        if n == 39:
            break

    print("Sum accuracy:", sum_true / sum_edge)
    print("Sum banding accuracry:", sum_pred_banding / sum_label_banding, "=", sum_pred_banding, "/",sum_label_banding)
    print("Sum Loss:", sum_loss.item())
            

                
            



if __name__ == '__main__':
    run_classify()
