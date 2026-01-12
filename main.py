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

# 读取VTK文件
def read_vtk_file(file_path):
    mesh = meshio.read(file_path)
    vertices = mesh.points
    cells = mesh.cells_dict['triangle']  # 假设使用三角形网格
    return vertices, cells

# 处理网格数据为可训练数据
def process_mesh_data(vertices, cells, opt):
    # 这里可以根据具体需求进行更复杂的处理
    # 例如，计算边特征等
    # 为了简单起见，这里假设已经有一个合适的数据集类可以处理这些数据
    dataset = ClassificationData(opt)
    # 这里需要根据实际情况将vertices和cells转换为dataset可以接受的格式
    # 假设可以通过某种方式生成边特征
    edge_features = np.random.rand(750, 5)  # 示例边特征
    data = {'edge_features': edge_features}
    return data

# 主函数
def main():
    # 初始化选项
    opt = TrainOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)

    dataset_size = len(dataset)

    print("Size:", dataset_size)

    # # 读取VTK文件
    # file_path = 'E://dataset//bkgm//Backgroundmesh_daoxiangyepian.vtk'
    # vertices, cells = read_vtk_file(file_path)

    model = create_model(opt)
    # writer = Writer(opt)
    total_steps = 0
    
    print("before training loss.grad:", model.loss)

    for param in model.net.parameters():
        print(param.requires_grad)

    # # 手动设置梯度
    # for name, param in model.net.named_parameters():
    #     gradinet = torch.randn_like(param)
    #     print(param.grad)
    #     param.grad = gradinet
    #     print(param.grad)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        

        # for name, param in model.net.named_parameters():
        #     print(f"参数 {name} 的梯度: {param.grad}", '-->grad_requirs:', param.requires_grad, '--weight', torch.mean(param.data),' -->grad_value:', param.grad, ' -->grad_fn:', param.grad_fn)
 
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print("loss:", model.loss.item())

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == "__main__":
    main()