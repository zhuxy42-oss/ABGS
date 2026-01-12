import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.label = None
        self.mesh = None
        self.soft_label = None
        self.loss = None
        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, 1, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        # self.criterion = networks.SimplificationLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()

        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)

        self.mesh = data['mesh']

        self.label = torch.from_numpy(data['target']).to(self.device) / 10
        
        # input_list = []
        # target_list = []

        # for mesh in data['original_mesh']:
        #     mesh.vertices = mesh.vertices.to(self.device)
        #     mesh.edges = mesh.edges.to(self.device)
        #     mesh.faces = mesh.faces.to(self.device)
        #     mesh.sizing_values = mesh.sizing_values.to(self.device)
        #     input_list.append(mesh)

        # for mesh in data['target_mesh']:
        #     mesh.vertices = mesh.vertices.to(self.device)
        #     mesh.edges = mesh.edges.to(self.device)
        #     mesh.faces = mesh.faces.to(self.device)
        #     mesh.sizing_values = mesh.sizing_values.to(self.device)
        #     target_list.append(mesh)

        # self.origin_mesh = data['original_mesh']
        # self.label = data['target_mesh']
        # self.origin_mesh = input_list
        # self.label = target_list

        # self.label = torch.cat(data['target_mesh'], dim=0).to(self.device)
        print("dataload success")

    def post_process(self, out):
        masked_out = out.clone().requires_grad_(True)
        
        edge_prob = []
        for i in range(masked_out.size(0)):
            start_index = out.size(1) - self.origin_mesh[i].edges.size(0)
            edge_prob.append(out[i, start_index:])

        collapsed_mesh = []
        for i in range(len(self.origin_mesh)):
            result_mesh = self.origin_mesh[i].collapse_edge(edge_prob[i])
            collapsed_mesh.append(result_mesh)

        return collapsed_mesh


    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, collapsed_mesh):
        self.loss = self.criterion(self.origin_mesh, collapsed_mesh, self.label)
        # loss = self.criterion(self.origin_mesh, collapsed_mesh, self.label)
        print("loss.grad_fn:", self.loss.grad_fn)
        print("loss.requires_grad:", self.loss.requires_grad)
        # self.loss.retain_grad()
        self.loss.backward()
        # print("loss.grad:", self.loss.grad)
        # print("loss backward success")

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        masked_out = out.clone().requires_grad_(True)
        
        edge_prob = []
        labels = []
        for i in range(masked_out.size(0)):
            start_index = out.size(1) - self.origin_mesh[i].edges.size(0)
            edge_prob.append(out[i, start_index:])

        #     labels.append(torch.rand(size=(self.origin_mesh[i].edges.size(0), ), device="cuda"))
        
        # lossing = torch.zeros(size=(len(self.origin_mesh), ), device="cuda")
        # for i in range(len(edge_prob)):
        #     lossing[i] += (torch.nn.MSELoss()(edge_prob[i], labels[i]))
        # self.loss = torch.mean(lossing)

        collapsed_mesh = []
        for i in range(len(self.origin_mesh)):
            result_mesh = self.origin_mesh[i].collapse_edge(edge_prob[i])
            collapsed_mesh.append(result_mesh)
        self.loss = self.criterion(self.origin_mesh, collapsed_mesh, self.label)        

        self.loss.backward()
        for param in self.net.parameters():
            print("inner", param.grad)
        
        # result_mesh = self.post_process(out)
        # self.backward(result_mesh)

        self.optimizer.step()
        # self.optimizer.zero_grad()
        


##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
