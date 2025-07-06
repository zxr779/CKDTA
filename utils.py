import os
import os.path as osp
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, Batch, HeteroData
from torch_geometric import data as DATA
import torch_geometric.transforms as T
import torch
from lifelines.utils import concordance_index
import torch_geometric
from tqdm import tqdm
import torch.nn.functional as F
def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None,smile_mtgraph=None, smile_tensor=None, target_graph=None, target_key=None,smile=None,pro=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.process(xd, xt, y, smile_graph, smile_tensor, target_graph, target_key,smile,pro)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_DrugData.pt', self.dataset + '_TargetData.pt',self.dataset + '_MdtiData.pt']

    def download(self):
        # Download to self.raw_dir.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph, smile_tensor, target_graph, target_key,smile,pro):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        Drug_data_list = []
        Target_data_list = []
        data_len = len(xd)
        print('data_len:', data_len)
        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            key = target_key[i]
            c_size, features, edge_index,attribute_features,attribute_edge_index,relation_features,fragfeature,atom_to_frag,frag_edg_index = smile_graph[smiles]
            smile_ten = smile_tensor[smiles]

            tar_size, tar_features, tar_edge_index = target_graph[key]
            DrugData=HeteroData()
            DrugData['atom'].x = torch.tensor(np.array(features), dtype=torch.float)
            DrugData['attribute'].x = torch.tensor(np.array(attribute_features), dtype=torch.float)
            DrugData['frag'].x = torch.tensor(np.array(fragfeature), dtype=torch.float)

            DrugData[('atom', 'bond', 'atom')].edge_index = torch.LongTensor(np.array(edge_index).transpose(1, 0))
            DrugData[('attribute', 'related_to', 'atom')].edge_index = torch.LongTensor(np.array(attribute_edge_index))
            DrugData[('attribute', 'related_to', 'atom')].edge_attr = torch.tensor(np.array(relation_features),
                                                                                   dtype=torch.float)
            DrugData[('atom', 'part_of', 'frag')].edge_index = torch.LongTensor(np.array(atom_to_frag))
            DrugData[('frag', 'bond', 'frag')].edge_index = torch.LongTensor(np.array(frag_edg_index).transpose(1, 0))

            DrugData['atom'].y = torch.FloatTensor([labels])
            DrugData.smiles = torch.LongTensor(np.array([smile_ten]))
            DrugData.smilesequ = smile
            DrugData.y = torch.FloatTensor([labels])
            DrugData.__setitem__('c_size', torch.LongTensor([c_size]))

            # Create TargetData
            TargetData = DATA.Data(
                x=torch.Tensor(np.array([tar_features])).view(-1, 54),
                edge_index=torch.LongTensor(np.array(tar_edge_index).transpose(1, 0)),
                y=torch.FloatTensor([labels])
            )
            TargetData.target = torch.LongTensor(np.array([target]))

            TargetData.__setitem__('tar_size', torch.LongTensor([tar_size]))

            Drug_data_list.append(DrugData)
            Target_data_list.append(TargetData)

        if self.pre_filter is not None:
            Drug_data_list = [data for data in Drug_data_list if self.pre_filter(data)]
            Target_data_list = [data for data in Target_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            Drug_data_list = [self.pre_transform(data) for data in Drug_data_list]
            Target_data_list = [self.pre_transform(data) for data in Target_data_list]
        print('Graph construction done. Saving to file.')
        self.DrugData = Drug_data_list
        self.TargetData = Target_data_list

    def __len__(self):
        return len(self.DrugData)

    def __getitem__(self, idx):
        return (self.DrugData[idx], self.TargetData[idx])
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    LOG_INTERVAL = 256
    loss_fn = torch.nn.SmoothL1Loss()

    total_loss = 0  # Initialize total_loss to accumulate losses
    num_batches = 0  # Counter for the batches
    for batch_idx, data in enumerate(train_loader):
        DrugData = data[0].to(device)
        TargetData = data[1].to(device)
        optimizer.zero_grad()
        output= model(DrugData, TargetData)
        loss = loss_fn(output, DrugData.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()


        total_loss += loss.item()  # Add current batch's loss
        num_batches += 1  # Increment batch counter
        # print("num_batches:"+str(num_batches))
        # print("batch_idx:" + str(batch_idx))
        # print("LOG_INTERVAL:" + str(LOG_INTERVAL))
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * TRAIN_BATCH_SIZE,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))

    average_loss = total_loss / num_batches  # Calculate average loss over all batches
    return average_loss

def evaluate(model, device, val_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0  # Initialize total_loss to accumulate losses
    num_batches = 0  # Counter for the batches

    with torch.no_grad():  # No need to compute gradients (for efficiency)
        for data in val_loader:
            DrugData = data[0].to(device)
            TargetData = data[1].to(device)

            output = model(DrugData, TargetData)
            loss= loss_fn(output, DrugData.y.view(-1, 1).float().to(device))

            total_loss += loss.item()  # Add current batch's loss
            num_batches += 1  # Increment batch counter

    average_loss = total_loss / num_batches  # Calculate average loss over all batches
    return average_loss


# predicting function
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
from torch_geometric.data import Batch

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


# metrics_rmse
def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


# metrics_mse
def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse



# metrics_pearson
def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


# metrics_spearman
def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


# metrics_ci
def ci_tensor(f, y):  # p,y
    y = np.array(y.tolist())
    f = np.array(f.tolist())
    ind = np.argsort(y, axis=0)
    ind = [x[0] for x in ind]
    y = [y[i][0] for i in ind]
    f = [f[i][0] for i in ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return torch.tensor(ci)


# metrics_ci
def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


#  r2 for r2m
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


# r0 for r2m
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


# metrics_rm2
def rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))