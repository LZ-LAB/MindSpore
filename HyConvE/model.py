# -- coding: utf-8 --

import numpy as np
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn

class BaseClass(nn.Cell):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = mindspore.Parameter(mindspore.tensor(0, dtype=mindspore.int32), requires_grad=False)
        self.best_mrr = mindspore.Parameter(mindspore.tensor(0, dtype=mindspore.float64), requires_grad=False)
        self.best_itr = mindspore.Parameter(mindspore.tensor(0, dtype=mindspore.int32), requires_grad=False)
        self.best_hit1 = mindspore.Parameter(mindspore.tensor(0, dtype=mindspore.float64), requires_grad=False)


class HyConvE(BaseClass):

    def __init__(self, dataset, emb_dim, emb_dim1):
        super(HyConvE, self).__init__()

        self.dataset = dataset
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        self.lmbda = 0.15
        self.ent_embeddings = x2ms_nn.Embedding(self.dataset.num_ent, self.emb_dim)
        self.rel_embeddings = x2ms_nn.Embedding(self.dataset.num_rel, self.emb_dim)



        self.conv_layer_2 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 3))
        self.conv_layer_3 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 4))
        self.conv_layer_4 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 5))
        self.conv_layer_5 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 6))
        self.conv_layer_6 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 7))
        self.conv_layer_7 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 8))
        self.conv_layer_8 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 9))
        self.conv_layer_9 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 10))
        self.fc_pos = x2ms_nn.Linear(in_features=self.dataset.arity_lst[-1], out_features=9)
        self.fc_rel_2 = x2ms_nn.Linear(in_features=self.emb_dim, out_features=3)
        self.pool = mindspore.nn.MaxPool3d((2, 1, 1),stride=(2, 1, 1),pad_mode="pad")
        self.pool1d = x2ms_nn.MaxPool2d((1, 2))

        self.inp_drop = x2ms_nn.Dropout(0.2)
        self.dropout = x2ms_nn.Dropout(0.2)
        self.dropout_3d = x2ms_nn.Dropout(0.2)
        self.dropout_2d = x2ms_nn.Dropout(0.2)
        self.nonlinear = x2ms_nn.ReLU()
        self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 2
        self.conv_size_1d = (self.emb_dim) * 3 // 2
        self.fc_layer = x2ms_nn.Linear(in_features=self.conv_size, out_features=1)
        self.fc_2 = x2ms_nn.Linear(in_features=2*self.conv_size_1d, out_features=self.conv_size)
        self.fc_3 = x2ms_nn.Linear(in_features=3*self.conv_size_1d, out_features=self.conv_size)
        self.fc_4 = x2ms_nn.Linear(in_features=4*self.conv_size_1d, out_features=self.conv_size)
        self.fc_5 = x2ms_nn.Linear(in_features=5*self.conv_size_1d, out_features=self.conv_size)
        self.fc_6 = x2ms_nn.Linear(in_features=6*self.conv_size_1d, out_features=self.conv_size)
        self.fc_7 = x2ms_nn.Linear(in_features=7*self.conv_size_1d, out_features=self.conv_size)
        self.fc_8 = x2ms_nn.Linear(in_features=8*self.conv_size_1d, out_features=self.conv_size)
        self.fc_9 = x2ms_nn.Linear(in_features=9*self.conv_size_1d, out_features=self.conv_size)

        self.bn1 = mindspore.nn.BatchNorm3d(num_features=1)
        self.bn2 = mindspore.nn.BatchNorm3d(num_features=4)
        self.bn3 = mindspore.nn.BatchNorm2d(num_features=1)
        self.bn4 = mindspore.nn.BatchNorm1d(num_features=self.conv_size)
        self.criterion = mindspore.ops.Softplus()

        # 初始化 embeddings 以及卷积层、全连接层的参数
        x2ms_adapter.nn_init.xavier_uniform_(self.ent_embeddings.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.rel_embeddings.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_2.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_3.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_4.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_5.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_6.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_7.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_8.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.conv_layer_9.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_layer.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_rel_2.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_2.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_3.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_4.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_5.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_6.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_7.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_8.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_9.weight.data)



    def shift(self, v, sh):
        y = x2ms_adapter.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def conv3d_process(self, batch):
        if len(batch) == 3:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_2(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 4:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_3(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)


        if len(batch) == 5:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_4(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 6:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_5(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 7:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, self.emb_dim1, self.emb_dim2)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5, e6), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_6(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 8:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, self.emb_dim1, self.emb_dim2)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, self.emb_dim1, self.emb_dim2)
            e7 = x2ms_adapter.tensor_api.view(batch[7], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5, e6, e7), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_7(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 9:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, self.emb_dim1, self.emb_dim2)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, self.emb_dim1, self.emb_dim2)
            e7 = x2ms_adapter.tensor_api.view(batch[7], -1, 1, self.emb_dim1, self.emb_dim2)
            e8 = x2ms_adapter.tensor_api.view(batch[8], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5, e6, e7, e8), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_8(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 10:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim1, self.emb_dim2)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, self.emb_dim1, self.emb_dim2)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, self.emb_dim1, self.emb_dim2)
            e7 = x2ms_adapter.tensor_api.view(batch[7], -1, 1, self.emb_dim1, self.emb_dim2)
            e8 = x2ms_adapter.tensor_api.view(batch[8], -1, 1, self.emb_dim1, self.emb_dim2)
            e9 = x2ms_adapter.tensor_api.view(batch[9], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5, e6, e7, e8, e9), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_9(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        x = x2ms_adapter.tensor_api.view(x, -1, self.conv_size)

        x = self.dropout_3d(x)

        return x

    def convolve(self, e_emb, r_emb, pos):

        x = e_emb
        x = self.inp_drop(x)

        k1 = self.fc_rel_2(r_emb)
        k1 = x2ms_adapter.tensor_api.view(k1, -1, 1, 3, 1, 1)
        k1 = x2ms_adapter.tensor_api.view(k1, x2ms_adapter.tensor_api.x2ms_size(e_emb, 0)*3, 1, 1, 1)
        x = x2ms_adapter.tensor_api.permute(x, 1, 0, 2, 3)
        x = x2ms_adapter.nn_functional.conv2d(x, k1, groups=x2ms_adapter.tensor_api.x2ms_size(e_emb, 0))


        one_hot_target = x2ms_adapter.to(x2ms_adapter.tensor_api.x2ms_float((pos == x2ms_adapter.arange(self.dataset.arity_lst[-1]).reshape(self.dataset.arity_lst[-1]))), self.dataset.device)
        poses = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.repeat(one_hot_target, r_emb.shape[0]), -1, self.dataset.arity_lst[-1])
        one_hot_target.requires_grad = False
        poses.requires_grad = False

        k = self.fc_pos(poses)
        k = x2ms_adapter.tensor_api.view(k, x2ms_adapter.tensor_api.x2ms_size(e_emb, 0)*3, 3, 1, 1)
        x = x2ms_adapter.nn_functional.conv2d(x, k, groups=x2ms_adapter.tensor_api.x2ms_size(e_emb, 0), stride=1)
        x = x2ms_adapter.tensor_api.view(x, x2ms_adapter.tensor_api.x2ms_size(e_emb, 0), 1, 3, 1, -1)
        x = x2ms_adapter.tensor_api.permute(x, 0, 3, 4, 1, 2)
        x = x2ms_adapter.x2ms_sum(x, dim=3)
        x = x2ms_adapter.tensor_api.permute(x, 0, 3, 1, 2).contiguous()

        return x

    def conv2d_process(self, batch):
        if len(batch) == 3:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)

            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            x = x2ms_adapter.cat((conv_e1, conv_e2), dim=1)

            x = self.pool1d(x)


            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)

            x = self.nonlinear(x)
            x = self.dropout(x)

            x = self.fc_2(x)

            return x

        if len(batch) == 4:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, 1, self.emb_dim)
            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            conv_e3 = x2ms_adapter.tensor_api.permute(self.convolve(e3, r, 2), 0, 2, 1, 3)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3), dim=1)
            x = self.pool1d(x)

            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_3(x)

            return x

        if len(batch) == 5:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, 1, self.emb_dim)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, 1, self.emb_dim)
            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            conv_e3 = x2ms_adapter.tensor_api.permute(self.convolve(e3, r, 2), 0, 2, 1, 3)
            conv_e4 = x2ms_adapter.tensor_api.permute(self.convolve(e4, r, 3), 0, 2, 1, 3)

            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4), dim=1)
            x = self.pool1d(x)

            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_4(x)

            return x

        if len(batch) == 6:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, 1, self.emb_dim)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, 1, self.emb_dim)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, 1, self.emb_dim)
            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            conv_e3 = x2ms_adapter.tensor_api.permute(self.convolve(e3, r, 2), 0, 2, 1, 3)
            conv_e4 = x2ms_adapter.tensor_api.permute(self.convolve(e4, r, 3), 0, 2, 1, 3)
            conv_e5 = x2ms_adapter.tensor_api.permute(self.convolve(e5, r, 4), 0, 2, 1, 3)

            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5), dim=1)
            x = self.pool1d(x)


            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_5(x)

            return x

        if len(batch) == 7:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, 1, self.emb_dim)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, 1, self.emb_dim)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, 1, self.emb_dim)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, 1, self.emb_dim)
            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            conv_e3 = x2ms_adapter.tensor_api.permute(self.convolve(e3, r, 2), 0, 2, 1, 3)
            conv_e4 = x2ms_adapter.tensor_api.permute(self.convolve(e4, r, 3), 0, 2, 1, 3)
            conv_e5 = x2ms_adapter.tensor_api.permute(self.convolve(e5, r, 4), 0, 2, 1, 3)
            conv_e6 = x2ms_adapter.tensor_api.permute(self.convolve(e6, r, 5), 0, 2, 1, 3)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6), dim=1)
            x = self.pool1d(x)

            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_6(x)
            return x

        if len(batch) == 8:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, 1, self.emb_dim)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, 1, self.emb_dim)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, 1, self.emb_dim)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, 1, self.emb_dim)
            e7 = x2ms_adapter.tensor_api.view(batch[7], -1, 1, 1, self.emb_dim)
            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            conv_e3 = x2ms_adapter.tensor_api.permute(self.convolve(e3, r, 2), 0, 2, 1, 3)
            conv_e4 = x2ms_adapter.tensor_api.permute(self.convolve(e4, r, 3), 0, 2, 1, 3)
            conv_e5 = x2ms_adapter.tensor_api.permute(self.convolve(e5, r, 4), 0, 2, 1, 3)
            conv_e6 = x2ms_adapter.tensor_api.permute(self.convolve(e6, r, 5), 0, 2, 1, 3)
            conv_e7 = x2ms_adapter.tensor_api.permute(self.convolve(e7, r, 6), 0, 2, 1, 3)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7), dim=1)
            x = self.pool1d(x)

            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_7(x)
            return x


        if len(batch) == 9:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, 1, self.emb_dim)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, 1, self.emb_dim)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, 1, self.emb_dim)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, 1, self.emb_dim)
            e7 = x2ms_adapter.tensor_api.view(batch[7], -1, 1, 1, self.emb_dim)
            e8 = x2ms_adapter.tensor_api.view(batch[8], -1, 1, 1, self.emb_dim)
            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            conv_e3 = x2ms_adapter.tensor_api.permute(self.convolve(e3, r, 2), 0, 2, 1, 3)
            conv_e4 = x2ms_adapter.tensor_api.permute(self.convolve(e4, r, 3), 0, 2, 1, 3)
            conv_e5 = x2ms_adapter.tensor_api.permute(self.convolve(e5, r, 4), 0, 2, 1, 3)
            conv_e6 = x2ms_adapter.tensor_api.permute(self.convolve(e6, r, 5), 0, 2, 1, 3)
            conv_e7 = x2ms_adapter.tensor_api.permute(self.convolve(e7, r, 6), 0, 2, 1, 3)
            conv_e8 = x2ms_adapter.tensor_api.permute(self.convolve(e8, r, 7), 0, 2, 1, 3)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8), dim=1)
            x = self.pool1d(x)

            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_8(x)

            return x

        if len(batch) == 10:
            r = x2ms_adapter.tensor_api.view(batch[0], -1, 1, 1, self.emb_dim)
            e1 = x2ms_adapter.tensor_api.view(batch[1], -1, 1, 1, self.emb_dim)
            e2 = x2ms_adapter.tensor_api.view(batch[2], -1, 1, 1, self.emb_dim)
            e3 = x2ms_adapter.tensor_api.view(batch[3], -1, 1, 1, self.emb_dim)
            e4 = x2ms_adapter.tensor_api.view(batch[4], -1, 1, 1, self.emb_dim)
            e5 = x2ms_adapter.tensor_api.view(batch[5], -1, 1, 1, self.emb_dim)
            e6 = x2ms_adapter.tensor_api.view(batch[6], -1, 1, 1, self.emb_dim)
            e7 = x2ms_adapter.tensor_api.view(batch[7], -1, 1, 1, self.emb_dim)
            e8 = x2ms_adapter.tensor_api.view(batch[8], -1, 1, 1, self.emb_dim)
            e9 = x2ms_adapter.tensor_api.view(batch[9], -1, 1, 1, self.emb_dim)
            conv_e1 = x2ms_adapter.tensor_api.permute(self.convolve(e1, r, 0), 0, 2, 1, 3)
            conv_e2 = x2ms_adapter.tensor_api.permute(self.convolve(e2, r, 1), 0, 2, 1, 3)
            conv_e3 = x2ms_adapter.tensor_api.permute(self.convolve(e3, r, 2), 0, 2, 1, 3)
            conv_e4 = x2ms_adapter.tensor_api.permute(self.convolve(e4, r, 3), 0, 2, 1, 3)
            conv_e5 = x2ms_adapter.tensor_api.permute(self.convolve(e5, r, 4), 0, 2, 1, 3)
            conv_e6 = x2ms_adapter.tensor_api.permute(self.convolve(e6, r, 5), 0, 2, 1, 3)
            conv_e7 = x2ms_adapter.tensor_api.permute(self.convolve(e7, r, 6), 0, 2, 1, 3)
            conv_e8 = x2ms_adapter.tensor_api.permute(self.convolve(e8, r, 7), 0, 2, 1, 3)
            conv_e9 = x2ms_adapter.tensor_api.permute(self.convolve(e9, r, 8), 0, 2, 1, 3)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8, conv_e9), dim=1)
            x = self.pool1d(x)

            x = x2ms_adapter.tensor_api.view(x, e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_9(x)

            return x

    def construct(self, batch, labels):

        r = self.rel_embeddings(batch[:, 0])
        ents = self.ent_embeddings(batch[:, 1:])

        e1 = ents[:, 0]
        e2 = ents[:, 1]
        if batch.shape[1] == 3:
            x1 = self.conv3d_process((r, e1, e2))
            x2 = self.conv2d_process((r, e1, e2))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)

            batch_score = -x2ms_adapter.tensor_api.view(x, -1)
            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2)
            for p in x2ms_adapter.parameters(self.conv_layer_2):
                l2_regular += mindspore.ops.norm(p)##p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()


            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[1] == 4:
            e3 = ents[:, 2]

            x1 = self.conv3d_process((r, e1, e2, e3))
            x2 = self.conv2d_process((r, e1, e2, e3))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x2ms_adapter.tensor_api.view(x, -1)
            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2) + x2ms_adapter.x2ms_mean(e3 ** 2)

            for p in x2ms_adapter.parameters(self.conv_layer_3):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_3):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()

            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[1] == 5:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            x1 = self.conv3d_process((r, e1, e2, e3, e4))
            x2 = self.conv2d_process((r, e1, e2, e3, e4))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x2ms_adapter.tensor_api.view(x, -1)

            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2) + x2ms_adapter.x2ms_mean(e3 ** 2) + x2ms_adapter.x2ms_mean(e4 ** 2)

            for p in x2ms_adapter.parameters(self.conv_layer_4):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_4):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()

            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[1] == 6:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x2ms_adapter.tensor_api.view(x, -1)
            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2) + x2ms_adapter.x2ms_mean(e3 ** 2) + x2ms_adapter.x2ms_mean(e4 ** 2) + x2ms_adapter.x2ms_mean(e5 ** 2)

            for p in x2ms_adapter.parameters(self.conv_layer_5):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_5):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()

            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[1] == 7:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]


            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x2ms_adapter.tensor_api.view(x, -1)
            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2) + x2ms_adapter.x2ms_mean(e3 ** 2) + x2ms_adapter.x2ms_mean(e4 ** 2) + x2ms_adapter.x2ms_mean(e5 ** 2) + x2ms_adapter.x2ms_mean(e6 ** 2)

            for p in x2ms_adapter.parameters(self.conv_layer_6):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_6):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()

            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[1] == 8:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x2ms_adapter.tensor_api.view(x, -1)
            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2) + x2ms_adapter.x2ms_mean(e3 ** 2) + x2ms_adapter.x2ms_mean(e4 ** 2) + x2ms_adapter.x2ms_mean(e5 ** 2) + x2ms_adapter.x2ms_mean(e6 ** 2) + x2ms_adapter.x2ms_mean(e7 ** 2)

            for p in x2ms_adapter.parameters(self.conv_layer_7):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_7):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()

            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular


        if batch.shape[1] == 9:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x2ms_adapter.tensor_api.view(x, -1)
            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2) + x2ms_adapter.x2ms_mean(e3 ** 2) + x2ms_adapter.x2ms_mean(e4 ** 2) + x2ms_adapter.x2ms_mean(e5 ** 2) + x2ms_adapter.x2ms_mean(e6 ** 2) + x2ms_adapter.x2ms_mean(e7 ** 2) + x2ms_adapter.x2ms_mean(e8 ** 2)

            for p in x2ms_adapter.parameters(self.conv_layer_8):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_8):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()

            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[1] == 10:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]
            e9 = ents[:, 8]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)

            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x2ms_adapter.tensor_api.view(x, -1)
            l2_regular = x2ms_adapter.x2ms_mean(r ** 2) + x2ms_adapter.x2ms_mean(e1 ** 2) + x2ms_adapter.x2ms_mean(e2 ** 2) + x2ms_adapter.x2ms_mean(e3 ** 2) + x2ms_adapter.x2ms_mean(e4 ** 2) + x2ms_adapter.x2ms_mean(e5 ** 2) + x2ms_adapter.x2ms_mean(e6 ** 2) + x2ms_adapter.x2ms_mean(e7 ** 2) + x2ms_adapter.x2ms_mean(e8 ** 2) + x2ms_adapter.x2ms_mean(e9 ** 2)

            for p in x2ms_adapter.parameters(self.conv_layer_9):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_layer):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_rel_2):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_pos):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.fc_9):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool):
                l2_regular += p.norm()
            for p in x2ms_adapter.parameters(self.pool1d):
                l2_regular += p.norm()

            mean = x2ms_adapter.x2ms_mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        return mean + regular

    def predict(self, test_batch):
        r = self.rel_embeddings(test_batch[:, 0])
        ents = self.ent_embeddings(test_batch[:, 1:])
        e1 = ents[:, 0]
        e2 = ents[:, 1]
        if test_batch.shape[1] == 3:
            x1 = self.conv3d_process((r, e1, e2))
            x2 = self.conv2d_process((r, e1, e2))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)
        if test_batch.shape[1] == 4:
            e3 = ents[:, 2]
            x1 = self.conv3d_process((r, e1, e2, e3))
            x2 = self.conv2d_process((r, e1, e2, e3))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)
        if test_batch.shape[1] == 5:
            e3 = ents[:, 2]
            e4 = ents[:, 3]

            x1 = self.conv3d_process((r, e1, e2, e3, e4))
            x2 = self.conv2d_process((r, e1, e2, e3, e4))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)
        if test_batch.shape[1] == 6:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)
        if test_batch.shape[1] == 7:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)

        if test_batch.shape[1] == 8:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)

        if test_batch.shape[1] == 9:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)

        if test_batch.shape[1] == 10:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]
            e9 = ents[:, 8]
            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x2ms_adapter.tensor_api.view(x, -1)

        return score
