# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:06:28 2023

@author: Zhao Li
# @File: The HJE Model
"""
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

class MyLoss(nn.Cell):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def construct(self, pred1, tar1):
        pred1 = mindspore.ops.softmax(pred1, axis=1)
        loss = -mindspore.ops.log(pred1[tar1 == 1]).sum()
        return loss

class HJE(BaseClass):

    def __init__(self, n_ent, n_rel, emb_dim, emb_dim1, max_arity, device=None):
        super(HJE, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        self.lmbda = 0.05
        self.max_arity = max_arity
        self.ent_embeddings = mindspore.Parameter(x2ms_adapter.Tensor(self.n_ent, self.emb_dim))
        self.rel_embeddings = mindspore.Parameter(x2ms_adapter.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = x2ms_nn.Embedding(self.max_arity, self.emb_dim)

        self.conv_layer_2 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 2))
        self.conv_layer_3 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 3))
        self.conv_layer_4 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 4))
        self.conv_layer_5 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 5))
        self.conv_layer_6 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 6))
        self.conv_layer_7 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 7))
        self.conv_layer_8 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 8))
        self.conv_layer_9 = x2ms_nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 9))
        self.fc_rel_2 = x2ms_nn.Linear(in_features=self.emb_dim, out_features=16)
        
        self.pool = mindspore.nn.MaxPool3d((4, 1, 1),stride=(4, 1, 1),pad_mode="pad")
        self.pool2d = x2ms_nn.MaxPool2d((1, 2))

        self.inp_drop = x2ms_nn.Dropout(0.2)
        self.dropout = x2ms_nn.Dropout(0.2)
        self.dropout_3d = x2ms_nn.Dropout(0.3)
        self.dropout_2d = x2ms_nn.Dropout(0.3)
        self.nonlinear = x2ms_nn.ReLU()
        self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 4
        self.conv_size_2d = (self.emb_dim) * 16 // 2
        self.fc_layer = x2ms_nn.Linear(in_features=self.conv_size, out_features=self.emb_dim)
        self.W = mindspore.Parameter(mindspore.numpy.empty(shape=(2 * emb_dim, emb_dim)))
        self.fc_1 = x2ms_nn.Linear(in_features=1*self.conv_size_2d, out_features=self.conv_size)
        self.fc_2 = x2ms_nn.Linear(in_features=2*self.conv_size_2d, out_features=self.conv_size)
        self.fc_3 = x2ms_nn.Linear(in_features=3*self.conv_size_2d, out_features=self.conv_size)
        self.fc_4 = x2ms_nn.Linear(in_features=4*self.conv_size_2d, out_features=self.conv_size)
        self.fc_5 = x2ms_nn.Linear(in_features=5*self.conv_size_2d, out_features=self.conv_size)
        self.fc_6 = x2ms_nn.Linear(in_features=6*self.conv_size_2d, out_features=self.conv_size)
        self.fc_7 = x2ms_nn.Linear(in_features=7*self.conv_size_2d, out_features=self.conv_size)
        self.fc_8 = x2ms_nn.Linear(in_features=8*self.conv_size_2d, out_features=self.conv_size)

        self.bn1 = nn.BatchNorm3d(num_features=1)
        self.bn2 = nn.BatchNorm3d(num_features=4)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm1d(num_features=self.conv_size)
        self.b   = mindspore.Parameter(mindspore.ops.zeros(n_ent))
        self.criterion = mindspore.ops.Softplus()

        # 初始化 embeddings 以及卷积层、全连接层的参数
        x2ms_adapter.nn_init.xavier_uniform_(self.ent_embeddings.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.rel_embeddings.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.pos_embeddings.weight.data)
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
        x2ms_adapter.nn_init.xavier_uniform_(self.W)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_1.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_2.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_3.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_4.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_5.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_6.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_7.weight.data)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc_8.weight.data)



    def shift(self, v, sh):
        y = x2ms_adapter.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def conv3d_process(self, concat_input):
        r = x2ms_adapter.tensor_api.view(concat_input[:, 0, :], -1, 1, self.emb_dim1, self.emb_dim2)

        if concat_input.shape[1] == 2:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_2(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 3:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(concat_input[:, 2, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_3(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 4:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(concat_input[:, 2, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(concat_input[:, 3, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_4(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 5:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(concat_input[:, 2, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(concat_input[:, 3, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(concat_input[:, 4, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_5(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 6:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(concat_input[:, 2, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(concat_input[:, 3, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(concat_input[:, 4, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(concat_input[:, 5, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_6(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 7:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(concat_input[:, 2, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(concat_input[:, 3, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(concat_input[:, 4, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(concat_input[:, 5, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e6 = x2ms_adapter.tensor_api.view(concat_input[:, 6, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5, e6), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_7(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 8:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(concat_input[:, 2, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(concat_input[:, 3, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(concat_input[:, 4, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(concat_input[:, 5, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e6 = x2ms_adapter.tensor_api.view(concat_input[:, 6, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e7 = x2ms_adapter.tensor_api.view(concat_input[:, 7, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5, e6, e7), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_8(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 9:
            e1 = x2ms_adapter.tensor_api.view(concat_input[:, 1, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e2 = x2ms_adapter.tensor_api.view(concat_input[:, 2, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e3 = x2ms_adapter.tensor_api.view(concat_input[:, 3, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e4 = x2ms_adapter.tensor_api.view(concat_input[:, 4, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e5 = x2ms_adapter.tensor_api.view(concat_input[:, 5, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e6 = x2ms_adapter.tensor_api.view(concat_input[:, 6, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e7 = x2ms_adapter.tensor_api.view(concat_input[:, 7, :], -1, 1, self.emb_dim1, self.emb_dim2)
            e8 = x2ms_adapter.tensor_api.view(concat_input[:, 8, :], -1, 1, self.emb_dim1, self.emb_dim2)
            cube = x2ms_adapter.cat((r, e1, e2, e3, e4, e5, e6, e7, e8), dim=1)
            x = x2ms_adapter.tensor_api.permute(cube, 0, 2, 3, 1)
            x = x2ms_adapter.tensor_api.unsqueeze(x, 1)
            x = self.bn1(x)
            x = self.conv_layer_9(x)
            x = x2ms_adapter.tensor_api.permute(x, 0, 4, 1, 2, 3)
            x = self.pool(x)

        x = x2ms_adapter.tensor_api.view(x, -1, self.conv_size)
        x = self.dropout_3d(x)
        return x

    def convolve(self, e_emb, r_emb):

        x = x2ms_adapter.tensor_api.view(e_emb, e_emb.shape[0], 1, 1, -1)
        x = self.inp_drop(x)

        k1 = self.fc_rel_2(r_emb)
        k1 = x2ms_adapter.tensor_api.view(k1, -1, 1, 16, 1, 1)
        k1 = x2ms_adapter.tensor_api.view(k1, x2ms_adapter.tensor_api.x2ms_size(e_emb, 0)*16, 1, 1, 1)
        x = x2ms_adapter.tensor_api.permute(x, 1, 0, 2, 3)
        x = x2ms_adapter.nn_functional.conv2d(x, k1, groups=x2ms_adapter.tensor_api.x2ms_size(e_emb, 0))
        x = x2ms_adapter.tensor_api.view(x, x2ms_adapter.tensor_api.x2ms_size(e_emb, 0), 1, 1, 1, -1)
        x = x2ms_adapter.tensor_api.permute(x, 0, 3, 4, 1, 2)
        x = x2ms_adapter.x2ms_sum(x, dim=3)
        x = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(x, 0, 3, 1, 2))

        return x

    def conv2d_process(self, concat_input):
        r = concat_input[:, 0, :]
        if concat_input.shape[1] == 2:
            e1 = concat_input[:, 1, :]
            x = self.convolve(e1, r)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_1(x)
        if concat_input.shape[1] == 3:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            x = x2ms_adapter.cat((conv_e1, conv_e2), dim=1)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_2(x)
        if concat_input.shape[1] == 4:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3), dim=1)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_3(x)
        if concat_input.shape[1] == 5:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4), dim=1)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_4(x)
        if concat_input.shape[1] == 6:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5), dim=1)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_5(x)
        if concat_input.shape[1] == 7:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            conv_e6 = self.convolve(e6, r)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6), dim=1)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_6(x)
        if concat_input.shape[1] == 8:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            e7 = concat_input[:, 7, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            conv_e6 = self.convolve(e6, r)
            conv_e7 = self.convolve(e7, r)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7), dim=1)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_7(x)
        if concat_input.shape[1] == 9:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            e7 = concat_input[:, 7, :]
            e8 = concat_input[:, 8, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            conv_e6 = self.convolve(e6, r)
            conv_e7 = self.convolve(e7, r)
            conv_e8 = self.convolve(e8, r)
            x = x2ms_adapter.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8), dim=1)
            x = self.pool2d(x)
            x = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x), e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_8(x)


        return x

    def construct(self, rel_idx, ent_idx, miss_ent_domain):

        r = x2ms_adapter.tensor_api.unsqueeze(self.rel_embeddings[rel_idx], 1)
        ents = self.ent_embeddings[ent_idx]
        pos = [i for i in range(ent_idx.shape[1]+1) if i + 1 != miss_ent_domain]
        pos = mindspore.tensor(pos)
        pos = pos.unsqueeze(0).expand_as(ent_idx)
        ents = ents + self.pos_embeddings(pos)
        concat_input = x2ms_adapter.cat((r, ents), dim=1)

        v1 = self.conv3d_process(concat_input)
        v2 = self.conv2d_process(concat_input)
        x = v1 + v2
        # x = v2
        # x = self.bn4(x)
        # x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.fc_layer(x)

        miss_ent_domain = x2ms_adapter.to(x2ms_adapter.LongTensor([miss_ent_domain-1]), self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos
        scores = x2ms_adapter.mm(x, x2ms_adapter.tensor_api.transpose(tar_emb, 0, 1))
        scores += x2ms_adapter.tensor_api.expand_as(self.b, scores)

        return scores

