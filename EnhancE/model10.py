# -- coding: utf-8 --
# @Time: 2022-03-27 15:50
# @Author: WangCx
# @File: model
# @Project: HypergraphNN
from tqdm import tqdm
import random
import math
import random
import math
import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, XavierNormal
from mindspore import Parameter
from Mindspore.EnhancE.EnhenceE_x2ms import x2ms_adapter
from Mindspore.EnhancE.EnhenceE_x2ms.x2ms_adapter.torch_api.nn_api import nn as x2ms_nn


class BaseClass(nn.Cell):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = mindspore.Parameter(mindspore.Tensor(0, dtype=mindspore.int32), requires_grad=False)
        self.best_mrr = mindspore.Parameter(mindspore.Tensor(0, dtype=mindspore.float64), requires_grad=False)
        self.best_itr = mindspore.Parameter(mindspore.Tensor(0, dtype=mindspore.int32), requires_grad=False)


class HyperNet(BaseClass):
    def __init__(self, dataset, emb_dim, hidden_drop):
        super(HyperNet, self).__init__()
        self.emb_dim = emb_dim
        self.E = x2ms_nn.Embedding(dataset.num_ent, emb_dim, padding_idx=0)
        self.R = x2ms_nn.Embedding(dataset.num_rel, emb_dim, padding_idx=0)
        # self.E = mindspore.nn.Embedding(dataset.num_ent, emb_dim, padding_idx=0)
        # self.R = mindspore.nn.Embedding(dataset.num_rel, emb_dim, padding_idx=0)


        self.W02 = mindspore.Parameter(mindspore.numpy.empty(shape=(3*emb_dim, 200)))
        self.W03 = mindspore.Parameter(mindspore.numpy.empty(shape=(3*emb_dim, 200)))
        self.W04 = mindspore.Parameter(mindspore.numpy.empty(shape=(3*emb_dim, 200)))
        self.W05 = mindspore.Parameter(mindspore.numpy.empty(shape=(3*emb_dim, 200)))
        self.W06 = mindspore.Parameter(mindspore.numpy.empty(shape=(3*emb_dim, 200)))
        self.W1 = mindspore.Parameter(mindspore.numpy.empty(shape=(2*emb_dim, 200)))
        self.W2 = mindspore.Parameter(mindspore.numpy.empty(shape=(emb_dim, 200)))
        self.W3 = mindspore.Parameter(mindspore.numpy.empty(shape=(emb_dim, 200)))

        self.a2 = mindspore.Parameter(mindspore.numpy.empty(shape=(200, 1)))
        self.a3 = mindspore.Parameter(mindspore.numpy.empty(shape=(200, 1)))
        self.a4 = mindspore.Parameter(mindspore.numpy.empty(shape=(200, 1)))
        self.a5 = mindspore.Parameter(mindspore.numpy.empty(shape=(200, 1)))
        self.a6 = mindspore.Parameter(mindspore.numpy.empty(shape=(200, 1)))
        self.E.weight.data = Parameter(initializer(XavierNormal(), self.E.weight.data.shape, mindspore.float32))
        self.R.weight.data = Parameter(initializer(XavierNormal(), self.R.weight.data.shape, mindspore.float32))
        self.W02 = Parameter(initializer(XavierNormal(), self.W02.data.shape, mindspore.float32))
        self.W03 = Parameter(initializer(XavierNormal(), self.W03.data.shape, mindspore.float32))
        self.W04 = Parameter(initializer(XavierNormal(), self.W04.data.shape, mindspore.float32))
        self.W05 = Parameter(initializer(XavierNormal(), self.W05.data.shape, mindspore.float32))
        self.W06 = Parameter(initializer(XavierNormal(), self.W06.data.shape, mindspore.float32))
        self.a2 = Parameter(initializer(XavierNormal(), self.a2.shape, mindspore.float32))
        self.a3 = Parameter(initializer(XavierNormal(), self.a3.shape, mindspore.float32))
        self.a4 = Parameter(initializer(XavierNormal(), self.a4.shape, mindspore.float32))
        self.a5 = Parameter(initializer(XavierNormal(), self.a5.shape, mindspore.float32))
        self.a6 = Parameter(initializer(XavierNormal(), self.a6.shape, mindspore.float32))
        self.W1 = Parameter(initializer(XavierNormal(), self.a6.shape, mindspore.float32))
        self.W2 = Parameter(initializer(XavierNormal(), self.a6.shape, mindspore.float32))
        self.W3 = Parameter(initializer(XavierNormal(), self.a6.shape, mindspore.float32))

        self.hidden_drop_rate = hidden_drop
        self.hidden_drop = mindspore.nn.Dropout(self.hidden_drop_rate)
        self.leakyrelu = mindspore.nn.LeakyReLU(alpha=0.2)


        self.in_channels = 1
        self.out_channels = 5
        self.filt_h = 1
        self.filt_w = 1
        self.stride = 2
        self.max_arity = 4


        # self.bn0 = x2ms_nn.BatchNorm2d(self.in_channels)
        self.bn0 = mindspore.nn.BatchNorm2d(self.in_channels)

        self.inp_drop = mindspore.nn.Dropout(0.2)

        fc_length = (1-self.filt_h+1)*math.floor((emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        # self.bn2 = x2ms_nn.BatchNorm1d(fc_length)
        self.bn2 = mindspore.nn.BatchNorm1d(fc_length)
        # Projection network
        self.fc = mindspore.nn.Dense(fc_length, emb_dim)
        # self.device = x2ms_adapter.Device('cuda:0' if x2ms_adapter.is_cuda_available() else 'cpu')

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        # self.fc1 = x2ms_nn.Linear(emb_dim + self.max_arity + 1, fc1_length) # (306, 5)
        # self.fc2 = x2ms_nn.Linear(self.max_arity + 1, fc1_length) # (6, 5)
        self.fc1 = mindspore.nn.Dense(emb_dim + self.max_arity + 1, fc1_length)  # (306, 5)
        self.fc2 = mindspore.nn.Dense(self.max_arity + 1, fc1_length)  # (6, 5)


    def er_pos_emb(self, r_emb, e_emb):
        # return mindspore.ops.mm(mindspore.ops.cat((r_emb, e_emb), dim=1), self.W1)
        return mindspore.ops.mm(mindspore.ops.cat((r_emb, e_emb), axis=1), self.W1)


    def convolve(self, r, ei, pos):

        e = ei.view((-1, 1, 1, self.E.weight.shape[1]))
        x = e
        x = self.inp_drop(x)
        # one_hot_target = x2ms_adapter.to(x2ms_adapter.tensor_api.x2ms_float((pos == x2ms_adapter.arange(self.max_arity + 1).reshape(self.max_arity + 1))), self.device)
        # one_hot_target = (pos == mindspore.ops.arange(self.max_arity + 1).float().reshape(self.max_arity + 1))
        one_hot_target = x2ms_adapter.tensor_api.x2ms_float((pos == x2ms_adapter.arange(self.max_arity + 1).reshape(self.max_arity + 1)))
        poses = x2ms_adapter.tensor_api.repeat(one_hot_target, r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False
        k = self.fc2(poses)
        k = k.view( -1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view((e.shape[0]*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w))
        x = x.permute( 1, 0, 2, 3).contiguous()
        x = mindspore.ops.conv2d(x, k, stride=self.stride, groups=e.shape[0],pad_mode="pad")
        x = x.view(e.shape[0], 1, self.out_channels, 1-self.filt_h+1, -1)
        x = x.permute( 0, 3, 4, 1, 2).contiguous()
        x = mindspore.ops.sum(x, dim=3)
        x = x.permute( 0, 3, 1, 2).contiguous()
        x = x.view(e.shape[0], -1)
        x = self.fc(x)
        return x

    def construct(self, batch, ms, bs):
        r_idx = batch[:, 0]
        r = self.R(r_idx)

        if batch.shape[1] == 3:
            # print("batch_shape:3")
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view( -1, 1) + bs[:,0].view( -1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view( -1, 1) + bs[:,1].view( -1, 1)

            e12 = mindspore.ops.mm(mindspore.ops.cat((e1, e2, r), axis=1), self.W02)
            e21 = mindspore.ops.mm(mindspore.ops.cat((e2, e1, r), axis=1), self.W02)

            e1_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a2))) / mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a2)))
            e2_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a2))) / mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a2)))


            new_e1 = mindspore.ops.mm(e1, self.W2) + mindspore.ops.tanh(e12*e1_e2_att)
            new_e2 = mindspore.ops.mm(e2, self.W2) + mindspore.ops.tanh(e21*e2_e1_att)


            e = new_e1 * new_e2 * r

        elif batch.shape[1] == 4:
            # print("batch_shape:4")
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view( -1, 1) + bs[:,0].view( -1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view( -1, 1) + bs[:,1].view( -1, 1)
            e3 = self.convolve(r, self.E(batch[:, 3]), 2) * ms[:,2].view( -1, 1) + bs[:,2].view( -1, 1)
            e12 = mindspore.ops.mm(mindspore.ops.cat((e1, e2, r), axis=1), self.W03)
            e13 = mindspore.ops.mm(mindspore.ops.cat((e1, e3, r), axis=1), self.W03)
            e21 = mindspore.ops.mm(mindspore.ops.cat((e2, e1, r), axis=1), self.W03)
            e23 = mindspore.ops.mm(mindspore.ops.cat((e2, e3, r), axis=1), self.W03)
            e31 = mindspore.ops.mm(mindspore.ops.cat((e3, e1, r), axis=1), self.W03)
            e32 = mindspore.ops.mm(mindspore.ops.cat((e3, e2, r), axis=1), self.W03)

            e1_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a3))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a3))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a3))))
            e1_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a3))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a3))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a3))))
            e2_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a3))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a3))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a3))))
            e2_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a3))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a3))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a3))))
            e3_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a3))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a3))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a3))))
            e3_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a3))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a3))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a3))))


            new_e1 = mindspore.ops.mm(e1, self.W2) + mindspore.ops.tanh(e12*e1_e2_att + e13*e1_e3_att)
            new_e2 = mindspore.ops.mm(e2, self.W2) + mindspore.ops.tanh(e21*e2_e1_att + e23*e2_e3_att)
            new_e3 = mindspore.ops.mm(e3, self.W2) + mindspore.ops.tanh(e31*e3_e1_att + e32*e3_e2_att)


            e = new_e1 * new_e2 * new_e3 * r

        elif batch.shape[1] == 5:
            # print("batch_shape:5")
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view( -1, 1) + bs[:,0].view( -1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view( -1, 1) + bs[:,1].view( -1, 1)
            e3 = self.convolve(r, self.E(batch[:, 3]), 2) * ms[:,2].view( -1, 1) + bs[:,2].view( -1, 1)
            e4 = self.convolve(r, self.E(batch[:, 4]), 3) * ms[:,3].view( -1, 1) + bs[:,3].view( -1, 1)


            e12 = mindspore.ops.mm(mindspore.ops.cat((e1, e2, r), axis=1), self.W04)
            e13 = mindspore.ops.mm(mindspore.ops.cat((e1, e3, r), axis=1), self.W04)
            e14 = mindspore.ops.mm(mindspore.ops.cat((e1, e4, r), axis=1), self.W04)
            e21 = mindspore.ops.mm(mindspore.ops.cat((e2, e1, r), axis=1), self.W04)
            e23 = mindspore.ops.mm(mindspore.ops.cat((e2, e3, r), axis=1), self.W04)
            e24 = mindspore.ops.mm(mindspore.ops.cat((e2, e4, r), axis=1), self.W04)
            e31 = mindspore.ops.mm(mindspore.ops.cat((e3, e1, r), axis=1), self.W04)
            e32 = mindspore.ops.mm(mindspore.ops.cat((e3, e2, r), axis=1), self.W04)
            e34 = mindspore.ops.mm(mindspore.ops.cat((e3, e4, r), axis=1), self.W04)
            e41 = mindspore.ops.mm(mindspore.ops.cat((e4, e1, r), axis=1), self.W04)
            e42 = mindspore.ops.mm(mindspore.ops.cat((e4, e2, r), axis=1), self.W04)
            e43 = mindspore.ops.mm(mindspore.ops.cat((e4, e3, r), axis=1), self.W04)

            e1_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a4))))
            e1_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a4))))
            e1_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a4))))
            e2_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a4))))
            e2_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a4))))
            e2_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a4))))
            e3_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a4))))
            e3_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a4))))
            e3_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a4))))
            e4_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a4))))
            e4_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a4))))
            e4_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a4))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a4))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a4))))

            new_e1 = mindspore.ops.mm(e1, self.W2) + mindspore.ops.tanh(e12 * e1_e2_att + e13 * e1_e3_att + e14 * e1_e4_att)
            new_e2 = mindspore.ops.mm(e2, self.W2) + mindspore.ops.tanh(e21 * e2_e1_att + e23 * e2_e3_att + e24 * e2_e4_att)
            new_e3 = mindspore.ops.mm(e3, self.W2) + mindspore.ops.tanh(e31 * e3_e1_att + e32 * e3_e2_att + e34 * e3_e4_att)
            new_e4 = mindspore.ops.mm(e4, self.W2) + mindspore.ops.tanh(e41 * e4_e1_att + e42 * e4_e2_att + e43 * e4_e3_att)


            e = new_e1 * new_e2 * new_e3 * new_e4 * r


        elif batch.shape[1] == 6:
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view( -1, 1) + bs[:,0].view( -1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view( -1, 1) + bs[:,1].view( -1, 1)
            e3 = self.convolve(r, self.E(batch[:, 3]), 2) * ms[:,2].view( -1, 1) + bs[:,2].view( -1, 1)
            e4 = self.convolve(r, self.E(batch[:, 4]), 3) * ms[:,3].view( -1, 1) + bs[:,3].view( -1, 1)
            e5 = self.convolve(r, self.E(batch[:, 5]), 4) * ms[:,4].view( -1, 1) + bs[:,4].view( -1, 1)

            e12 = mindspore.ops.mm(mindspore.ops.cat((e1, e2, r), axis=1), self.W05)
            e13 = mindspore.ops.mm(mindspore.ops.cat((e1, e3, r), axis=1), self.W05)
            e14 = mindspore.ops.mm(mindspore.ops.cat((e1, e4, r), axis=1), self.W05)
            e15 = mindspore.ops.mm(mindspore.ops.cat((e1, e5, r), axis=1), self.W05)
            e21 = mindspore.ops.mm(mindspore.ops.cat((e2, e1, r), axis=1), self.W05)
            e23 = mindspore.ops.mm(mindspore.ops.cat((e2, e3, r), axis=1), self.W05)
            e24 = mindspore.ops.mm(mindspore.ops.cat((e2, e4, r), axis=1), self.W05)
            e25 = mindspore.ops.mm(mindspore.ops.cat((e2, e5, r), axis=1), self.W05)
            e31 = mindspore.ops.mm(mindspore.ops.cat((e3, e1, r), axis=1), self.W05)
            e32 = mindspore.ops.mm(mindspore.ops.cat((e3, e2, r), axis=1), self.W05)
            e34 = mindspore.ops.mm(mindspore.ops.cat((e3, e4, r), axis=1), self.W05)
            e35 = mindspore.ops.mm(mindspore.ops.cat((e3, e5, r), axis=1), self.W05)
            e41 = mindspore.ops.mm(mindspore.ops.cat((e4, e1, r), axis=1), self.W05)
            e42 = mindspore.ops.mm(mindspore.ops.cat((e4, e2, r), axis=1), self.W05)
            e43 = mindspore.ops.mm(mindspore.ops.cat((e4, e3, r), axis=1), self.W05)
            e45 = mindspore.ops.mm(mindspore.ops.cat((e4, e5, r), axis=1), self.W05)
            e51 = mindspore.ops.mm(mindspore.ops.cat((e5, e1, r), axis=1), self.W05)
            e52 = mindspore.ops.mm(mindspore.ops.cat((e5, e2, r), axis=1), self.W05)
            e53 = mindspore.ops.mm(mindspore.ops.cat((e5, e3, r), axis=1), self.W05)
            e54 = mindspore.ops.mm(mindspore.ops.cat((e5, e4, r), axis=1), self.W05)



            e1_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a5))))
            e1_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a5))))
            e1_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a5))))
            e1_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a5))))
            e2_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a5))))
            e2_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a5))))
            e2_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a5))))
            e2_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a5))))
            e3_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a5))))
            e3_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a5))))
            e3_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a5))))
            e3_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a5))))
            e4_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a5))))
            e4_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a5))))
            e4_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a5))))
            e4_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a5))))
            e5_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a5))))
            e5_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a5))))
            e5_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a5))))
            e5_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a5))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a5))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a5))))


            new_e1 = mindspore.ops.mm(e1, self.W2) + mindspore.ops.tanh(e12*e1_e2_att + e13*e1_e3_att + e14*e1_e4_att + e15*e1_e5_att)
            new_e2 = mindspore.ops.mm(e2, self.W2) + mindspore.ops.tanh(e21*e2_e1_att + e23*e2_e3_att + e24*e2_e4_att + e25*e2_e5_att)
            new_e3 = mindspore.ops.mm(e3, self.W2) + mindspore.ops.tanh(e31*e3_e1_att + e32*e3_e2_att + e34*e3_e4_att + e35*e3_e5_att)
            new_e4 = mindspore.ops.mm(e4, self.W2) + mindspore.ops.tanh(e41*e4_e1_att + e42*e4_e2_att + e43*e4_e3_att + e45*e4_e5_att)
            new_e5 = mindspore.ops.mm(e5, self.W2) + mindspore.ops.tanh(e51*e5_e1_att + e52*e5_e2_att + e53*e5_e3_att + e54*e5_e4_att)

            e = new_e1 * new_e2 * new_e3 * new_e4 * new_e5 * r

        elif batch.shape[1] == 7:
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view( -1, 1) + bs[:,0].view( -1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view( -1, 1) + bs[:,1].view( -1, 1)
            e3 = self.convolve(r, self.E(batch[:, 3]), 2) * ms[:,2].view( -1, 1) + bs[:,2].view( -1, 1)
            e4 = self.convolve(r, self.E(batch[:, 4]), 3) * ms[:,3].view( -1, 1) + bs[:,3].view( -1, 1)
            e5 = self.convolve(r, self.E(batch[:, 5]), 4) * ms[:,4].view( -1, 1) + bs[:,4].view( -1, 1)
            e6 = self.convolve(r, self.E(batch[:, 6]), 5) * ms[:,5].view( -1, 1) + bs[:,5].view( -1, 1)

            e12 = mindspore.ops.mm(mindspore.ops.cat((e1, e2, r), axis=1), self.W06)
            e13 = mindspore.ops.mm(mindspore.ops.cat((e1, e3, r), axis=1), self.W06)
            e14 = mindspore.ops.mm(mindspore.ops.cat((e1, e4, r), axis=1), self.W06)
            e15 = mindspore.ops.mm(mindspore.ops.cat((e1, e5, r), axis=1), self.W06)
            e16 = mindspore.ops.mm(mindspore.ops.cat((e1, e6, r), axis=1), self.W06)
            e21 = mindspore.ops.mm(mindspore.ops.cat((e2, e1, r), axis=1), self.W06)
            e23 = mindspore.ops.mm(mindspore.ops.cat((e2, e3, r), axis=1), self.W06)
            e24 = mindspore.ops.mm(mindspore.ops.cat((e2, e4, r), axis=1), self.W06)
            e25 = mindspore.ops.mm(mindspore.ops.cat((e2, e5, r), axis=1), self.W06)
            e26 = mindspore.ops.mm(mindspore.ops.cat((e2, e6, r), axis=1), self.W06)
            e31 = mindspore.ops.mm(mindspore.ops.cat((e3, e1, r), axis=1), self.W06)
            e32 = mindspore.ops.mm(mindspore.ops.cat((e3, e2, r), axis=1), self.W06)
            e34 = mindspore.ops.mm(mindspore.ops.cat((e3, e4, r), axis=1), self.W06)
            e35 = mindspore.ops.mm(mindspore.ops.cat((e3, e5, r), axis=1), self.W06)
            e36 = mindspore.ops.mm(mindspore.ops.cat((e3, e6, r), axis=1), self.W06)
            e41 = mindspore.ops.mm(mindspore.ops.cat((e4, e1, r), axis=1), self.W06)
            e42 = mindspore.ops.mm(mindspore.ops.cat((e4, e2, r), axis=1), self.W06)
            e43 = mindspore.ops.mm(mindspore.ops.cat((e4, e3, r), axis=1), self.W06)
            e45 = mindspore.ops.mm(mindspore.ops.cat((e4, e5, r), axis=1), self.W06)
            e46 = mindspore.ops.mm(mindspore.ops.cat((e4, e6, r), axis=1), self.W06)
            e51 = mindspore.ops.mm(mindspore.ops.cat((e5, e1, r), axis=1), self.W06)
            e52 = mindspore.ops.mm(mindspore.ops.cat((e5, e2, r), axis=1), self.W06)
            e53 = mindspore.ops.mm(mindspore.ops.cat((e5, e3, r), axis=1), self.W06)
            e54 = mindspore.ops.mm(mindspore.ops.cat((e5, e4, r), axis=1), self.W06)
            e56 = mindspore.ops.mm(mindspore.ops.cat((e5, e6, r), axis=1), self.W06)
            e61 = mindspore.ops.mm(mindspore.ops.cat((e6, e1, r), axis=1), self.W06)
            e62 = mindspore.ops.mm(mindspore.ops.cat((e6, e2, r), axis=1), self.W06)
            e63 = mindspore.ops.mm(mindspore.ops.cat((e6, e3, r), axis=1), self.W06)
            e64 = mindspore.ops.mm(mindspore.ops.cat((e6, e4, r), axis=1), self.W06)
            e65 = mindspore.ops.mm(mindspore.ops.cat((e6, e5, r), axis=1), self.W06)

            e1_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e16, self.a6))))
            e1_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e16, self.a6))))
            e1_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e16, self.a6))))
            e1_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e16, self.a6))))
            e1_e6_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e16, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e12, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e13, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e14, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e15, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e16, self.a6))))
            e2_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e26, self.a6))))
            e2_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e26, self.a6))))
            e2_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e26, self.a6))))
            e2_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e26, self.a6))))
            e2_e6_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e26, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e21, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e23, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e24, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e25, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e26, self.a6))))
            e3_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e36, self.a6))))
            e3_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e36, self.a6))))
            e3_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e36, self.a6))))
            e3_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e36, self.a6))))
            e3_e6_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e36, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e31, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e32, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e34, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e35, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e36, self.a6))))
            e4_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e46, self.a6))))
            e4_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e46, self.a6))))
            e4_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e46, self.a6))))
            e4_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e46, self.a6))))
            e4_e6_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e46, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e41, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e42, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e43, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e45, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e46, self.a6))))
            e5_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e56, self.a6))))
            e5_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e56, self.a6))))
            e5_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e56, self.a6))))
            e5_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e56, self.a6))))
            e5_e6_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e56, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e51, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e52, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e53, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e54, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e56, self.a6))))
            e6_e1_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e61, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e61, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e62, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e63, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e64, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e65, self.a6))))
            e6_e2_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e62, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e61, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e62, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e63, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e64, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e65, self.a6))))
            e6_e3_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e63, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e61, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e62, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e63, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e64, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e65, self.a6))))
            e6_e4_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e64, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e61, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e62, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e63, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e64, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e65, self.a6))))
            e6_e5_att = mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e65, self.a6))) / (mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e61, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e62, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e63, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e64, self.a6))) + mindspore.ops.exp(self.leakyrelu(mindspore.ops.mm(e65, self.a6))))




            new_e1 = mindspore.ops.mm(e1, self.W2) + mindspore.ops.tanh(e12*e1_e2_att + e13*e1_e3_att + e14*e1_e4_att + e15*e1_e5_att + e16*e1_e6_att)
            new_e2 = mindspore.ops.mm(e2, self.W2) + mindspore.ops.tanh(e21*e2_e1_att + e23*e2_e3_att + e24*e2_e4_att + e25*e2_e5_att + e26*e2_e6_att)
            new_e3 = mindspore.ops.mm(e3, self.W2) + mindspore.ops.tanh(e31*e3_e1_att + e32*e3_e2_att + e34*e3_e4_att + e35*e3_e5_att + e36*e3_e6_att)
            new_e4 = mindspore.ops.mm(e4, self.W2) + mindspore.ops.tanh(e41*e4_e1_att + e42*e4_e2_att + e43*e4_e3_att + e45*e4_e5_att + e46*e4_e6_att)
            new_e5 = mindspore.ops.mm(e5, self.W2) + mindspore.ops.tanh(e51*e5_e1_att + e52*e5_e2_att + e53*e5_e3_att + e54*e5_e4_att + e56*e5_e6_att)
            new_e6 = mindspore.ops.mm(e6, self.W2) + mindspore.ops.tanh(e61*e6_e1_att + e62*e6_e2_att + e63*e6_e3_att + e64*e6_e4_att + e65*e6_e5_att)

            e = new_e1 * new_e2 * new_e3 * new_e4 * new_e5 * new_e6 * r




        x = e
        x = self.hidden_drop(x)
        return mindspore.ops.sum(x, dim=1)

