import math

import numpy as np
import jittor as jt
import jittor.nn as nn


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n, k1, k2, _ = weight.shape
    std = math.sqrt(2. / (n * k1 * k2 * branches))
    nn.init.gauss_(weight, mean=0, std=std)  # Jittor 的高斯初始化
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)



def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def execute(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = jt.array(A.astype(np.float32))
        self.PA = self.PA * 1e-6  # 初始化为 1e-6

        # self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        # nn.init.constant_(self.PA, 1e-6)
        self.A = jt.array(A.astype(np.float32))
        # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # self.down = nn.Identity()
            self.down = self.tmp_func
            # self.down = lambda x: x


        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def tmp_func(self, x):
        return x

    def execute(self, x):
        N, C, T, V = x.shape
        A = self.A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).transpose(0, 3, 1, 2).reshape(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).reshape(N, self.inter_c * T, V)
            A1 = self.soft(jt.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](jt.matmul(A2, A1).reshape(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

class ZeroModule(nn.Module):
    def execute(self, x):
        return 0  # 返回和输入同形状的 0

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            # self.residual = ZeroModule()
            self.residual = self.zero_func

        elif (in_channels == out_channels) and (stride == 1):
            # self.residual = lambda x: x
            # self.residual = nn.Identity()
            self.residual = self.tmp_func
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
    def tmp_func(self, x):
        return x

    def zero_func(self, x):
        return 0
    def execute(self, x):
        # print(self.tcn1(self.gcn1(x)).shape)
        # print(self.residual(x).shape)
        x1 = self.tcn1(self.gcn1(x))
        x2 = self.residual(x)

        # x = self.tcn1(self.gcn1(x)) + self.residual(x)
        x = x1 + x2
        return self.relu(x)
        # x = self.tcn1(self.gcn1(x)) + self.residual(x)
        # return self.relu(x)
# class fusion(nn.Module):
#     def __init__(self,in_channels,out_channel):
#         super(fusion,self).__init__()
#         self.conv_out=nn.Conv2d(in_channels,out_channel,1, bias=False)
#     def forward(self,x_p,x_m):
#         x=torch.cat((x_p,x_m),1)
#
#         x=self.conv_out(x)
#         return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveAvgPool2d(1)#nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
    

class fusion(nn.Module):
    def __init__(self,T_in_channels):
        super(fusion,self).__init__()
        self.att_T_p=ChannelAttention(T_in_channels)
        self.att_N_p=ChannelAttention(16)
        self.att_T_m = ChannelAttention(T_in_channels)
        self.att_N_m = ChannelAttention(16)
        
    def execute(self,x_p,x_m):
        #B*C*T*N 自适应融合，T和N各自轴上
        #情感特征的参数化，或者平均方式的参数化。
        B,C,T,N=x_p.size()
        x_p_T=x_p.permute(0,2,1,3)
        x_p_N=x_p.permute(0,3,2,1)
        x_m_T = x_m.permute(0, 2, 1, 3)
        x_m_N = x_m.permute(0, 3, 2, 1)


        att_N_p_map = (self.att_N_p(x_p_N)).permute(0, 3, 2, 1)
        x_p_mid = (x_p * att_N_p_map).permute(0, 2, 1, 3)
        att_T_p_map=(self.att_T_p(x_p_mid)).permute(0,2,1,3)

        att_N_m_map = (self.att_N_m(x_m_N)).permute(0, 3, 2, 1)
        x_m_mid = (x_m * att_N_m_map).permute(0, 2, 1, 3)
        att_T_m_map = (self.att_T_m(x_m_mid)).permute(0, 2, 1, 3)



        x_p=x_p+x_m*att_T_m_map
        x_m=x_m+x_p*att_T_p_map

        # x_p=x_p+x_m*att_T_m_map*att_N_m_map
        # x_m=x_m+x_p*att_T_p_map*att_N_p_map
        # x_p = x_p + x_m * (att_T_m_map+att_N_m_map)
        # x_m = x_m + x_p * (att_T_p_map+att_T_p_map)

        return x_p,x_m

class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_constraints=31, graph=None, graph_args=dict(), in_channels_p=3,in_channels_m=8):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(in_channels_m * num_point)

        self.l1_p = TCN_GCN_unit(in_channels_p, 64, A, residual=False)
        self.l1_m = TCN_GCN_unit(in_channels_m, 64, A, residual=False)

        self.l2_p = TCN_GCN_unit(64, 64, A)
        self.l3_p = TCN_GCN_unit(64, 64, A)
        self.l4_p = TCN_GCN_unit(64, 64, A)
        self.l5_p = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_p = TCN_GCN_unit(128, 128, A)
        self.l7_p = TCN_GCN_unit(128, 128, A)
        self.l8_p = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_p = TCN_GCN_unit(256, 256, A)
        self.l10_p = TCN_GCN_unit(256, 256, A)

        self.l2_m = TCN_GCN_unit(64, 64, A)
        self.l3_m = TCN_GCN_unit(64, 64, A)
        self.l4_m = TCN_GCN_unit(64, 64, A)
        self.l5_m = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6_m = TCN_GCN_unit(128, 128, A)
        self.l7_m = TCN_GCN_unit(128, 128, A)
        self.l8_m = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9_m = TCN_GCN_unit(256, 256, A)
        self.l10_m = TCN_GCN_unit(256, 256, A)

        self.fusion1 = fusion(48)
        self.fusion2 = fusion(24)
        self.fusion3 = fusion(12)

        self.fc1_classifier_p = nn.Linear(256, num_class)
        self.fc1_classifier_m = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, num_constraints*48)

        # nn.init.normal_(self.fc1_classifier_m.weight, 0, math.sqrt(2. / num_class))
        # nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        # nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints*48)))

        nn.init.gauss_(self.fc1_classifier_m.weight, 0, math.sqrt(2. / num_class))
        nn.init.gauss_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        nn.init.gauss_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints * 48)))

        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)

    def execute(self, x_p, x_m):
        N, C_p, T, V, M = x_p.shape
        N, C_m, T, V, M = x_m.shape

        x_p = x_p.permute(0, 4, 3, 1, 2).reshape(N, M * V * C_p, T)
        x_m = x_m.permute(0, 4, 3, 1, 2).reshape(N, M * V * C_m, T)


        x_p = self.data_bn_p(x_p)
        x_m = self.data_bn_m(x_m)

        x_p = x_p.reshape(N, M, V, C_p, T).permute(0, 1, 3, 4, 2).reshape(N * M, C_p, T, V)
        x_m = x_m.reshape(N, M, V, C_m, T).permute(0, 1, 3, 4, 2).reshape(N * M, C_m, T, V)

        x_p = self.l1_p(x_p)
        x_m = self.l1_m(x_m)
        x_p = self.l2_p(x_p)
        x_m = self.l2_m(x_m)
        x_p = self.l3_p(x_p)
        x_m = self.l3_m(x_m)
        x_p = self.l4_p(x_p)
        x_m = self.l4_m(x_m)

        x_p,x_m=self.fusion1(x_p,x_m)

        x_p = self.l5_p(x_p)
        x_m = self.l5_m(x_m)
        x_p = self.l6_p(x_p)
        x_m = self.l6_m(x_m)
        x_p = self.l7_p(x_p)
        x_m = self.l7_m(x_m)

        x_p,x_m=self.fusion2(x_p,x_m)

        x_p = self.l8_p(x_p)
        x_m = self.l8_m(x_m)
        x_p = self.l9_p(x_p)
        x_m = self.l9_m(x_m)
        x_p = self.l10_p(x_p)
        x_m = self.l10_m(x_m)

        x_p,x_m=self.fusion3(x_p,x_m)


        # N*M,C,T,V
        c_new_m = x_m.shape[1]
        x_m = x_m.reshape(N, M, c_new_m, -1)
        x_m = x_m.mean((3, 1), keepdims=False)

        c_new_p = x_p.shape[1]
        x_p = x_p.reshape(N, M, c_new_p, -1)
        x_p = x_p.mean((3, 1), keepdims=False)

        # x_cat=torch.cat((x_m,x_p),1)

        return self.fc1_classifier_p(x_p),self.fc2_aff(x_p),self.fc1_classifier_m(x_m)


class Model_Single(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_person=1, graph=None, graph_args=dict(), in_channels_p=3, in_channels_m=8):
        super(Model_Single, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(num_person * (in_channels_m+in_channels_p) * num_point)
        self.l1 = TCN_GCN_unit((in_channels_m+in_channels_p), 64, A, residual=False)

        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc1_classifier = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, 31*48)

        nn.init.gauss_(self.fc1_classifier.weight, 0, math.sqrt(2. / num_class))
        nn.init.gauss_(self.fc2_aff.weight, 0, math.sqrt(2. / (31*48)))
        bn_init(self.data_bn, 1)

    def forward(self, x_p,x_m):
        N, C_p, T, V, M = x_p.size()
        N, C_m, T, V, M = x_m.size()

        x = jt.concat((x_p, x_m), 1)

        x = x.permute(0, 4, 3, 1, 2).reshape(N, M * V * (C_m+C_p), T)

        x = self.data_bn(x)

        x = x.reshape(N, M, V, (C_m+C_p), T).permute(0, 1, 3, 4, 2).reshape(N * M, (C_p+C_m), T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        c_new_x = x.shape[1]
        x = x.reshape(N, M, c_new_x, -1)
        x = x.mean(dim=[3, 1], keepdim=False)

        return self.fc1_classifier(x), self.fc2_aff(x)

