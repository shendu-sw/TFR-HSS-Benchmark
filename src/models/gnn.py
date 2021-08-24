import torch
from torch.nn import Sequential as Seq
from torch.nn import Linear as Lin
import torch_geometric as tg
from .util.gcn_lib.dense import BasicConv, GraphConv2d, PlainDynBlock2d, ResDynBlock2d, DenseDynBlock2d, DenseDilatedKnnGraph
from .util.gcn_lib.sparse import MultiSeq, MLP, GraphConv, PlainDynBlock, ResDynBlock, DenseDynBlock, DilatedKnnGraph


__all__ = ["DenseDeepGCN"]


class DenseDeepGCN(torch.nn.Module):
    def __init__(
        self,
        input_dim=None,
        output_dim=None,
        dropout=0.8,
        in_channels=2+1,
        k=5,
        n_classes=1,
        block="dense",
        conv="edge",
        act="relu",
        norm="batch",
        bias=True,
        n_filters=64,
        n_blocks=3,
        epsilon=0.8,
        stochastic=False,
        dim=2,
    ):
        super(DenseDeepGCN, self).__init__()
        self.dim = dim
        self.n_classes = n_classes #
        self.k = k #
        self.in_channels = in_channels #
        self.dropout = dropout #
        self.block = block
        self.conv = conv #
        self.act = act #
        self.norm = norm #
        self.bias = bias #
        self.channels = n_filters #
        self.n_blocks = n_blocks #
        self.epsilon = epsilon #
        self.stochastic = stochastic #

        c_growth = self.channels

        #print(self.dropout)
        self.knn = DenseDilatedKnnGraph(k, 1, self.stochastic, self.epsilon)
        self.head = GraphConv2d(self.in_channels, self.channels, conv, act, norm, bias)

        if self.block.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock2d(self.channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(self.channels + c_growth * (self.n_blocks - 1))
        elif self.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(self.channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (self.channels + self.channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            stochastic = False

            self.backbone = Seq(*[PlainDynBlock2d(self.channels, k, 1, conv, act, norm,
                                                  bias, stochastic, epsilon)
                                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(self.channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = BasicConv([fusion_dims, 1024], act, norm, bias)
        self.prediction = Seq(*[BasicConv([fusion_dims+1024, 512], act, norm, bias),
                                BasicConv([512, 256], act, norm, bias),
                                torch.nn.Dropout(p=self.dropout),
                                BasicConv([256, self.n_classes], None, None, bias)])

    def forward(self, inputs):
        feats = [self.head(inputs, self.knn(inputs[:, 0:self.dim]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)

        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        return self.prediction(torch.cat((fusion, feats), dim=1)).squeeze(-1)


class SparseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(SparseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels

        self.n_blocks = opt.n_blocks

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block.lower() == 'res':
            self.backbone = MultiSeq(*[ResDynBlock(channels, k, 1+i, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                                       for i in range(self.n_blocks-1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif opt.block.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels+c_growth*i, c_growth, k, 1+i,
                                                     conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                                       for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            # Use PlainGCN without skip connection and dilated convolution.
            stochastic = False
            self.backbone = MultiSeq(
                *[PlainDynBlock(channels, k, 1, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = MLP([fusion_dims, 1024], act, norm, bias)
        self.prediction = MultiSeq(*[MLP([fusion_dims+1024, 512], act, norm, bias),
                                     MLP([512, 256], act, norm, bias, drop=opt.dropout),
                                     MLP([256, opt.n_classes], None, None, bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        corr, color, batch = data.pos, data.x, data.batch
        x = torch.cat((corr, color), dim=1)
        feats = [self.head(x, self.knn(x[:, 0:3], batch))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], batch)[0])
        feats = torch.cat(feats, dim=1)

        fusion = tg.utils.scatter_('max', self.fusion_block(feats), batch)
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[0]//fusion.shape[0], dim=0)
        return self.prediction(torch.cat((fusion, feats), dim=1))



if __name__ == "__main__":
    
    import random, numpy as np, argparse
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    N = 1024
    device = 'cuda'

    parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')
    parser.add_argument('--in_channels', default=9, type=int, help='input channels (default:9)')
    parser.add_argument('--n_classes', default=13, type=int, help='num of segmentation classes (default:13)')
    parser.add_argument('--k', default=4, type=int, help='neighbor num (default:16)')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
    parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
    parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
    parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=False, type=bool, help='stochastic for gcn, True or False')
    args = parser.parse_args()

    pos = torch.rand((batch_size, N, 2), dtype=torch.float).to(device)
    x = torch.rand((batch_size, N, 6), dtype=torch.float).to(device)

    inputs = torch.cat((pos, x), 2).transpose(1, 2).unsqueeze(-1)
    print(inputs.size())

    # net = DGCNNSegDense().to(device)
    net = DenseDeepGCN(in_channels=2+6).to(device)
    # net = SparseDeepGCN(args).to(device)
    print(net)
    out = net(inputs)
    print(out.shape)
    