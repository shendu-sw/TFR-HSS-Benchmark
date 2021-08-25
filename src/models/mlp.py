import torch
import torch.nn as nn


__all__ = ["MLP"]


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        layer = nn.GELU()
    elif act == "sigmoid":
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer


class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, act="gelu", bias=True, hidden_layer=[512, 512, 512]
    ):
        super(MLP, self).__init__()
        self.hidden_layer = hidden_layer
        net = []
        net.append(nn.Linear(input_dim, hidden_layer[0]))
        if act is not None and act.lower() != "none":
            net.append(act_layer(act))

        if len(hidden_layer) > 1:
            for i in range(1, len(hidden_layer)):
                net.append(nn.Linear(hidden_layer[i - 1], hidden_layer[i], bias))
                if act is not None and act.lower() != "none":
                    net.append(act_layer(act))

        net.append(nn.Linear(hidden_layer[-1], output_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
