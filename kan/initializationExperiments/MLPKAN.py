import torch
from torch import nn

class subnetwork(nn.Module):
    def __init__(self, subnetworkshape = [2,2]):
        super(subnetwork, self).__init__()
        self.subnetworkshape = subnetworkshape
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1, subnetworkshape[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(subnetworkshape)-1):
            self.layers.append(nn.Linear(subnetworkshape[i], subnetworkshape[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(subnetworkshape[-1], 1))

    def forward(self, x):
        self.preacts = x.squeeze()
        for layer in self.layers:
            x = layer(x)
        self.postacts = x.squeeze()
        return x

        

class MLPKAN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[3], output_size=1, subnetworkshape = [2,2]):
        super(MLPKAN, self).__init__()
        self.subnetworkshape = subnetworkshape
        self.layers = nn.ModuleList()

        layerSizes = [input_size] + hidden_sizes + [output_size]
        self.layerSizes = layerSizes

        for i in range(len(layerSizes)-1):
            subnetworksi = nn.ModuleDict()
            for j in range(layerSizes[i]):
                for k in range(layerSizes[i+1]):
                    subnet = subnetwork(subnetworkshape)
                    subnetworksi[f'subnet_{j}_{k}'] = subnet
            self.layers.append(subnetworksi)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        
        for i in range(len(self.layerSizes)-1):
            subnetworksi = self.layers[i]
            out = torch.zeros(x.shape[0], self.layerSizes[i+1], device=device, dtype=x.dtype)
            for j in range(self.layerSizes[i]):
                for k in range(self.layerSizes[i+1]):
                    subnet = subnetworksi[f'subnet_{j}_{k}']
                    x_j = x[:, j:j+1]
                    subnet_out = subnet(x_j)
                    out[:, k] += subnet_out.squeeze(-1)
            x = out
        return x
    
# model = MLPKAN(input_size=1, hidden_sizes=[3], output_size=1, subnetworkshape=[2,2])
# # model.initialize(scale=5.0)
# # x = torch.randn(5, 2)
# # .expand(-1, 2)
# x = torch.linspace(-1, 1, steps=5).unsqueeze(1)
# print(x)
# output = model(x)
# print(output)
        
