import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def R2(preds, targets):
    """
    Coefficient of Determination (R²).
    Note: R² can be negative if predictions are worse than the mean baseline.
    R² = 1 - (SS_res / SS_tot)
    """
    pred_mean = torch.mean(preds, dim=0, keepdim=True)
    target_mean = torch.mean(targets, dim=0, keepdim=True)
    SS_res = torch.sum((targets - preds)**2, dim=0)
    SS_tot = torch.sum((targets - target_mean)**2, dim=0)
    r2_score = 1 - (SS_res / (SS_tot + 1e-8))
    return torch.nan_to_num(r2_score).item()


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
    
    def fit(self, dataset, steps, batch_size=16, lr=0.01, earlyStop=False):
        train_dataset = TensorDataset(dataset['train_input'],dataset['train_label'])
        test_dataset = TensorDataset(dataset['test_input'], dataset['test_label'])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = "cpu"

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        rmse_history = []
        R2_history = []
        for t in range(steps):
            self.train()
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.eval()
            with torch.no_grad():
                test_pred = self(dataset['test_input'].to(device))
                rmse_value = torch.sqrt(loss_fn(test_pred, dataset['test_label'].to(device))).item()
                rmse_history.append(rmse_value)
                R2_value = R2(test_pred, dataset['test_label'])
                R2_history.append(R2_value)
                print(f"Epoch {t+1}/{steps}, RMSE: {rmse_value:.4f}, R2: {R2_value:.4f} ", end='\r')
                if earlyStop and R2_value > 0.99:
                    print(f"\nEarly stopping at epoch {t+1} with R2: {R2_value:.4f}")
                    break
        
        return {'rmse_history': rmse_history, 'R2_history': R2_history}
    
# model = MLPKAN(input_size=1, hidden_sizes=[3], output_size=1, subnetworkshape=[2,2])
# # model.initialize(scale=5.0)
# # x = torch.randn(5, 2)
# # .expand(-1, 2)
# x = torch.linspace(-1, 1, steps=5).unsqueeze(1)
# print(x)
# output = model(x)
# print(output)
        
