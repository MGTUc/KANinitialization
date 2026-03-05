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
    def __init__(self, output_size=1, subnetworkshape = [2,2]):
        super(subnetwork, self).__init__()
        self.subnetworkshape = subnetworkshape
        
        # Network that takes 1 input and produces output_size outputs
        layers = [nn.Linear(1, subnetworkshape[0]), nn.ReLU()]
        for i in range(len(subnetworkshape)-1):
            layers.append(nn.Linear(subnetworkshape[i], subnetworkshape[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(subnetworkshape[-1], output_size))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

        

class MLPKAN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[3], output_size=1, subnetworkshape = [2,2]):
        super(MLPKAN, self).__init__()
        self.subnetworkshape = subnetworkshape
        self.layers = nn.ModuleList()

        layerSizes = [input_size] + hidden_sizes + [output_size]
        self.layerSizes = layerSizes

        for i in range(len(layerSizes)-1):
            # For each input dimension, create a subnet that outputs to all next layer dimensions
            subnetworks_layer = nn.ModuleList()
            for j in range(layerSizes[i]):
                subnet = subnetwork(output_size=layerSizes[i+1], subnetworkshape=subnetworkshape)
                subnetworks_layer.append(subnet)
            self.layers.append(subnetworks_layer)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        
        for i in range(len(self.layerSizes)-1):
            subnetworks_layer = self.layers[i]
            input_size = self.layerSizes[i]
            
            out = None
            # For each input dimension, apply its subnet (which outputs to all dimensions)
            for j in range(input_size):
                subnet = subnetworks_layer[j]
                x_j = x[:, j:j+1]  # (batch_size, 1)
                subnet_out = subnet(x_j)  # (batch_size, output_size)
                
                if out is None:
                    out = subnet_out
                else:
                    out = out + subnet_out
            x = out
        return x
    
    
    def fit(self, dataset, steps, batch_size=16, lr=1, earlyStop=False):
        train_dataset = TensorDataset(dataset['train_input'],dataset['train_label'])
        test_dataset = TensorDataset(dataset['test_input'], dataset['test_label'])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20, line_search_fn='strong_wolfe')


        X_train = dataset['train_input'].to(device)
        y_train = dataset['train_label'].to(device)

        rmse_history = []
        R2_history = []
        test_input = dataset['test_input'].to(device)
        test_label = dataset['test_label'].to(device)
        
        for t in range(steps):
            self.train()
            def closure():
                optimizer.zero_grad()
                pred = self.forward(X_train)
                loss = loss_fn(pred, y_train)
                loss.backward()
                # Clip gradients to prevent exploding gradients
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                return loss
            
            optimizer.step(closure)
            
            # Check for NaN and break if detected
            if any(torch.isnan(p).any() for p in self.parameters()):
                print(f"\nNaN detected at epoch {t+1}, stopping training")
                break
            
            self.eval()
            with torch.no_grad():
                test_pred = self(test_input)
                rmse_value = torch.sqrt(loss_fn(test_pred, test_label)).item()
                rmse_history.append(rmse_value)
                R2_value = R2(test_pred, test_label)
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
        
