import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call, vmap

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
        
        # Network that takes 1 input and produces output_size outputs
        layers = [nn.Linear(1, subnetworkshape[0]), nn.ReLU()]
        for i in range(len(subnetworkshape)-1):
            layers.append(nn.Linear(subnetworkshape[i], subnetworkshape[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(subnetworkshape[-1], 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class MLPKANlayer(nn.Module):
    def __init__(self, input_size, output_size, subnetworkshape=[2, 2]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_subnets = input_size * output_size

        self.subnetworks = nn.ModuleList(
            [subnetwork(subnetworkshape) for _ in range(self.n_subnets)]
        )

    def forward(self, x):
        # x: [B, I]
        B = x.shape[0]

        # Arrange inputs so each subnet gets its own [B, 1] tensor
        # [B, I] -> [B, I, O] -> [B, I*O, 1] -> [I*O, B, 1]
        x_pairs = (
            x.unsqueeze(2)
             .expand(B, self.input_size, self.output_size)
             .reshape(B, self.n_subnets, 1)
             .permute(1, 0, 2)
        )

        # Stack real parameters from ModuleList so gradients flow back to each subnet.
        param_names = [name for name, _ in self.subnetworks[0].named_parameters()]
        buffer_names = [name for name, _ in self.subnetworks[0].named_buffers()]

        params = {
            name: torch.stack([module.get_parameter(name) for module in self.subnetworks], dim=0)
            for name in param_names
        }
        buffers = {
            name: torch.stack([module.get_buffer(name) for module in self.subnetworks], dim=0)
            for name in buffer_names
        }

        def fmodel(p, b, x_one):
            # x_one: [B, 1] for one subnet
            return functional_call(self.subnetworks[0], (p, b), (x_one,)).squeeze(-1)  # [B]

        # Vectorize across subnetworks (no Python for-loop over subnets)
        y = vmap(fmodel, in_dims=(0, 0, 0))(params, buffers, x_pairs)  # [I*O, B]

        # Reshape to [B, I, O], sum over I -> [B, O]
        y = y.permute(1, 0).reshape(B, self.input_size, self.output_size).sum(dim=1)
        return y

        

class MLPKAN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[3], output_size=1, subnetworkshape = [2,2]):
        super(MLPKAN, self).__init__()
        self.subnetworkshape = subnetworkshape

        layerSizes = [input_size] + hidden_sizes + [output_size]
        self.layerSizes = layerSizes

        layers = []

        for i in range(len(layerSizes)-1):
            mlpkan_layer = MLPKANlayer(input_size=layerSizes[i], output_size=layerSizes[i+1], subnetworkshape=subnetworkshape)
            layers.append(mlpkan_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def fit(self, dataset, steps, batch_size=16, lr=0.001, earlyStop=False):
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
        
