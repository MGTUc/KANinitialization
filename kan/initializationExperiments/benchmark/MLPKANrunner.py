from kan import KAN
from kan.initializationExperiments.MLPKAN import MLPKAN
from kan.feynman import get_feynman_dataset
import torch
import numpy as np
import sympy as sp
import random
import pandas as pd
from pathlib import Path
import time

def r2(preds,targets):
    pred_mean = torch.mean(preds, dim=[0], keepdim=True)
    target_mean =torch.mean(targets, dim=[0], keepdim=True)
    numerator = torch.sum((preds - pred_mean)*(targets-target_mean), dim=0)**2
    denominator = torch.sum((preds - pred_mean)**2, dim=0)*torch.sum((targets - target_mean)**2, dim=0)
    r2 = numerator/(denominator+1e-4)
    r2 = torch.nan_to_num(r2)
    return r2

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
    return torch.nan_to_num(r2_score)

def main():
    # Set random seeds for reproducibility
    seed = 500
    device = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    easy_set = {
    "I.12.1", "I.12.4", "I.12.5", "I.14.3", "I.14.4", 
    "I.18.12", "I.18.14", "I.25.13", "I.26.2", "I.27.6", 
    "I.30.5", "I.43.16", "I.47.23", "II.2.42", "II.3.24", 
    "II.4.23", "II.8.31", "II.10.9", "II.11.17", "II.15.4", 
    "II.15.5", "II.27.16", "II.27.18", "II.34.11", "II.34.29b", 
    "II.38.3", "II.38.14", "III.7.38", "III.12.43", "III.15.27"
    }

    # Load the Feynman dataset
    folder_path = Path('./feynmanDataset')

    results = []
    for file_path in folder_path.glob('train/*.csv'):
        if file_path.stem.replace('_train', '') not in easy_set:
            continue
        print(f"Processing {file_path.name}")
        try:
            function_name = file_path.stem.replace('_train', '')

            train_df = pd.read_csv(file_path, header=None)
            test_df = pd.read_csv(folder_path / 'test' / f'{function_name}_test.csv', header=None)
            X_train = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32)
            y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
            X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
            y_test = torch.tensor(test_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
            dataset = {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}
            # # Initialize KAN and fit the model

            MLPKANmodel = MLPKAN(input_size=X_train.size()[1], hidden_sizes=[3], output_size=1, subnetworkshape=[4,4,4])
            

            t0 = time.perf_counter()
            MLPKANmodel.fit(dataset=dataset, steps=50, batch_size=16, lr=0.001, earlyStop=True);
            t_MLPKAN = time.perf_counter() - t0

            t1 = time.perf_counter()
            kan =KAN(width=[X_train.size()[1],3,1], grid=3, k=3, seed=seed, device=device, auto_save=False)
            kan.set_splines_MLPKAN(dataset['test_input'], MLPKANmodel)
            t_conversion = time.perf_counter() - t1
            t5 = time.perf_counter()
            kan.fit(dataset, opt="LBFGS", steps=50, lamb=0.001, earlyStop=True);
            t_kanTraining = time.perf_counter() - t5

            y_pred_kan = kan(dataset['test_input'])
            R2_score_kan = R2(y_pred_kan, dataset['test_label']).item()
            # r_score_kan = torch.sqrt(r2(y_pred_kan, dataset['test_label'])).item()
            print(f"R2 Score KAN: {R2_score_kan}")
            # print(f"pearson correlation coefficient KAN: {r_score_kan}")

            kan = kan.prune()

            t2 = time.perf_counter()
            kan.auto_symbolic(weight_simple=0.8);
            t_symbolic = time.perf_counter() - t2

            t3 = time.perf_counter()
            kan.fit(dataset, opt="Adam", lr=0.01, steps=500, lamb=0.001, update_grid=False, singularity_avoiding=True, earlyStop=True);
            t_adam = time.perf_counter() - t3

            extracted_function, symbols = kan.symbolic_formula()
            expr = extracted_function[0]
            free_syms = list(expr.free_symbols)
            f = sp.lambdify(free_syms, expr, "numpy")
            X_test_np = dataset['test_input'].detach().cpu().numpy()
            symbol_to_idx = {sp.Symbol(f'x_{i+1}'): i for i in range(X_test_np.shape[1])}
            f_args = [X_test_np[:, symbol_to_idx[s]] for s in free_syms]
            y_pred_np = f(*f_args) if free_syms else np.full(len(X_test_np), float(expr))
            
            y_pred_formula = torch.tensor(y_pred_np, dtype=dataset["test_label"].dtype, device=dataset["test_label"].device).reshape(-1, 1)
            R2_score_formula = R2(y_pred_formula, dataset['test_label']).item()
            # r_score_formula = torch.sqrt(r2(y_pred_formula, dataset['test_label'])).item()
            print(f"R2 Score: {R2_score_formula}")
            # print(f"pearson correlation coefficient: {r_score_formula}")

            results.append([function_name, R2_score_formula, R2_score_kan, t_MLPKAN,t_conversion, t_kanTraining, t_symbolic, t_adam])
            break

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            results.append([function_name, None, None, None, None, None, None, None])

    results_df = pd.DataFrame(results, columns=['Function', 'R2 Score', 'R2 Score KAN', 'MLPKAN time', 'Conversion time', 'KAN training time', 'Symbolic time', 'Adam time'])
    results_df.to_csv('kan_feynman_results_MLPKAN.csv', index=False)


if __name__ == "__main__":
    main()
