from kan.feynman import get_feynman_dataset
from kan import create_dataset
import pandas as pd

for i in range(1,121):
    print(i)
    symbols, _, func, ranges, name = get_feynman_dataset(i)
    dataset = create_dataset(func, n_var=len(symbols), normalize_input=True, seed=42, train_num=1000, test_num=1000, ranges=ranges)

    X_train = pd.DataFrame(dataset['train_input'])
    y_train = pd.DataFrame(dataset['train_label'])
    X_test = pd.DataFrame(dataset['test_input'])
    y_test = pd.DataFrame(dataset['test_label'])

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f'feynmanDataset/train/{name}_train.csv', index=False, header=False)
    test_df.to_csv(f'feynmanDataset/test/{name}_test.csv', index=False, header=False)


