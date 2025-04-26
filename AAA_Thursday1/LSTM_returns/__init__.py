import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# optuna scrap goes in the ipynb, okay?

def vectorize_train_data(splitted_df, target="Return_Target", features=["Open", "High", "Low", "Close", "Volume"], episode_length=30):
    # for each row with row["SPLIT"] = "train"
    # yield a train_X and a train_y array, where train_y is just "Return"
    # and train_X is the rest of the columns, but without "SPLIT" and "Return"
    
    train_X = []
    train_y = []
    train_index = []
    test_X = []
    test_y = []
    test_index = []

    # Create an array of dataframes with progressively dropped rows
    dfs = [splitted_df.iloc[i:len(splitted_df) - episode_length + i] for i in range(episode_length)][::-1]

    def process_row(i):
        current_row = splitted_df.iloc[i]
        current_row_features = np.array([
            df.iloc[i][features] for df in dfs
        ])
        if current_row["SPLIT"] == "train":
            train_X.append(current_row_features)
            train_y.append(current_row[target])
            train_index.append(splitted_df.index[i])
        elif current_row["SPLIT"] == "test":
            test_X.append(current_row_features)
            test_y.append(current_row[target])
            test_index.append(splitted_df.index[i])

    list(map(process_row, range(len(splitted_df))))
    
    return { 'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y,
            'train_index': train_index, 'test_index': test_index }

def vectorize_train_data(splitted_df, target="Return_Target", features=["Open", "High", "Low", "Close", "Volume"], episode_length=30):
    features_df = splitted_df[features]
    target = splitted_df[target]

    i_s = np.arange(episode_length-1, len(splitted_df))
    train_i_s = i_s[splitted_df.iloc[i_s]["SPLIT"] == "train"]
    test_i_s = i_s[splitted_df.iloc[i_s]["SPLIT"] == "test"]

    train_X = np.lib.stride_tricks.sliding_window_view(features_df.values, (episode_length, features_df.shape[1]))[train_i_s - episode_length + 1][:, 0, :, :]
    test_X = np.lib.stride_tricks.sliding_window_view(features_df.values, (episode_length, features_df.shape[1]))[test_i_s - episode_length + 1][:, 0, :, :]
    train_y = target.values[train_i_s]
    test_y = target.values[test_i_s]
    train_index = splitted_df.index[train_i_s]
    test_index = splitted_df.index[test_i_s]
    
    return { 'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y,
            'train_index': train_index, 'test_index': test_index }


def create(hyperparams):
    hyperparams = model_hyperparams = {
        'input_shape': hyperparams['input_shape'], # not optional
        'layers': hyperparams.get('layers', [64, 32]),
        'activation': hyperparams.get('activation', 'relu'),
        'dropout': hyperparams.get('dropout', 0.2),
        'learning_rate': hyperparams.get('learning_rate', 0.001),
        'scoring': hyperparams.get('scoring', 'mean_squared_error'),
        'optimizer': hyperparams.get('optimizer', 'rmsprop'),
        'loss': hyperparams.get('loss', 'mean_squared_error'),
        'metrics': hyperparams.get('metrics', ['mae']),
        '__+ABD_exceptLast': hyperparams.get('__+ABD_exceptLast', True),
        '__penultRelu': False, # hyperparams.get('__penultRelu', False),
        '__finalAct': hyperparams.get('__finalAct', 'linear'), # or sigmoid
        'LSTM_activation': hyperparams.get('activation', 'tanh')
    }
    layers = []
    for i, layer in enumerate(hyperparams['layers']):
        if i == 0:
            layers.append(LSTM(layer, input_shape=hyperparams['input_shape'], return_sequences=True, activation=hyperparams['LSTM_activation']))
        elif i == len(hyperparams['layers']) - 1:
            layers.append(LSTM(layer, return_sequences=False, activation=hyperparams['LSTM_activation']))
        else:
            layers.append(LSTM(layer, return_sequences=True, activation=hyperparams['LSTM_activation']))

        if hyperparams['__+ABD_exceptLast'] and i != len(hyperparams['layers']) - 1:
            layers.append(Activation(hyperparams['activation']))
            layers.append(BatchNormalization())
            layers.append(Dropout(hyperparams['dropout']))
    
    # if hyperparams['__penultRelu']:
    #     layers.append(Dense(8, activation='relu'))
    layers.append(Dense(1, activation=hyperparams['__finalAct']))

    model = Sequential(layers)
    model.compile(
        loss=hyperparams['loss'],
        optimizer=hyperparams['optimizer'],
        metrics=hyperparams['metrics']
    )
    return model


def evaluate(model, x, y):
    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return {
        'mse': mse,
        'r2': r2,
    }


def train(model, hyperparams, train_X_y, evaluateAtEnd=True):
    hyperparams = train_hyperparams = {
        'epochs': hyperparams.get('epochs', 10),
        'batch_size': hyperparams.get('batch_size', 32),
    }
    results = model.fit(train_X_y["train_X"], train_X_y["train_y"], epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=1)
    if evaluateAtEnd:
        return evaluate(model, train_X_y["train_X"], train_X_y["train_y"])
    else:
        return results



def all_folds_plot(model, folds, mean=False):
    keys = folds.keys()
    prefixes = set(key.rsplit('_', 1)[0] for key in keys)

    if not mean:
        predictions = [model.predict(folds[prefix + '_X'])[:, -1] for prefix in prefixes]
    else:
        predictions = [model.predict(folds[prefix + '_X']).mean(axis=1) for prefix in prefixes]
    trues = [folds[prefix + '_y'] for prefix in prefixes]

    # Concatenate predictions and trues for all folds
    all_predictions = np.concatenate(predictions)
    all_trues = np.concatenate(trues)

    # Create a dummy range for the x-axis
    all_indices = np.arange(len(all_predictions))

    # Create ticks to demarcate each fold
    fold_ticks = [0]
    for prefix in prefixes:
        fold_ticks.append(fold_ticks[-1] + len(folds[prefix + '_y']))

    # Calculate residuals
    residuals = all_trues - all_predictions.flatten()

    # Create a figure with two vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)

    # Plot predictions vs true values on the first subplot
    ax1.plot(all_indices, all_trues, label='True Values', linestyle='--')
    ax1.plot(all_indices, all_predictions, label='Predictions', linestyle='-')
    ax1.set_ylabel('Value')
    ax1.set_title('Predictions vs True Values Across Folds')
    ax1.legend()

    # Plot residuals on the second subplot
    ax2.scatter(all_indices, residuals, label='Residuals', color='red', alpha=0.2, s=10)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8)  # Add a line for y=0
    ax2.set_ylabel('Residuals')
    ax2.set_xlabel('Index')


    # Add vertical lines for fold boundaries
    # for tick, prefix in zip(fold_ticks, prefixes):
    #     ax1.axvline(x=tick, color='gray', linestyle=':', alpha=0.7)
    #     ax1.text(tick, ax1.get_ylim()[1], prefix, rotation=90, verticalalignment='bottom', fontsize=8, color='gray')
    # for tick, prefix in zip(fold_ticks, prefixes):
    #     ax2.axvline(x=tick, color='gray', linestyle=':', alpha=0.7)
    #     ax2.text(tick, ax1.get_ylim()[1], prefix, rotation=90, verticalalignment='bottom', fontsize=8, color='gray')


    for tick in fold_ticks:
        ax1.axvline(x=tick, color='gray', linestyle=':', alpha=0.7)
        ax2.axvline(x=tick, color='gray', linestyle=':', alpha=0.7)
    for tick, prefix in zip(fold_ticks[:-1], prefixes):
        ax2.text(tick, ax2.get_ylim()[1]*0.95, prefix, rotation=90, verticalalignment='top', fontsize=8, color='gray')

    plt.tight_layout()  # Ensure the layout fits within the figure size
    # plt.show()
    return plt



# def predict(model, df, SPLIT="train", features=[], target="Return", episode_length=30):
#     df = df.copy()

#     predictions = []
#     prediction_indices = []
#     for i in range(len(df)):
#         if df.iloc[i]["SPLIT"] == SPLIT:
#             current_row = df.iloc[i]
            
#             current_row_features = [current_row[feature] for feature in features]

#             predictions.append(current_row_features)
#             prediction_indices.append(current_row.name)
    
#     for prediction in model.predict(predictions):
#         df.loc[prediction_indices, f"{target}_predicted"] = prediction[0]

#     return df