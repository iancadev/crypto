import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score
# optuna scrap goes in the ipynb, okay?

def vectorize_train_data(splitted_df, target="Return", features=["Open", "High", "Low", "Close", "Volume"], episode_length=30):
    # for each row with row["SPLIT"] = "train"
    # yield a train_X and a train_y array, where train_y is just "Return"
    # and train_X is the rest of the columns, but without "SPLIT" and "Return"
    
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for i in range(len(splitted_df)):
        if splitted_df.iloc[i]["SPLIT"] == "train":
            current_row = splitted_df.iloc[i]
            
            current_row_features = [current_row[feature] for feature in features]
            train_X.append(current_row_features)
            
            train_y.append(current_row["Return"])
        if splitted_df.iloc[i]["SPLIT"] == "test":
            current_row = splitted_df.iloc[i]
            
            current_row_features = [current_row[feature] for feature in features]
            test_X.append(current_row_features)
            
            test_y.append(current_row["Return"])

    # Convert train_X and test_X into episodes
    train_X = np.array([train_X[i:i + episode_length] for i in range(len(train_X) - episode_length + 1)])
    train_y = np.array(train_y[episode_length - 1:])

    # Combine train and test data for overlapping episodes
    combined_X = train_X.tolist() + test_X
    combined_y = train_y.tolist() + test_y

    # Generate episodes for test_X and test_y
    test_X = np.array([combined_X[i:i + episode_length] for i in range(len(train_X), len(combined_X) - episode_length + 1)])
    test_y = np.array(combined_y[len(train_X) + episode_length - 1:])
    
    return { 'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y }


def create(hyperparams):
    hyperparams = model_hyperparams = {
        'input_shape': hyperparams['input_shape'], # not optional
        'layers': hyperparams.get('layers', [64, 32]),
        'activation': hyperparams.get('activation', 'relu'),
        'dropout': hyperparams.get('dropout', 0.2),
        'learning_rate': hyperparams.get('learning_rate', 0.001),
        'scoring': hyperparams.get('scoring', 'mean_squared_error'),
        'optimizer': hyperparams.get('optimizer', 'adam'),
        'loss': hyperparams.get('loss', 'mean_squared_error'),
        'metrics': hyperparams.get('metrics', ['accuracy']),
    }
    layers = []
    for i, layer in enumerate(hyperparams['layers']):
        if i == 0:
            layers.append(LSTM(layer, input_shape=hyperparams['input_shape'], return_sequences=True))
        elif i == len(hyperparams['layers']) - 1:
            layers.append(LSTM(layer, return_sequences=False))
        else:
            layers.append(LSTM(layer, return_sequences=True))

        layers.append(BatchNormalization())
        layers.append(Dropout(hyperparams['dropout']))
        layers.append(Activation(hyperparams['activation']))

    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(
        loss=hyperparams['scoring'],
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


def train(model, hyperparams, train_X_y):
    hyperparams = train_hyperparams = {
        'epochs': hyperparams.get('epochs', 10),
        'batch_size': hyperparams.get('batch_size', 32),
    }
    model.fit(train_X_y["train_X"], train_X_y["train_y"], epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=1)
    return evaluate(model, train_X_y["train_X"], train_X_y["train_y"])


def predict(model, df, SPLIT="train", features=[], target="Return", episode_length=30):
    df = df.copy()

    predictions = []
    prediction_indices = []
    for i in range(len(df)):
        if df.iloc[i]["SPLIT"] == SPLIT:
            current_row = df.iloc[i]
            
            current_row_features = [current_row[feature] for feature in features]

            predictions.append(current_row_features)
            prediction_indices.append(current_row.name)
    
    for prediction in model.predict(predictions):
        df.loc[prediction_indices, f"{target}_predicted"] = prediction[0]

    return df