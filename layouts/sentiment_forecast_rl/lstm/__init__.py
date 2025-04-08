import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score
import optuna


def cast_train_data(df, lookback=10, walkforward_folds=0, flavor="plain"):
    if "Target" not in df.columns:
        raise ValueError("Missing target column")
        return None

    if "SPLIT" in df.columns:
        train_df = df[df["SPLIT"] == "train"].copy()
        train_df.drop(columns=["SPLIT"], inplace=True)
    else:
        train_df = df.copy()


    def formatXy(fold):
        if len(fold) <= lookback:
            raise ValueError("Your fold is smaller than the lookback+1 period, so bro, fix it, please")
            return None, None
        X = pd.concat([fold.shift(i) for i in range(lookback, 0, -1)], axis=1).dropna()
        X = np.reshape(X.values, (len(X), lookback, len(fold.columns)))
        y = np.reshape(fold["Target"].shift(-lookback).dropna(), (-1, 1))
        return X, y
    

    if walkforward_folds == 0:
        foldEndIndices = [0, len(train_df)]
        X, y = formatXy(train_df)
        return { 'X_train': X, 'y_train': y }
    
    foldEndIndices = [int(len(df) * (i + 1) / walkforward_folds) for i in range(walkforward_folds)]
    foldEndIndices = [0] + foldEndIndices + [len(df)]

    folds = [df.iloc[foldEndIndices[i]:foldEndIndices[i + 1]] for i in range(len(foldEndIndices) - 1)]
    folds = [fold.reset_index(drop=True) for fold in folds]

    # format X and format Y
    trainTestFolds = []
    for i in range(len(folds)-2):
        train_fold = pd.concat(folds[0:i + 1], axis=0)
        test_fold = folds[i + 1]
        X_train, y_train = formatXy(train_fold)
        X_test, y_test = formatXy(test_fold)
        trainTestFolds.append({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        })
    
    train_final = pd.concat(folds, axis=0)
    X_final, y_final = formatXy(train_final)
    finalData = {
        'X_train': X_final,
        'y_train': y_final,
    }

    return { 'train-test-folds': trainTestFolds, 'final-train': finalData }
    


def fill_hyperparams(hyperparams, example_input=None):
    if example_input is None and not hyperparams.get('input_shape', False):
        raise ValueError("Missing input shape in hyperparameters")
        return None
    
    if not hyperparams.get('input_shape', False):
        hyperparams['input_shape'] = example_input.shape[1:]

    hyperparams['learning_rate'] = hyperparams.get('learning_rate', 0.001)
    hyperparams['batch_size'] = hyperparams.get('batch_size', 32)
    hyperparams['epochs'] = hyperparams.get('epochs', 10)
    hyperparams['layers'] = hyperparams.get('layers', (50,))
    hyperparams['dropout'] = hyperparams.get('dropout', 0.2)
    hyperparams['activation'] = hyperparams.get('activation', 'relu')
    hyperparams['optimizer'] = hyperparams.get('optimizer', 'adam')
    hyperparams['scoring'] = hyperparams.get('scoring', 'mse')
    return hyperparams


def make_model(hyperparams):
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

    model = Sequential(layers)
    model.compile(
        loss=hyperparams['scoring'],
        optimizer=hyperparams['optimizer'],
        metrics=['accuracy']
    )
    return model


def model_evaluations(model, test_x, test_y):
    predictions = model.predict(test_x)
    mse = mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    return {'mse': mse, 'r2': r2}


def train_model(model, train_x, train_y, hyperparams, until=None):
    model.train(train_x, train_y, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=1)
    return model_evaluations(model, train_x, train_y)


def train_model_on_folds(casted_data, hyperparams, until=None):
    trainTestFolds = casted_data['train-test-folds']

    hyperparams = hyperparams.copy()
    hyperparams['input_shape'] = trainTestFolds[0]['X_train'].shape[1:]

    # train the model on each fold and evaluate
    scores = []
    for fold in trainTestFolds:
        model = make_model(hyperparams)
        X_train, y_train = fold['X_train'], fold['y_train']
        X_test, y_test = fold['X_test'], fold['y_test']

        # fit the model
        model.fit(X_train, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=0)

        # evaluate the model
        scores.append(model_evaluations(model, X_test, y_test))

    model = make_model(hyperparams)
    X_train, y_train = casted_data['final-train']['X_train'], casted_data['final-train']['y_train']
    model.fit(X_train, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=0)

    scores.append(model_evaluations(model, X_train, y_train))
    return {'model': model, 'scores': scores}


def forecast_model(model, test_x, lookback=10):
    test_x = test_x.copy()
    test_x["Target"] = test_x["Open"].copy()
    casted = cast_train_data(test_x, lookback=lookback)
    X_test = casted['X_train']
    return model.predict(X_test)


def train_optimal_models(hyperparam_ranges, train_x, train_y, trials=None):
    def objective(trial):
        hyperparams = {
            'input_shape': train_x.shape[1:],
            'learning_rate': trial.suggest_float('learning_rate', hyperparam_ranges['learning_rate'][0], hyperparam_ranges['learning_rate'][1]),
            'batch_size': trial.suggest_categorical('batch_size', hyperparam_ranges['batch_size']),
            'epochs': trial.suggest_int('epochs', hyperparam_ranges['epochs'][0], hyperparam_ranges['epochs'][1]),
            'layers': tuple(trial.suggest_int(f'layers_{i}', hyperparam_ranges['layers'][0], hyperparam_ranges['layers'][1]) for i in range(hyperparam_ranges['layers'][2])),
            'dropout': trial.suggest_float('dropout', hyperparam_ranges['dropout'][0], hyperparam_ranges['dropout'][1]),
            'activation': trial.suggest_categorical('activation', hyperparam_ranges['activation']),
            'optimizer': trial.suggest_categorical('optimizer', hyperparam_ranges['optimizer']),
            'scoring': trial.suggest_categorical('scoring', hyperparam_ranges['scoring'])
        }

        model = make_model(hyperparams)
        model.fit(train_x, train_y, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=0)
        return model_evaluations(model, train_x, train_y)['mse']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)
    # TODO: format the information...
    return study





def save_model(model, path):
    model.save(path)

def load_model(path):
    return load_model(path)