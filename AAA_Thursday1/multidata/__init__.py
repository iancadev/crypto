import pandas as pd
from .. import data
from .. import LSTM_returns
from sklearn.preprocessing import StandardScaler

def load_assets(tickers, sampling="30m"):
    multi_df = pd.DataFrame(columns=tickers, index=['df'], dtype=object)
    multi_df.loc['df'] = [ data.load_asset(ticker, sampling=sampling) for ticker in tickers ]
    return multi_df

def normalize(multi_df, numerical_features=[], exclude=[]):
    """Categorical encoding is something you do during LOADING, okay?"""
    scalers = {}
    for ticker in multi_df.columns:
        df = multi_df.loc['df', ticker]
        
        # Automatically determine numerical and categorical features if not provided
        if not numerical_features:
            numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        # if not categorical_features:
        #     categorical_features = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Exclude specified columns
        numerical_features = [col for col in numerical_features if col not in exclude]
        # categorical_features = [col for col in categorical_features if col not in exclude]
        
        # Apply standard scaling to numerical features
        for col in numerical_features:
            if col not in df.columns:
                continue

            if col not in scalers:
                scalers[col] = StandardScaler()
                # Fit the scaler on the first dataframe
                scalers[col].fit(df[[col]])
            # Transform the column
            df[col] = scalers[col].transform(df[[col]])
        
        # Update the dataframe in multi_df
        multi_df.loc['df', ticker] = df

    return multi_df, scalers

def train_test_split(multi_df, split=pd.Timestamp('2024-01-01')):
    for ticker in multi_df.columns:
        df = multi_df.loc['df', ticker]
        df['SPLIT'] = 'train'
        df.loc[df.index >= split, 'SPLIT'] = 'test'
    return multi_df

def vectorize_train_data(multi_df, target="Return_Target", features=["Open", "High", "Low", "Close"], episode_length=30):
    stuff = {}
    for ticker in multi_df.columns:
        df = multi_df.loc['df', ticker]
        new_stuff = vectorize_train_data(df, target=target, features=features, episode_length=episode_length)
        for key, value in new_stuff.items():
            if key in stuff and isinstance(stuff[key], list) and isinstance(value, list):
                stuff[key] += value
            else:
                stuff[key] = value
    return stuff
