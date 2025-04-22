from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time, os, requests, zipfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def __load_file(fname):
    fname = os.path.join(os.path.dirname(__file__), fname)
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        return None

def __download_asset(ticker, sampling):
    url = f"https://data.binance.vision/?prefix=data/spot/monthly/klines/{ticker}/{sampling}/"
    output_csv = f"tmp/{ticker}.csv"
    target_csv = f"assets/{ticker}_{sampling}.csv"

    # Set up Selenium WebDriver
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    # service = Service('path/to/msedgedriver')  # Replace with the path to your Edge WebDriver
    driver = webdriver.Edge(options=options)

    try:
        driver.get(url)
        time.sleep(3.5) 

        # Find all links ending with ".zip"
        links = driver.find_elements(By.XPATH, "//a[substring(@href, string-length(@href) - 3) = '.zip']")
        zip_links = [link.get_attribute('href') for link in links]

        for link in zip_links:
            print(link)

        # Create the local folder if it doesn't exist
        os.makedirs('tmp', exist_ok=True)

        with open(output_csv, 'w') as file:
            file.write("Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore\n")

        for i, link in enumerate(zip_links):
            print(f"{i+1}/{len(zip_links)}")
            file_name = os.path.join('tmp', os.path.basename(link))
            response = requests.get(link)
            with open(file_name, 'wb') as file:
                file.write(response.content)

            time.sleep(0.1)

            if zipfile.is_zipfile(file_name):
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall('tmp')

            os.remove(file_name)

            time.sleep(0.1)

            # Get the name of the extracted CSV file
            extracted_files = zip_ref.namelist()
            if len(extracted_files) == 1:
                extracted_csv = os.path.join('tmp', extracted_files[0])
                
                # Append the content of the extracted CSV to master.csv
                with open(extracted_csv, 'r') as source_file:
                    with open(output_csv, 'a') as master_file:
                        master_file.write(source_file.read())
                
                # Delete the original extracted CSV file
                os.remove(extracted_csv)
            time.sleep(1)
        
        # Move the consolidated CSV to the target location
        os.makedirs(os.path.dirname(target_csv), exist_ok=True)
        os.replace(output_csv, target_csv)
    finally:
        driver.quit()

def load_asset(ticker, sampling="30m"):
    df = __load_file(f"assets/{ticker}_{sampling}.csv")
    if df is None:
        __download_asset(ticker, sampling)
        return load_asset(ticker, sampling)

    df['Open time'] = df['Open time'].apply(lambda x: x * 1000 if len(str(x)) < len(str(1738470600000000)) else x)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='us')
    df.set_index('Open time', inplace=True)
    df = df.drop(columns=['Close time', 'Ignore'])

    df.sort_index(inplace=True)
    return df



def subset(df, start=pd.Timestamp('2000-01-01'), end=pd.Timestamp('2000-01-01')):
    start = pd.to_datetime(start)
    if start not in df.index:
        start = df.index[df.index.get_indexer([start], method='nearest')[0]]
    end = pd.to_datetime(end)
    if end not in df.index:
        end = df.index[df.index.get_indexer([end], method='nearest')[0]]
    return df.loc[start:end]

def row_delta(row1, row2):
    return abs(pd.Timedelta(row1.name - row2.name))

def report_gaps(df, delta=pd.Timedelta('30m')):
    gapTimes = []
    rows = list(df.iterrows())
    for S1, S2 in zip(rows[:-1], rows[1:]):
        i1, row1 = S1
        i2, row2 = S2
        if row_delta(row1, row2) > delta:
            gapTimes.append((row1.name, row2.name, row_delta(row1, row2)))
    return gapTimes

def report_and_print_gaps(df, delta=pd.Timedelta('30m')):
    gaps = report_gaps(df, delta=delta)
    for gap in gaps:
        print(f"Gap of {gap[2]} \t\t from {gap[0]} to {gap[1]}")


def normalize_data(df, numerical_columns=[], categorical_columns=[], exclude_columns=['Return']):
    scaler = MinMaxScaler()
    if not numerical_columns:
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        numerical_columns = [nc for nc in numerical_columns if nc not in exclude_columns]
    if not categorical_columns:
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        categorical_columns = [cc for cc in categorical_columns if cc not in exclude_columns]

    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    for col in categorical_columns:
        if df[col].nunique() > 2:
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, one_hot], axis=1)
            df = df.drop(columns=[col])
        else:
            df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})
    return df






def add_returns(df):
    df['Return'] = (df['Open'].shift(-1) - df['Open']) / df['Open']
    return df

# this F&G is *normalized*
def add_fear_and_greed(df):
    f_n_g_csv = __load_file("assets/fear_and_greed.csv")
    if f_n_g_csv is None:
        raise FileNotFoundError("Fear and Greed CSV file not found.")
    
    f_n_g_csv['date'] = pd.to_datetime(f_n_g_csv['date'])
    f_n_g_csv.set_index('date', inplace=True)

    df = df.copy()  # Avoid potential SettingWithCopyWarning
    df.loc[:, 'plain-date'] = df.index.map(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    df.loc[:, 'F&G value'] = df['plain-date'].map(lambda x: f_n_g_csv['value'].loc[x] / 100 if x.date() in f_n_g_csv.index.date else None)
    df.drop(columns=['plain-date'], inplace=True)
    return df

def add_high_low_range(df):
    pass

def add_high_open_ratio(df):
    pass

def add_low_open_ratio(df):
    pass

def add_normalized_volume(df):
    pass

def add_norm_momentum(df):
    pass

def add_norm_RSI(df):
    pass

def add_norm_adx(df):
    pass



def add_target_returns(df):
    df['Return_Target'] = df['Return'].shift(-1)
    return df






def add_RSI(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_ADX(df, window=14):
    high = df['High'].rolling(window).max()
    low = df['Low'].rolling(window).min()
    tr = pd.concat([high - low, abs(high - df['Close'].shift()), abs(low - df['Close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_dm = df['High'].diff().where(df['High'].diff() > df['Low'].diff(), 0)
    minus_dm = -df['Low'].diff().where(df['Low'].diff() > df['High'].diff(), 0)
    
    plus_di = 100 * (plus_dm.rolling(window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window).sum() / atr)
    
    adx = 100 * ((abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window).mean())
    df['ADX'] = adx
    return df

def add_momentum(df, window=14):
    df['Momentum'] = df['Close'].diff(window)
    return df

def add_high_low(df, window=14):
    df['High_Low'] = df['High'].rolling(window).max() - df['Low'].rolling(window).min()
    return df

def add_high_open(df, window=14):
    df['High_Open'] = df['High'].rolling(window).max() - df['Open'].rolling(window).min()
    return df

def add_low_open(df, window=14):
    df['Low_Open'] = df['Low'].rolling(window).min() - df['Open'].rolling(window).max()
    return df

def add_expected_returns(df, window=14):
    df['Expected Return'] = df['Return'].rolling(Window)
    pass

def add_indicators(df, window=14):
    df = add_RSI(df, window)
    df = add_ADX(df, window)
    df = add_momentum(df, window)
    df = add_high_low(df, window)
    df = add_high_open(df, window)
    df = add_low_open(df, window)
    return df



def add_lookback_returns(df, lookback=14):
    for i in range(1, lookback + 1):
        df[f'Return_Lookback_{i}'] = df['Return'].shift(i)
    return df



def train_test_split(df, split=0.8):
    train_size = int(len(df) * split)
    df['SPLIT'] = pd.Categorical([None] * len(df), categories=["train", "test"])
    train_indices = df.index[:train_size]
    test_indices = df.index[train_size:]
    df.loc[train_indices, "SPLIT"] = "train"
    df.loc[test_indices, "SPLIT"] = "test"
    return df

def test_data_iterator(splitted_df, lookback=14):
    # for each row with row["SPLIT"] = "test"
    # yield a dictionary: { 'current': that row, 'lookback': previous lookback rows }

    for i in range(lookback, len(splitted_df)):
        if splitted_df.iloc[i]["SPLIT"] == "test":
            current_row = splitted_df.iloc[i]
            lookback_rows = splitted_df["Return"].iloc[i - lookback:i]
            yield {"current": current_row, "lookback": lookback_rows}