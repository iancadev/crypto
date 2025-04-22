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


def __get_tickers_list():
    url = f"https://data.binance.vision/?prefix=data/spot/daily/klines/"
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Edge(options=options)

    try:
        driver.get(url)
        time.sleep(1)

        # Locate all <a> elements inside <tbody id="listing">
        elements = driver.find_elements(By.CSS_SELECTOR, "tbody#listing a")
        texts = [element.text for element in elements]
        texts = [texts.split("/")[0] for i in texts]
        texts = [i for i in texts if i.isalnum()]

        return texts
    finally:
        driver.quit()



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



available_times = {}
def __get_available_times(ticker):
    if ticker in available_times:
        return available_times[ticker]

    url = f"https://data.binance.vision/?prefix=data/spot/daily/klines/{ticker}/30m/"
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Edge(options=options)

    try:
        driver.get(url)
        time.sleep(1)

        # Locate all <a> elements inside <tbody id="listing">
        elements = driver.find_elements(By.CSS_SELECTOR, "tbody#listing a")
        texts = [element.text for element in elements]

        texts = [i.split(".")[0].split("-30m-")[1] for i in texts if i.endswith(".zip")]

        available_times[ticker] = texts
        return texts
    finally:
        driver.quit()


def plot_available_times(tickers):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        times = pd.Series(__get_available_times(ticker))
        asset_times = pd.to_datetime(times, format='%Y-%m-%d')
        asset_times = asset_times.sort_values()
        
        if asset_times.empty:
            print(f"No available times for ticker: {ticker}")
            continue

        consecutive_pairs = []
        start_pair = asset_times.iloc[0]
        end_pair = asset_times.iloc[0]
        for i in range(1, len(asset_times)):
            if (asset_times.iloc[i] - asset_times.iloc[i-1]).days > 1:
                if start_pair != end_pair:
                    consecutive_pairs.append((start_pair, end_pair))
                start_pair = asset_times.iloc[i]
            end_pair = asset_times.iloc[i]

        for start, end in consecutive_pairs:
            plt.plot([start, end], [ticker, ticker], marker='o', label=ticker)

    plt.xlabel("Time")
    plt.ylabel("Ticker")
    plt.title("Available Times for Tickers")
    plt.grid(True)
    plt.show()