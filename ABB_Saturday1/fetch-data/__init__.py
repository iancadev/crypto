from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time, os, requests, zipfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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



def load_scaled_asset(ticker1, scale_by, sampling="30m", save_to=None):
    # for example: ticker1 = "ETHBTC" scale_by = "BTCUSDT"
    asset = load_asset(ticker1, sampling=sampling)
    scaling = load_asset(scale_by, sampling=sampling)

    # Open,High,Low,Close,Volume,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume
    scaled_asset = pd.DataFrame()

    for index, row in asset.iterrows():
        if index in scaling.index:
            scaling_row = scaling.loc[index]
            scaled_row = row.copy()
            scaled_row['Open'] /= scaling_row['Open']
            scaled_row['High'] /= scaling_row['High']
            scaled_row['Low'] /= scaling_row['Low']
            scaled_row['Close'] /= scaling_row['Close']
            scaled_row['Quote asset volume'] /= scaling_row['Close']
            # scaled_row['Taker buy base asset volume'] /= scaling_row['Close']
            scaled_row['Taker buy quote asset volume'] /= scaling_row['Close']
            scaled_asset = pd.concat([scaled_asset, scaled_row.to_frame().T])

    if save_to:
        scaled_asset.to_csv(save_to)

    return scaled_asset