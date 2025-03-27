import pandas as pd

def contiguousZones(*args):
    dfs = []
    for file_path in args:
        df = pd.read_csv(file_path)
        df["ticker"] = file_path.split("/")[-1].split(".")[0]
        df.drop(columns=["Ignore", "Close time"], inplace=True)

        df["Open time"] = df["Open time"].apply(lambda x: x * 1000 if len(str(x)) < len("1738369800000000") else x)
        df["Open time"] = pd.to_datetime(df["Open time"], unit='us')
        df.set_index("Open time", inplace=True)
        df.sort_index(inplace=True)

        dfs.append(df)

    master_df = pd.concat(dfs, axis=1, keys=[df["ticker"].iloc[0] for df in dfs])
    master_df.dropna(inplace=True)
    for ticker in master_df.columns.levels[0]:
        master_df = master_df.drop(columns=[(ticker, "ticker")])

    master_df["Gap"] = master_df.index.to_series().diff().dt.total_seconds().div(60).fillna(0)
    valid_intervals = master_df[master_df["Gap"] <= 30]
    last_gap_index = master_df[master_df["Gap"] > 30].index[-1]
    master_df = master_df.loc[last_gap_index:]
    

    master_df.drop(columns=["Gap"], inplace=True)
    master_df.columns.levels[0].drop("Gap")

    return master_df