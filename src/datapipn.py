import pandas as pd

def datapipeline(pathfile):
    df=pd.read_csv(pathfile)
    df.drop_duplicates(subset="headline",
                     keep='last', inplace=True)
    print(df.head())
    return df
