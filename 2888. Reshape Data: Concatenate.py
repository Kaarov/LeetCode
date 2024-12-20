import pandas as pd


def concatenateTables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1, df2])


if __name__ == "__main__":
    df1 = pd.read_csv("pandas dataset/2888. Reshape Data: Concatenate.csv").head(4)
    df2 = pd.read_csv("pandas dataset/2888. Reshape Data: Concatenate.csv").tail(2)
    print(concatenateTables(df1, df2))

# Done ✅
