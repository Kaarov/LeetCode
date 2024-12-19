import pandas as pd


def fillMissingValues(products: pd.DataFrame) -> pd.DataFrame:
    products["quantity"] = products["quantity"].fillna(value=0)
    return products


if __name__ == "__main__":
    products = pd.read_csv("pandas dataset/2887. Fill Missing Data.csv")
    print(fillMissingValues(products))

# Done ✅
