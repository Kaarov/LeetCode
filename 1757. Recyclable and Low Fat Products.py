import pandas as pd


def find_products(products: pd.DataFrame) -> pd.DataFrame:
    # SELECT product_id from Products WHERE low_fats = recyclable AND low_fats = 'Y';
    return products[(products["low_fats"] == "Y") & (products["recyclable"] == "Y")]["product_id"].to_frame()


if __name__ == "__main__":
    products = pd.read_csv("pandas dataset/1757. Recyclable and Low Fat Products.csv")
    print(find_products(products))

# Done ✅
