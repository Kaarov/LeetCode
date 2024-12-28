import pandas as pd


def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    # SELECT name as Customers FROM Customers WHERE id NOT IN (
    #   SELECT customerId FROM Orders
    # );
    return pd.DataFrame({"Customers": customers[~customers["id"].isin(orders["customerId"])]["name"]})


if __name__ == "__main__":
    customers = pd.read_csv("pandas dataset/183. Customers Who Never Order Customers.csv")
    orders = pd.read_csv("pandas dataset/183. Customers Who Never Order Orders.csv")
    print(find_customers(customers, orders))

# Done ✅
