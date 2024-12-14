import pandas as pd


def dropDuplicateEmails(customers: pd.DataFrame) -> pd.DataFrame:
    # df = customers.drop_duplicates(subset='email', keep='first')
    uniqueEmails = set()
    for idx, customer in enumerate(customers.itertuples()):
        if customer.email in uniqueEmails:
            customers.drop(index=idx, inplace=True)
        else:
            uniqueEmails.add(customer.email)
    return customers


if __name__ == "__main__":
    customers = pd.read_csv("pandas dataset/2882. Drop Duplicate Rows.csv")
    print(dropDuplicateEmails(customers))

# Done ✅
