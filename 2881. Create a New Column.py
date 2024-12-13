import pandas as pd


def createBonusColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees["bonus"] = employees["salary"] * 2
    return employees


if __name__ == "__main__":
    employees = pd.read_csv("pandas dataset/2881. Create a New Column.csv")
    print(createBonusColumn(employees))

# Done ✅
