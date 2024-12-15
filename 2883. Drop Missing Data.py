import pandas as pd


def dropMissingData(students: pd.DataFrame) -> pd.DataFrame:
    students.dropna(inplace=True, subset=["name"])
    return students


if __name__ == "__main__":
    students = pd.read_csv("pandas dataset/2883. Drop Missing Data.csv")
    print(dropMissingData(students))

# Done ✅
