import pandas as pd


def changeDatatype(students: pd.DataFrame) -> pd.DataFrame:
    # return students.astype({"grade": int})
    students["grade"] = students["grade"].apply(lambda x: int(x))
    return students


if __name__ == "__main__":
    students = pd.read_csv("pandas dataset/2886. Change Data Type.csv")
    print(changeDatatype(students))

# Done ✅
