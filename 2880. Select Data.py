import pandas as pd


def selectData(students: pd.DataFrame) -> pd.DataFrame:
    students = students[students["student_id"] == 101]
    return students[["name", "age"]]


if __name__ == "__main__":
    students = pd.read_csv("pandas dataset/2880. Select Data.csv")
    print(selectData(students))

# Done ✅
