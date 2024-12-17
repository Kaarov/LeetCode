import pandas as pd


def renameColumns(students: pd.DataFrame) -> pd.DataFrame:
    # students = students.rename(columns={
    #     "id": "student_id",
    #     "first": "first_name",
    #     "last": "last_name",
    #     "age": "age_in_years",
    # })
    students.columns = ["student_id", "first_name", "last_name", "age_in_years"]
    return students


if __name__ == "__main__":
    students = pd.read_csv('pandas dataset/2885. Rename Columns.csv')
    print(renameColumns(students))

# Done ✅
