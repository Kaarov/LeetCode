import pandas as pd


def selectFirstRows(employees: pd.DataFrame) -> pd.DataFrame:
    return employees.head(3)


if __name__ == '__main__':
    employees = pd.read_csv('pandas dataset/2879. Display the First Three Rows.csv')
    print(selectFirstRows(employees))

# Done ✅
