import pandas as pd
from typing import List


def createDataframe(student_data: List[List[int]]) -> pd.DataFrame:
    # df = pd.DataFrame(student_data, columns=['student_id', 'age'])
    # return df

    data = {
        "student_id": [],
        "age": [],
    }
    for student in student_data:
        data["student_id"].append(student[0])
        data["age"].append(student[1])

    ans = pd.DataFrame(data)
    return ans


if __name__ == '__main__':
    student_data = [
        [1, 15],
        [2, 11],
        [3, 11],
        [4, 20]
    ]
    print(createDataframe(student_data))

# Done ✅
