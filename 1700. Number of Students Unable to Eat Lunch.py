class Solution:
    def countStudents(self, students: list[int], sandwiches: list[int]) -> int:
        while students and sandwiches[0] in students:
            student = students.pop(0)
            if student == sandwiches[0]:
                sandwiches.pop(0)
            else:
                students.append(student)

        return len(students)


if __name__ == "__main__":
    slt = Solution()
    assert slt.countStudents([1, 1, 0, 0], [0, 1, 0, 1]) == 0
    assert slt.countStudents([1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1]) == 3

# Done ✅
