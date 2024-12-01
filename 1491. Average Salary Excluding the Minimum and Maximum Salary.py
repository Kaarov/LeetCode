from typing import List


class Solution:
    def average(self, salary: List[int]) -> float:
        salary.remove(max(salary))
        salary.remove(min(salary))
        return sum(salary) / len(salary)


slt = Solution()
print(slt.average([4000, 3000, 1000, 2000]))

# Done ✅
