from typing import List
from collections import Counter


class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        if len(nums) % 2 != 0:
            return False
        c = Counter(nums)
        for value in c.values():
            if value % 2 != 0:
                return False
        return True


if __name__ == '__main__':
    slt = Solution()
    print(slt.divideArray([3, 2, 3, 2, 2, 2]))  # True
    print(slt.divideArray([1, 2, 3, 4]))  # False

# Done ✅
