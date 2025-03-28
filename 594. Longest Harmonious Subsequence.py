from typing import List
from collections import Counter


class Solution:
    def findLHS(self, nums: List[int]) -> int:
        # Version 1
        # c = Counter(nums)
        # return max((c[key] + c[key + 1] for key in c if key + 1 in c), default=0)

        # Version 2
        c = Counter(nums)
        num_set = c.keys()
        ans = 0
        for key in num_set:
            if key - 1 in num_set:
                maximum = max(ans, c[key] + c[key - 1])
                ans = maximum
            if key + 1 in num_set:
                maximum = max(ans, c[key] + c[key + 1])
                ans = maximum
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.findLHS([1, 3, 2, 2, 5, 2, 3, 7]))  # 5
    print(slt.findLHS([1, 2, 3, 4]))  # 2
    print(slt.findLHS([1, 1, 1, 1]))  # 0

# Done ✅
