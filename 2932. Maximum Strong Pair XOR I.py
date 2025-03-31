from itertools import combinations
from typing import List


class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        ans = 0
        comb = combinations(nums, 2)
        for i, j in comb:
            if abs(i - j) <= min(i, j):
                ans = max(ans, i ^ j)
        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.maximumStrongPairXor([1, 2, 3, 4, 5]))  # 7
    print(slt.maximumStrongPairXor([10, 100]))  # 0
    print(slt.maximumStrongPairXor([5, 6, 25, 30]))  # 7

# Done ✅
