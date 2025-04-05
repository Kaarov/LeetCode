from typing import List
from itertools import combinations
from operator import ixor
from functools import reduce


class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        ans = 0
        for num in range(1, len(nums) + 1):
            for cmb in combinations(nums, num):
                ans += reduce(ixor, cmb)

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.subsetXORSum([1, 3]))  # 6
    print(slt.subsetXORSum([5, 1, 6]))  # 28
    print(slt.subsetXORSum([3, 4, 5, 6, 7, 8]))  # 480

# Done ✅
