from typing import List


class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        first = nums.pop(0)
        nums.sort()
        return first + nums.pop(0) + nums.pop(0)


if __name__ == '__main__':
    slt = Solution()
    print(slt.minimumCost([1, 2, 3, 12]))  # 6
    print(slt.minimumCost([5, 4, 3]))  # 12
    print(slt.minimumCost([10, 3, 1, 1]))  # 12

# Done ✅
