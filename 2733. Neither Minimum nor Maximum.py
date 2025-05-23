from typing import List


class Solution:
    def findNonMinOrMax(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return -1

        nums.sort()

        return nums[len(nums) // 2]


if __name__ == '__main__':
    slt = Solution()
    print(slt.findNonMinOrMax([3, 2, 1, 4]))  # 2
    print(slt.findNonMinOrMax([1, 2]))  # -1
    print(slt.findNonMinOrMax([2, 1, 3]))  # 2

# Done ✅
