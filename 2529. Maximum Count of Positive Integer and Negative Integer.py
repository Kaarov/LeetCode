from typing import List


class Solution:
    def maximumCount(self, nums: List[int]) -> int:
        neg = 0
        i = 0
        while i < len(nums) and nums[i] < 0:
            neg += 1
            i += 1
        zero = nums.count(0)
        return max(neg, len(nums[i + zero:]))


if __name__ == '__main__':
    slt = Solution()
    print(slt.maximumCount(nums=[-2, -1, -1, 1, 2, 3]))  # 3
    print(slt.maximumCount(nums=[-3, -2, -1, 0, 0, 1, 2]))  # 3
    print(slt.maximumCount(nums=[5, 20, 66, 1314]))  # 4

# Done ✅
