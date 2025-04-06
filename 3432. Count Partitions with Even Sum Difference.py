from typing import List


class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        ans = 0
        left, right = 0, sum(nums)
        for i in range(1, len(nums)):
            left += nums[i]
            right -= nums[i]
            if (left - right) % 2 == 0:
                ans += 1
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.countPartitions([10, 10, 3, 7, 6]))  # 4
    print(slt.countPartitions([1, 2, 2]))  # 0
    print(slt.countPartitions([2, 4, 6, 8]))  # 3

# Done ✅
