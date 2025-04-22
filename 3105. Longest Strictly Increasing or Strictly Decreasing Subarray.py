from typing import List


class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        increasing = 1
        decreasing = 1

        ans = 0

        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                decreasing += 1
                increasing = 1
            elif nums[i] < nums[i + 1]:
                increasing += 1
                decreasing = 1
            else:
                decreasing = increasing = 1

            ans = max(ans, increasing, decreasing)

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.longestMonotonicSubarray([1, 4, 3, 3, 2]))  # 2
    print(slt.longestMonotonicSubarray([3, 3, 3, 3]))  # 1
    print(slt.longestMonotonicSubarray([3, 2, 1]))  # 3

# Done ✅
