from typing import List


class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        ans = -1

        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] < nums[j]:
                    ans = max(ans, nums[j] - nums[i])

        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.maximumDifference([7, 1, 5, 4]))  # 4
    print(slt.maximumDifference([9, 4, 3, 2]))  # -1
    print(slt.maximumDifference([1, 5, 2, 10]))  # 9

# Done ✅
