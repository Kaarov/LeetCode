from typing import List


class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        nums.append(-1)
        ans = []
        summa = 0
        for i in range(len(nums) - 1):
            summa += nums[i]
            if nums[i] >= nums[i + 1]:
                ans.append(summa)
                summa = 0
        return max(ans)


if __name__ == "__main__":
    slt = Solution()
    print(slt.maxAscendingSum([10, 20, 30, 5, 10, 50]))  # 65
    print(slt.maxAscendingSum([10, 20, 30, 40, 50]))  # 150
    print(slt.maxAscendingSum([12, 17, 15, 13, 10, 11, 12]))  # 33

# Done ✅
