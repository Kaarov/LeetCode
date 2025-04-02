from typing import List


class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        ans = 0
        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums) - 1):
                for k in range(j + 1, len(nums)):
                    ans = max((nums[i] - nums[j]) * nums[k], ans)
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.maximumTripletValue([12, 6, 1, 2, 7]))  # 77
    print(slt.maximumTripletValue([1, 10, 3, 4, 19]))  # 133
    print(slt.maximumTripletValue([1, 2, 3]))  # 0

# Done ✅
