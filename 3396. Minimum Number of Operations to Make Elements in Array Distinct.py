from typing import List


class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        ans = 0
        while True:
            if len(nums) == len(set(nums)) or not nums:
                return ans

            nums = nums[3:]
            ans += 1


if __name__ == '__main__':
    slt = Solution()
    print(slt.minimumOperations([1, 2, 3, 4, 2, 3, 3, 5, 7]))  # 2
    print(slt.minimumOperations([4, 5, 6, 4, 4]))  # 2
    print(slt.minimumOperations([6, 7, 8, 9]))  # 0

# Done ✅
