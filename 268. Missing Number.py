from typing import List


class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        numbers = {i for i in range(len(nums) + 1)}
        ans = numbers - set(nums)
        return ans.pop()


if __name__ == '__main__':
    slt = Solution()
    print(slt.missingNumber([3, 0, 1]))  # 2
    print(slt.missingNumber([0, 1]))  # 2
    print(slt.missingNumber([9, 6, 4, 2, 3, 5, 7, 0, 1]))  # 8

# Done ✅
