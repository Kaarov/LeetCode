from typing import List


class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        ans = 0
        nums = list(set(nums))
        if nums.count(0):
            nums.remove(0)
        while nums:
            x = min(nums)
            nums = [
                num - x
                for num in nums
                if num - x > 0
            ]
            ans += 1
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.minimumOperations([1, 5, 0, 3, 5]))  # 3
    print(slt.minimumOperations([0]))  # 0

# Done ✅
