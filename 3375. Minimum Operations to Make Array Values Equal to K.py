from typing import List


class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if min(nums) < k:
            return -1

        nums.sort()

        ans = 0
        while len(set(nums)) != 1:
            count = nums.count(max(nums))
            nums = nums[:-count]
            ans += 1

        if k not in nums:
            ans += 1

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.minOperations([5, 2, 5, 4, 5], 2))  # 2
    print(slt.minOperations([2, 1, 2], 2))  # -1
    print(slt.minOperations([9, 7, 5, 3], 1))  # 4

# Done ✅
