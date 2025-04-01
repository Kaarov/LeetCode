from typing import List


class Solution:
    def unequalTriplets(self, nums: List[int]) -> int:
        ans = 0
        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums) - 1):
                for k in range(j + 1, len(nums)):
                    if nums[i] != nums[j] and nums[i] != nums[k] and nums[j] != nums[k]:
                        ans += 1
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.unequalTriplets([4, 4, 2, 4, 3]))  # 3
    print(slt.unequalTriplets([1, 1, 1, 1, 1]))  # 0

# Done ✅
