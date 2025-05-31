from typing import List


class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        ans = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] < target:
                    ans += 1

        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.countPairs(nums=[-1, 1, 2, 3, 1], target=2))  # 3
    print(slt.countPairs(nums=[-6, 2, 5, -2, -7, -1, 3], target=-2))  # 10

# Done ✅
