from typing import List


class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        ans: int = 0

        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j] and (i * j) % k == 0:
                    ans += 1

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.countPairs(nums=[3, 1, 2, 2, 2, 1, 3], k=2))  # 4
    print(slt.countPairs(nums=[1, 2, 3, 4], k=1))  # 0

# Done ✅
