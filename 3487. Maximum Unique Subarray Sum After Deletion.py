from typing import List


class Solution:
    def maxSum(self, nums: List[int]) -> int:
        nums = list(set(nums))
        ans = []
        neg = []

        for i in nums:
            if i < 0:
                neg.append(i)
            else:
                ans.append(i)

        return sum(ans) if ans != [] else max(neg)


if __name__ == "__main__":
    slt = Solution()
    print(slt.maxSum([1, 2, 3, 4, 5]))  # 15
    print(slt.maxSum([1, 1, 0, 1, 1]))  # 1
    print(slt.maxSum([1, 2, -1, -2, 1, 0, -1]))  # 3
    print(slt.maxSum([-100]))  # -100
    print(slt.maxSum([-17, -15]))  # -15

# Done ✅
