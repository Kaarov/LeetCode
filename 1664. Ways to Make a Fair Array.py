class Solution:
    def waysToMakeFair(self, nums: list[int]) -> int:
        s1, s2 = [0, 0], [sum(nums[0::2]), sum(nums[1::2])]
        ans = 0
        for i, num in enumerate(nums):
            s2[i % 2] -= num
            ans += s1[0] + s2[1] == s1[1] + s2[0]
            s1[i % 2] += num
        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.waysToMakeFair([2, 1, 6, 4]) == 1
    assert slt.waysToMakeFair([1, 1, 1]) == 3
    assert slt.waysToMakeFair([1, 2, 3]) == 0
