class Solution:
    def recurse(self, nums):
        if len(nums) <= 2: return nums
        return self.recurse(nums[::2]) + self.recurse(nums[1::2])

    def beautifulArray(self, n: int) -> list[int]:
        return self.recurse([i for i in range(1, n + 1)])


if __name__ == "__main__":
    slt = Solution()
    assert slt.beautifulArray(4) == [1, 3, 2, 4]
    assert slt.beautifulArray(5) == [1, 5, 3, 2, 4]
