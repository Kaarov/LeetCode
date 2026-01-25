class Solution:
    def minSubarray(self, nums: list[int], p: int) -> int:
        total = sum(nums)
        remain = total % p

        if remain == 0:
            return 0

        ans = len(nums)
        curr_sum = 0

        remain_to_idx = {
            0: -1
        }

        for i, n in enumerate(nums):
            curr_sum = (curr_sum + n) % p
            prefix = (curr_sum - remain + p) % p

            if prefix in remain_to_idx:
                length = i - remain_to_idx[prefix]
                ans = min(ans, length)
            remain_to_idx[curr_sum] = i

        return -1 if ans == len(nums) else ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.minSubarray([3, 1, 4, 2], 6) == 1
    assert slt.minSubarray([6, 3, 5, 2], 9) == 2
    assert slt.minSubarray([1, 2, 3], 3) == 0
