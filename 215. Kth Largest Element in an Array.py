from random import choice


class Solution:
    # version 1
    def findKthLargest(self, nums: list[int], k: int) -> int:
        nums.sort(reverse=True)
        return nums[k - 1]

    # version 2
    # def findKthLargest(self, nums: list[int], k: int) -> int:
    #     p, l, m, r = choice(nums), [], [], []
    #     for v in nums: (m, l, r)[(p < v) - (v < p)].append(v)
    #     if k <= len(l): return self.findKthLargest(l, k)
    #     if len(l) + len(m) < k: return self.findKthLargest(r, k - len(l) - len(m))
    #     return p


if __name__ == "__main__":
    slt = Solution()
    assert slt.findKthLargest([3, 2, 1, 5, 6, 4], 2) == 5
    assert slt.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
