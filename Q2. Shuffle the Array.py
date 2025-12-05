class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        ans = []
        for i in range(n):
            ans.extend([nums[i], nums[i + n]])
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.shuffle([2, 5, 1, 3, 4, 7], 3))  # [2, 3, 5, 4, 1, 7]
    print(slt.shuffle([1, 2, 3, 4, 4, 3, 2, 1], 4))  # [1, 4, 2, 3, 3, 2, 4, 1]
    print(slt.shuffle([1, 1, 2, 2], 2))  # [1, 2, 1, 2]

# Done ✅
