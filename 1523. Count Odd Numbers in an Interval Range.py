class Solution:
    def countOdds(self, low: int, high: int) -> int:
        length = high - low + 1
        return (length // 2) + (low % 2 and high % 2)


slt = Solution()
print(slt.countOdds(3, 7))

# Done ✅
