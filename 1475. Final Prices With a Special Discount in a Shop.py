class Solution:
    def finalPrices(self, prices: list[int]) -> list[int]:
        for i in range(len(prices) - 1):
            for j in range(i, len(prices)):
                if j > i and prices[i] >= prices[j]:
                    prices[i] -= prices[j]
                    break
        return prices


if __name__ == "__main__":
    slt = Solution()
    assert slt.finalPrices([8, 4, 6, 2, 3]) == [4, 2, 4, 2, 3]
    assert slt.finalPrices([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert slt.finalPrices([10, 1, 1, 6]) == [9, 0, 1, 6]

# Done ✅
