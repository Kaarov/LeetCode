from typing import List

flowerbed = [1, 0, 1, 0, 1, 0, 1]
n = 0


class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if n == 0:
            return True
        for i in range(len(flowerbed)):
            if (flowerbed[i] == 0 and (i == 0 or flowerbed[i - 1] == 0)
                    and (flowerbed[i + 1] == 0) or i == len(flowerbed) - 1):
                flowerbed[i] = 1
                n -= 1
                if n == 0:
                    return True
        return False


slt = Solution()
print(slt.canPlaceFlowers(flowerbed, n))

# Done ✅
