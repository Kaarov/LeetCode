number = 3


class Solution:
    def divisorGame(self, n: int) -> bool:
        if n % 2 == 0:
            return True
        return False


slt = Solution()
print(slt.divisorGame(number))

# Done ✅
