pow_x = 2.00000
pow_n = 10


class Solution:
    def myPow(self, x: float, n: int) -> float:
        result = x ** n
        return result


s = Solution()
print(s.myPow(pow_x, pow_n))

# Done ✅
