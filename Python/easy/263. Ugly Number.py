# number = 14
# number = 6
number = -2147483648


class Solution:
    def isUgly(self, n: int) -> bool:
        if n == 1:
            return True

        while n > 1 or n < -1:
            if n % 2 == 0:
                n /= 2
            elif n % 3 == 0:
                n /= 3
            elif n % 5 == 0:
                n /= 5
            else:
                return False
        return True


slt = Solution()
print(slt.isUgly(number))
