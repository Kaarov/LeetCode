number = 5


class Solution:
    def trailingZeroes(self, n: int) -> int:
        fact = 1
        for i in range(1, n+1):
            fact *= i
        ans = 0
        str_n = str(fact)[::-1]
        for i in str_n:
            if i == "0":
                ans += 1
            else:
                break
        return ans


slt = Solution()
print(slt.trailingZeroes(number))

# Done ✅
