number = 38


class Solution:
    def addDigits(self, num: int) -> int:
        if num < 10:
            return num
        while num > 9:
            ans = 0
            for i in str(num):
                ans += int(i)
            num = ans
        return num


slt = Solution()
print(slt.addDigits(number))

# Done ✅
