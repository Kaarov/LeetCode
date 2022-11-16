number = 28


class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        ans = 0
        count = 1
        while ans < num:
            if num % count == 0:
                ans += count
            count += 1
        if ans == num:
            return True
        return False


slt = Solution()
print(slt.checkPerfectNumber(number))
