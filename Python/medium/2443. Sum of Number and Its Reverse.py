number = 443


class Solution:
    def sumOfNumberAndReverse(self, num: int) -> bool:
        for i in range(num + 1):
            ans = int(str(i)[::-1]) + i
            if ans == num:
                return True
        return False


slt = Solution()
print(slt.sumOfNumberAndReverse(number))

# Done ✅
