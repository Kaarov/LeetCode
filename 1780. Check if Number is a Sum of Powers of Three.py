class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        ans = []
        while n > 0:
            count = 0
            while 3 ** count <= n and count not in ans:
                count += 1
            if count == 0 and count in ans:
                break
            count -= 1
            ans.append(count)
            n -= 3 ** count
        if n == 0:
            return True
        return False


if __name__ == '__main__':
    slt = Solution()
    print(slt.checkPowersOfThree(12))  # True
    print(slt.checkPowersOfThree(91))  # True
    print(slt.checkPowersOfThree(21))  # False
    print(slt.checkPowersOfThree(4672715))  # False

# Done ✅
