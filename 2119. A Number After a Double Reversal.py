class Solution:
    def isSameAfterReversals(self, num: int) -> bool:
        reversed1 = int(str(num)[::-1])
        reversed2 = int(str(reversed1)[::-1])

        return num == reversed2


if __name__ == '__main__':
    slt = Solution()

    print(slt.isSameAfterReversals(526))  # True
    print(slt.isSameAfterReversals(1800))  # False
    print(slt.isSameAfterReversals(0))  # True

# Done ✅
