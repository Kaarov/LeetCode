import math


class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        square_root = set()

        for i in range(int(math.sqrt(c)) + 1):
            square_root.add(i * i)

        a = 0
        while a * a <= c:
            if c - a * a in square_root:
                return True
            a += 1

        return False


if __name__ == "__main__":
    slt = Solution()
    assert slt.judgeSquareSum(5) == True
    assert slt.judgeSquareSum(3) == False

# Done ✅
# Note: Could be improved further
