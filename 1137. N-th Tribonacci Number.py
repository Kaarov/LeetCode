n = 4


class Solution:
    def tribonacci(self, n: int) -> int:
        array = [0, 1, 1]
        for i in range(3, n):
            array.append(sum(array))
            array.remove(array[0])
        return sum(array) if n > 2 else array[n]


slt = Solution()
print(slt.tribonacci(n))

# Done ✅
