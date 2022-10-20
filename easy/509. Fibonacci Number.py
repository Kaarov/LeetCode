number = 4


class Solution:
    def fib(self, n):
        # F(n) = F(n - 1) + F(n - 2),

        def f(num):
            if num <= 1:
                return num
            else:
                return f(num-1) + f(num-2)

        return f(n)


slt = Solution()
print(slt.fib(number))

# Done ✅
