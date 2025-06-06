class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num <= 1:
            return False

        divisors = [1]

        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                divisors.append(i)
                if i != num // i:
                    divisors.append(num // i)

        return sum(divisors) == num


if __name__ == "__main__":
    slt = Solution()
    print(slt.checkPerfectNumber(28))  # True
    print(slt.checkPerfectNumber(7))  # False
    print(slt.checkPerfectNumber(120))  # False
