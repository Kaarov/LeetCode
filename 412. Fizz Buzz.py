class Solution:
    def fizzBuzz(self, n: int) -> list[str]:
        ans = []
        for i in range(1, n + 1):
            if not i % 3 and not i % 5:
                ans.append("FizzBuzz")
            elif not i % 3:
                ans.append("Fizz")
            elif not i % 5:
                ans.append("Buzz")
            else:
                ans.append(str(i))

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.fizzBuzz(3))  # ["1", "2", "Fizz"]
    print(slt.fizzBuzz(5))  # ["1", "2", "Fizz", "4", "Buzz"]
    print(
        slt.fizzBuzz(15)
    )  # ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]

# Done ✅
