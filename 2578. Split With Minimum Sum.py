class Solution:
    def splitNum(self, num: int) -> int:
        num1 = ""
        num2 = ""
        flag = True

        num = sorted(list(map(int, str(num))), reverse=True)

        while num:
            n = num.pop()

            if flag:
                num1 += str(n)
            else:
                num2 += str(n)

            flag = not flag

        return int(num1) + int(num2)


if __name__ == "__main__":
    slt = Solution()
    print(slt.splitNum(4325))  # 59
    print(slt.splitNum(687))  # 75

# Done ✅
