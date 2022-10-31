number = 4


class Solution:
    def countAndSay(self, n):
        if n == 1:
            return "1"

        def countandsay(num):
            num += "1"
            total = 0
            ans = ""
            count = 0
            number_count = 0
            while num:
                index = num[0]
                if index == num[count]:
                    number_count += 1
                    count += 1
                else:
                    ans += number_count
                    ans += index
                    count, number_count = 0, 0
                    num.replace(index, "")
            total += 1
            if total == n:
                return ans
            else:
                return countandsay(ans)

        countandsay("1")

solution = Solution()
print(solution.countAndSay(number))
