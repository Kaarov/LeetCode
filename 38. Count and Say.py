class Solution:
    def countAndSay(self, n: int) -> str:
        ans = "1"

        def recursion(s: str) -> str:
            result = ""

            count = 1
            for i in range(len(s)):
                if i == len(s) - 1:
                    result += str(count) + s[i] if count != 1 else "1" + s[i]
                    continue
                if s[i] == s[i + 1]:
                    count += 1
                else:
                    result += str(count) + s[i]
                    count = 1
            return result

        for _ in range(n - 1):
            ans = recursion(ans)

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.countAndSay(4))  # "1211"
    print(slt.countAndSay(1))  # 1

# Done ✅
