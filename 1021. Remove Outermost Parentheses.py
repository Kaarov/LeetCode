class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        ans = ""
        right = 0

        for i in s:
            if i == '(':
                right += 1
            elif i == ')' and right == 1:
                right -= 1
                continue

            if i == '(' and right >= 2:
                ans += i
            elif i == ')' and right >= 2:
                ans += i
                right -= 1

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.removeOuterParentheses("(()())(())"))  # "()()()"
    print(slt.removeOuterParentheses("(()())(())(()(()))"))  # "()()()()(())"
    print(slt.removeOuterParentheses("()()"))  # ""

# Done ✅
