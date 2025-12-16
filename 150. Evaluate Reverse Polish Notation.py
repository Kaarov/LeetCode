from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        operations = {'+', '-', '*', '/'}
        stack = []
        for token in tokens:
            if token in operations:
                a = stack.pop()
                b = stack.pop()
                result = int(eval(f"{b}{token}{a}"))
                stack.append(result)
            else:
                stack.append(token)
        return int(stack.pop())


if __name__ == '__main__':
    slt = Solution()
    assert slt.evalRPN(["2", "1", "+", "3", "*"]) == 9
    assert slt.evalRPN(["4", "13", "5", "/", "+"]) == 6

# Done ✅
