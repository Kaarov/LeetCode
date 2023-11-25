# s = "3[a]2[bc]"
s = "3[a2[c]]"


class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for i in s:
            if i == ']':
                enc = ''
                while stack[-1] != '[':
                    enc = stack.pop() + enc
                stack.pop()

                k = ''
                while stack and stack[-1].isdigit():
                    k = stack.pop() + k

                stack.append(enc * int(k))
            else:
                stack.append(i)

        return ''.join(stack)


slt = Solution()
print(slt.decodeString(s))
