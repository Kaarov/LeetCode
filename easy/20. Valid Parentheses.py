# s = "([)]"
s = "){"


class Solution:
    def isValid(self, s):
        result = []
        check = [')', '}', ']']
        if len(s) <= 1:
            return False
        while s:
            if s[0] in check:
                if result:
                    if result[-1] == '(' and s[0] == check[0] or result[-1] == '{' and s[0] == check[1] or \
                            result[-1] == '[' and s[0] == check[2]:
                        result.pop()
                    else:
                        return False
                else:
                    return False
            elif s[0] in ['(', '{', '[']:
                result.append(s[0])
            s = s[1:]
        if not result:
            return True
        else:
            return False


print(Solution().isValid(s))

# Not Done Yet ✅
