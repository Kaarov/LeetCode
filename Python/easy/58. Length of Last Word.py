# a = "   fly me   to   the moon  "
# a = "luffy is still joyboy"
# a = ' a'
a = 'b a '


class Solution:
    def lengthOfLastWord(self, s):
        ans = ''
        result = []
        for x in range(len(s)):
            if s[x] == ' ':
                pass
            elif s[x] != ' ' and x == len(s)-1:
                ans += s[x]
                result.append(ans)
                ans = ''
            elif s[x] != ' ' and s[x+1] == ' ':
                ans += s[x]
                result.append(ans)
                ans = ''
            elif s[x-1] == ' ' and s[x] != ' ':
                ans += s[x]
            else:
                ans += s[x]

        if result:
            return len(result[-1])
        else:
            return len(ans)


c = Solution()
print(c.lengthOfLastWord(a))
# Done ✅
