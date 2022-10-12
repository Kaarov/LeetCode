# sii = "egg"
# tii = "add"
# sii = "aaaaabbbbbcccccdddddeeeee"
# tii = "pppppqqqqqrrrrrsssssttttt"
sii = "aa"
tii = "bb"


class Solution:
    def isIsomorphic(self, s, t):
        s_d = dict()
        t_d = dict()
        ans_s = ""
        ans_t = ""
        nextchr = 'a'
        for i in range(len(s)):
            if s[i] not in s_d:
                s_d[s[i]] = nextchr
            if t[i] not in t_d:
                t_d[t[i]] = nextchr
                nextchr = chr(ord(nextchr) + 1)
            ans_s += s_d[s[i]]
            ans_t += t_d[t[i]]

        return ans_s == ans_t


test = Solution()
print(test.isIsomorphic(sii, tii))

# Done ✅
