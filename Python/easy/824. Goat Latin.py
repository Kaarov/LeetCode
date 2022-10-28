sentence = "I speak Goat Latin"


class Solution:
    def toGoatLatin(self, s):
        vowel = ['a', 'e', 'i', 'o', 'u']
        ans = []
        s = s.split()
        for i in range(len(s)):
            if s[i][0].lower() in vowel:
                ans.append(s[i] + "ma" + ("a"*(i+1)))
            else:
                ans.append(s[i][1:] + s[i][0] + "ma" + ("a"*(i+1)))
        ans = " ".join(ans)
        return ans


slt = Solution()
print(slt.toGoatLatin(sentence))

# Done ✅
