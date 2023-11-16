string = "hello"


class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        s = list(s)

        vowelInS = [i for i in s if i in vowels]

        for i in range(len(s)):
            if s[i] in vowels:
                s[i] = vowelInS.pop()

        return ''.join(s)


slt = Solution()
print(slt.reverseVowels(string))

# Done ✅
