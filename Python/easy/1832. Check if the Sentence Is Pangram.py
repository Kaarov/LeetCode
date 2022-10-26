# text = "thequickbrownfoxjumpsoverthelazydog"
text = "leetcode"


class Solution:
    def checkIfPangram(self, sentence):
        for i in range(97, 123):
            if chr(i) not in sentence:
                return False
        return True


test = Solution()
print(test.checkIfPangram(text))

# Done ✅
