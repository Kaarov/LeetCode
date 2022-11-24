words_list = ["Hello", "Alaska", "Dad", "Peace"]


def row(word, rows):
    for i in word:
        if i.lower() not in rows:
            return False
    return True


class Solution:
    def findWords(self, words):
        row1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
        row2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
        row3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
        ans = []

        for i in words:
            if i[0].lower() in row1:
                if row(i, row1):
                    ans.append(i)
            elif i[0].lower() in row2:
                if row(i, row2):
                    ans.append(i)
            else:
                if row(i, row3):
                    ans.append(i)
        return ans


slt = Solution()
print(slt.findWords(words_list))

# Done ✅
