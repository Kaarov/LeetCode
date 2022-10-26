# word = ["i", "love", "leetcode", "i", "love", "coding"]
# kl = 2
# word = ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
# kl = 4
# word = ["i", "love", "leetcode", "i", "love", "coding"]
# kl = 1
word = ["i", "love", "leetcode", "i", "love", "coding"]
kl = 3


class Solution:
    def topKFrequent(self, words, k):
        word_dict = dict()
        check = []
        count = 0
        ans = []
        for i in words:
            if i in word_dict:
                word_dict[i] += 1
            else:
                word_dict[i] = 1
        for i in word_dict.values():
            if i not in check:
                check.append(i)
        check.sort(reverse=True)
        word_dict = dict(sorted(word_dict.items()))
        while count < k:
            for key, value in word_dict.items():
                if value == check[0]:
                    ans.append(key)
                    count += 1
                if count >= k:
                    break
            check.pop(0)
        return ans


slt = Solution()
print(slt.topKFrequent(word, kl))

# Done ✅
