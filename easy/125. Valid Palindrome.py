# check = "A man, a plan, a canal: Panama"
check = "race a car"


class Solution:
    def isPalindrome(self, s):
        result = ''
        punc = [' ', ',', '.', ':', ';', '?', '!', "'", '@', '#', '$', '%', '(', ')', '-', '_', '+', '=', '\\', '/',
                '[', ']', '{', '}', '|', '>', '<', '`', '~', '"']
        for x in s:
            if x not in punc:
                result += x
        result = result.lower()
        if result == result[::-1]:
            return True
        else:
            return False


a = Solution()
print(a.isPalindrome(check))

# Done ✅
