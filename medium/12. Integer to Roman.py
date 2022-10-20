# number = 58
number = 2994

# roman = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L', 90: 'XC', 100: 'C',
#          400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'}


class Solution:
    def intToRoman(self, num):
        roman = {1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX',
                 5: 'V', 4: 'IV', 1: 'I'}
        ans = ''

        for key, val in roman.items():
            while num >= key:
                num = num - key
                ans = ans + val
        return ans


slt = Solution()
print(slt.intToRoman(number))

# Done ✅
