# strs = ["dog", "racecar", "car"]
strs = ["flower", "flow", "flight"]


class Solution:
    def longestCommonPrefix(self, strs):
        longest = ''
        for x in strs:
            if len(x) > len(longest):
                longest = x
        strs.remove(longest)
        ans = longest
        for x in strs:
            check = ''
            for y in range(len(ans)):
                if len(x) == y:
                    break
                elif ans[y] == x[y]:
                    check += ans[y]
                else:
                    break
            ans = check
        return ans

# Second Way
# def longestCommonPrefix(strs):
#     if not strs:
#         return ""
#     shortest = min(strs, key=len)
#     for x, ch in enumerate(shortest):
#         for other in strs:
#             if other[x] != ch:
#                 return shortest[:x]
#     return shortest

# Done✅
