s = "leet**cod*e"


class Solution:
    def removeStars(self, s: str) -> str:
        ans = []
        for i in s:
            if i == "*":
                ans.pop()
            else:
                ans.append(i)
        return "".join(ans)


slt = Solution()
print(slt.removeStars(s))

# Done ✅
