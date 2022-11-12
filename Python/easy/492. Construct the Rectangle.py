a = 4


class Solution:
    def constructRectangle(self, area):
        ans = int(area ** 0.5)
        for l in range(ans, 0, -1):
            if area % l == 0:
                return [area // l, l]


slt = Solution()
print(slt.constructRectangle(a))
