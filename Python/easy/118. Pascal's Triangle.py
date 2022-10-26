# test = 5
test = 5


class Solution:
    def generate(self, numRows):
        ans = [[1], [1, 1]]
        if numRows == 1:
            return [[1]]
        elif numRows == 2:
            return ans
        elif numRows == 0:
            return []

        for x in range(numRows-2):
            check = ans[-1]
            result = []
            for y in range(len(check)+1):
                if y == 0 or y == len(check):
                    result.append(1)
                else:
                    result.append(check[y-1]+check[y])
            ans.append(result)
        return ans


s = Solution()
print(s.generate(test))

# Done ✅
