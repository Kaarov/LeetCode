# pr = [7, 1, 5, 3, 6, 4]
# pr = [7, 6, 4, 3, 1]
pr = [2, 4, 1]


class Solution:
    def maxProfit(self, prices):
        max_number = min(prices)
        min_number = max(prices)
        max_i = 0
        min_i = 0

        for i in range(len(prices)):
            index = prices[i]
            if min_number > index:
                min_number = index
                min_i = i
                max_number = index
                max_i = i
            if max_number < index:
                max_number = index
                max_i = i
        if max_i > min_i:
            return max_number - min_number
        else:
            return 0


test = Solution()
print(test.maxProfit(pr))
