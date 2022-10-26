# list1 = [2, 4, 3]
# list2 = [5, 6, 4]

list1 = [9, 9, 9, 9, 9, 9, 9]
list2 = [9, 9, 9, 9]


class Solution:
    def addTwoNumbers(self, l1, l2):
        rl1 = ''
        for x in range(len(l1) - 1, -1, -1):
            rl1 += str(l1[x])
        rl2 = ''
        for x in range(len(l2) - 1, -1, -1):
            rl2 += str(l2[x])
        result = str(int(rl1) + int(rl2))
        ans = []
        for x in range(len(result) - 1, -1, -1):
            ans.append(int(result[x]))
        return ans


s = Solution()
print(s.addTwoNumbers(list1, list2))

# Not Done Yet ❌
