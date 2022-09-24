# list1 = [2, 4, 3]
# list2 = [5, 6, 4]

list1 = [9, 9, 9, 9, 9, 9, 9]
list2 = [9, 9, 9, 9]


class Solution:
    def addTwoNumbers(self, l1, l2):
        l1.reverse()
        l2.reverse()

        l1_int = ''
        l2_int = ''

        for x in l1:
            l1_int += str(x)

        for x in l2:
            l2_int += str(x)

        l1_int = int(l1_int)
        l2_int = int(l2_int)

        result = str(l1_int + l2_int)
        ans = []

        for x in result:
            ans.append(int(x))

        ans.reverse()
        return ans


s = Solution()
print(s.addTwoNumbers(list1, list2))

# Not Done Yet ❌
