class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        idx_first = -1
        idx_second = -1

        if len(s) != len(goal):
            return False

        for i in range(len(s)):
            if s[i] != goal[i]:
                if idx_first == -1:
                    idx_first = i
                else:
                    idx_second = i
                    break
        ans = list(s)
        ans[idx_first], ans[idx_second] = ans[idx_second], ans[idx_first]
        ans = "".join(ans)

        if ans == goal:
            if len(set(ans)) != len(s):
                return True
            if idx_first != -1 and idx_second != -1:
                return True
        return False


if __name__ == '__main__':
    slt = Solution()
    print(slt.buddyStrings(s="ab", goal="ba"))  # True
    print(slt.buddyStrings(s="ab", goal="ab"))  # False
    print(slt.buddyStrings(s="aa", goal="aa"))  # True
    print(slt.buddyStrings(s="abcaa", goal="abcbb"))  # False

# Done ✅
