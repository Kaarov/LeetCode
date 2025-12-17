class Solution:
    def buildArray(self, target: list[int], n: int) -> list[str]:
        stack = []
        ans = []
        count = 1
        index = 0

        while stack != target:
            stack.append(count)
            ans.append("Push")
            if stack[index] != target[index]:
                stack.pop()
                ans.append("Pop")
            else:
                index += 1
            count += 1

        return ans


if __name__ == '__main__':
    slt = Solution()
    assert slt.buildArray([1, 3], 3) == ["Push", "Push", "Pop", "Push"]
    assert slt.buildArray([1, 2, 3], 3) == ["Push", "Push", "Push"]
    assert slt.buildArray([1, 2], 4) == ["Push", "Push"]

# Done ✅
