from typing import List

rooms = [[1], [2], [3], []]


class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = []
        ans = []
        stack = [rooms[0]]

        while stack:
            current = stack.pop()
            for i in current:
                if i not in ans:
                    ans.append(i)
                if i not in visited:
                    visited.append(i)
                    stack.append(rooms[i])
        if 0 not in ans: ans.append(0)
        return len(ans) == len(rooms)


slt = Solution()
print(slt.canVisitAllRooms(rooms))

# Done ✅
