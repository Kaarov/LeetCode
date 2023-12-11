from typing import List

isConnected = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def explore(graph, current, visited):
            if current in visited: return False

            visited.add(current)

            for neighbor in graph[current]:
                explore(graph, neighbor, visited)

            return True

        graph = {}
        n = len(isConnected)
        for i in range(n):
            graph[i + 1] = []
            for j in range(n):
                if i != j and isConnected[i][j] == 1:
                    graph[i + 1].append(j + 1)

        visited = set()
        count = 0
        for node in graph:
            if explore(graph, node, visited) == True:
                count += 1

        return count


slt = Solution()
print(slt.findCircleNum(isConnected))

# Done ✅
