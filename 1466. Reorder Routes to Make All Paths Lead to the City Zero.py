from typing import List


class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        edges = {(a, b) for a, b in connections}
        neighbors = {city: [] for city in range(n)}
        visited = set()
        changes = 0
        for a, b in connections:
            neighbors[a].append(b)
            neighbors[b].append(a)

        def dfs(city):
            nonlocal edges, neighbors, visited, changes

            for neighbor in neighbors[city]:
                if neighbor in visited:
                    continue
                if (neighbor, city) not in edges:
                    changes += 1
                visited.add(neighbor)
                dfs(neighbor)

        visited.add(0)
        dfs(0)
        return changes


slt = Solution()
n = 6
connections = [[0, 1], [1, 3], [2, 3], [4, 0], [4, 5]]
print(slt.minReorder(n, connections))
