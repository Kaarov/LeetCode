from collections import defaultdict
from typing import List

equations = [["a", "b"], ["b", "c"]]
values = [2.0, 3.0]
queries = [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]


class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = defaultdict(dict)

        for i in range(len(equations)):
            a, b = equations[i][0], equations[i][1]
            graph[a][b] = values[i]
            graph[b][a] = 1 / values[i]

        def dfs(start, end, res):
            if start in seen:
                return -1.0

            if start == end:
                return res
            seen.add(start)

            for n in graph[start]:
                if n not in seen:
                    tmp = dfs(n, end, res * graph[start][n])
                    if tmp != -1.0:
                        return tmp

            return -1.0

        res = []
        for a, b in queries:
            if a not in graph or b not in graph:
                res.append(-1.0)
            else:
                seen = set()
                res.append(dfs(a, b, 1))

        return res


slt = Solution()
print(slt.calcEquation(equations, values, queries))
