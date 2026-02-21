# Algorithm Compendium

A comprehensive reference intended to mirror the structure and clarity of a university-level algorithms textbook. This compendium organizes essential algorithmic knowledge into coherent themes, outlining the motivation, core concepts, and typical applications for each area. Use this index as a map: start with the high-level survey below, then jump directly to any topic of interest via the linked sections.

---

## Table of Contents

### Part I: Algorithm Design Paradigms
1. [Algorithm Design Paradigms](#algorithm-design-paradigms)
2. [Dynamic Programming](#dynamic-programming)
3. [Greedy Algorithms](#greedy-algorithms)
4. [Probabilistic and Approximation Algorithms](#probabilistic-and-approximation-algorithms)

### Part II: Core Algorithms
5. [Binary Search Algorithm](#binary-search-algorithm)
6. [Sorting Algorithms](#sorting-algorithms)
7. [Searching Algorithms](#searching-algorithms)
8. [String Processing and Pattern Matching](#string-processing-and-pattern-matching)
9. [Array Algorithms](#array-algorithms)
10. [Graph Algorithms](#graph-algorithms)

### Part III: Data Structures
11. [Fundamental Data Structures](#fundamental-data-structures)

### Part IV: Specialized Domains
12. [Numerical and Scientific Algorithms](#numerical-and-scientific-algorithms)
13. [Optimization Techniques](#optimization-techniques)
14. [Machine Learning and Data Analysis Algorithms](#machine-learning-and-data-analysis-algorithms)
15. [Cryptographic Algorithms](#cryptographic-algorithms)
16. [Data Compression Algorithms](#data-compression-algorithms)
17. [Computational Geometry Algorithms](#computational-geometry-algorithms)
18. [Parallel and Distributed Algorithms](#parallel-and-distributed-algorithms)
19. [Constraint Solving and Logic-Based Algorithms](#constraint-solving-and-logic-based-algorithms)
20. [Specialized Application Algorithms](#specialized-application-algorithms)

### Part V: Practice Problems
21. [Solved Problems Index](#solved-problems-index)

### Part VI: Resources
22. [Further Reading and Study Resources](#further-reading-and-study-resources)

---

## Binary Search Algorithm

**Binary search** finds a target value in a **sorted** array (or in a range with a **monotonic** condition) by repeatedly comparing the target to the middle element and discarding the half where it cannot lie. Each step halves the search space, giving **O(log n)** time and **O(1)** space (iterative) or **O(log n)** space (recursive).

### How it works

1. Maintain a range `[left, right]` that is guaranteed to contain the answer (or the insertion point).
2. Compute `mid = left + (right - left) // 2` (avoids overflow in other languages; in Python `(left + right) // 2` is fine).
3. Compare `arr[mid]` with the target and shrink the range:
   - If `arr[mid] == target` → found (for “find exact”).
   - If `arr[mid] < target` → search right: `left = mid + 1`.
   - If `arr[mid] > target` → search left: `right = mid - 1` (or `right = mid` when using “first position ≥ target” style).
4. Stop when the range is empty or reduced to one position, depending on the variant.

**Loop invariant (classic “find target”):** If `target` is in the array, then it lies in `[left, right]`. When `left > right`, the target is not present.

### When to use

- The array (or the **value space** you’re searching) is **sorted** or has a **monotonic** property (e.g. “first index where condition becomes true”).
- Typical tasks:
  - Find an exact value.
  - Find **insertion index** (where to insert to keep order).
  - Find **first** or **last** occurrence of a value.
  - **Binary search on the answer**: find the smallest (or largest) value in a range such that a condition holds (e.g. minimum capacity, maximum day, etc.).

### 1. Classic iterative — find exact index

Returns the index of `target` if present, otherwise `-1`.

```python
def binary_search(arr, target):
    """Return index of target in sorted arr, or -1 if not found."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Examples
arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search(arr, 7))   # 3
print(binary_search(arr, 8))  # -1
print(binary_search(arr, 1))   # 0
print(binary_search(arr, 15)) # 7
```

### 2. Recursive version

Same contract as above, implemented recursively. Space O(log n) due to call stack.

```python
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    return binary_search_recursive(arr, target, left, mid - 1)

# Example
arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search_recursive(arr, 9))  # 4
```

### 3. Lower bound — first index where arr[i] >= target

Use when you need the **leftmost** position for `target` (or the insertion point if target is missing). Uses `right = mid` (not `mid - 1`) and typically `right = len(arr)` so that “insert at end” is representable.

```python
def lower_bound(arr, target):
    """Smallest index i such that arr[i] >= target. If all < target, returns len(arr)."""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

# Examples
arr = [1, 2, 2, 2, 3, 4, 5]
print(lower_bound(arr, 2))   # 1  (first 2)
print(lower_bound(arr, 0))   # 0
print(lower_bound(arr, 6))   # 7  (insert at end)
print(lower_bound(arr, 2.5)) # 4  (insert between 2 and 3)
```

### 4. Upper bound — first index where arr[i] > target

Smallest index such that `arr[i] > target`. Useful for ranges: count of `target` = `upper_bound(arr, target) - lower_bound(arr, target)`.

```python
def upper_bound(arr, target):
    """Smallest index i such that arr[i] > target. If all <= target, returns len(arr)."""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left

# Examples
arr = [1, 2, 2, 2, 3, 4, 5]
print(upper_bound(arr, 2))   # 4  (first index > 2)
print(upper_bound(arr, 5))   # 7
# Count of 2: upper_bound(arr, 2) - lower_bound(arr, 2) == 4 - 1 == 3
```

### 5. Search insert position

Same as lower bound: index at which to insert `target` to keep non-decreasing order.

```python
def search_insert(nums, target):
    """Index where target should be inserted to keep nums sorted."""
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

# Examples
nums = [1, 3, 5, 6]
print(search_insert(nums, 5))  # 2
print(search_insert(nums, 2))  # 1
print(search_insert(nums, 7))  # 4
print(search_insert(nums, 0))  # 0
```

### 6. Binary search on the answer (find minimum valid value)

Search over a **range of values** (e.g. integers) to find the **smallest** value for which a condition is true. Example: “minimum capacity such that we can ship all packages in D days.”

```python
def can_ship(weights, capacity, max_days):
    """True if we can ship all weights with given capacity within max_days."""
    days = 1
    current = 0
    for w in weights:
        if current + w <= capacity:
            current += w
        else:
            days += 1
            current = w
            if days > max_days:
                return False
    return True

def min_capacity_to_ship(weights, days):
    """Minimum capacity so that shipping takes <= days. Binary search on capacity."""
    low = max(weights)
    high = sum(weights)
    while low < high:
        mid = (low + high) // 2
        if can_ship(weights, mid, days):
            high = mid
        else:
            low = mid + 1
    return low

# Example: weights = [1,2,3,4,5,6,7,8,9,10], days = 5 → need capacity at least 15
print(min_capacity_to_ship([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5))  # 15
```

### 7. Find peak in a “mountain” array

Array increases then decreases. Find any peak index (binary search on the “slope”).

```python
def find_peak_index(arr):
    """Peak index in mountain array (arr increases then decreases)."""
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left

# Example: [1, 2, 3, 1] -> peak at index 2
print(find_peak_index([1, 2, 3, 1]))       # 2
print(find_peak_index([1, 3, 5, 4, 2]))  # 2
```

### 8. Find the smallest value in a rotated sorted array

Array is sorted then rotated (e.g. `[4,5,6,7,0,1,2]`). Find the minimum element.

```python
def find_min_rotated(arr):
    """Minimum element in rotated sorted array (no duplicates)."""
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    return arr[left]

# Example
print(find_min_rotated([4, 5, 6, 7, 0, 1, 2]))  # 0
print(find_min_rotated([3, 1, 2]))               # 1
```

### Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Bounds** | For “insertion / lower bound” style use `right = len(arr)` and `while left < right`; for “find exact” use `right = len(arr)-1` and `while left <= right`. |
| **Mid** | Prefer `mid = left + (right - left) // 2` in languages where `left + right` can overflow; in Python `(left + right) // 2` is fine. |
| **Shrinking** | In “first position ≥ target” variants use `right = mid` (keep mid in range); in “find exact” use `right = mid - 1`. |
| **Duplicates** | Use lower_bound for first occurrence, upper_bound for last (or “one past last”); count = upper_bound − lower_bound. |

### Related sections and problems

- More search techniques: [Searching Algorithms](#searching-algorithms).
- Solved problems tagged Binary Search: see [Solved Problems Index](#solved-problems-index) (e.g. Find Peak Element, Koko Eating Bananas, Search Insert Position).

---

## Algorithm Design Paradigms

An overview of the foundational strategies used to conceive algorithms, including divide-and-conquer, dynamic programming, greedy methods, backtracking, and probabilistic techniques. This section clarifies when and why a paradigm is appropriate, highlighting the trade-offs in complexity, implementation effort, and optimality guarantees.

### Key Concepts
- Divide and Conquer
- Backtracking
- Recursion
- Memoization
- Tabulation

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Sorting Algorithms

Introduces the landscape of sorting, from comparison-based methods (e.g., merge sort, quicksort) to non-comparison approaches (counting, radix). Emphasizes stability, in-place operation, and asymptotic behavior, setting the stage for deeper dives into algorithmic design choices.

### Key Concepts
- Comparison-based sorting (Quick Sort, Merge Sort, Heap Sort)
- Non-comparison sorting (Counting Sort, Radix Sort, Bucket Sort)
- Stability and in-place operations
- Time complexity analysis

### Python Examples

#### 1. Quick Sort
```python
def quicksort(arr):
    """Quick sort algorithm - O(n log n) average, O(n²) worst case."""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Example: Sort array
arr = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(arr))  # [1, 1, 2, 3, 6, 8, 10]
```

#### 2. Merge Sort
```python
def merge_sort(arr):
    """Merge sort algorithm - O(n log n) guaranteed."""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example: Merge sort
arr = [3, 6, 8, 10, 1, 2, 1]
print(merge_sort(arr))  # [1, 1, 2, 3, 6, 8, 10]
```

#### 3. Heap Sort
```python
def heapify(arr, n, i):
    """Heapify subtree rooted at index i."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    """Heap sort algorithm - O(n log n)."""
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    
    return arr

# Example: Heap sort
arr = [12, 11, 13, 5, 6, 7]
print(heap_sort(arr.copy()))  # [5, 6, 7, 11, 12, 13]
```

#### 4. Counting Sort
```python
def counting_sort(arr):
    """Counting sort - O(n + k) where k is range of values."""
    if not arr:
        return []
    
    max_val = max(arr)
    count = [0] * (max_val + 1)
    
    # Count occurrences
    for num in arr:
        count[num] += 1
    
    # Build sorted array
    result = []
    for i in range(len(count)):
        result.extend([i] * count[i])
    
    return result

# Example: Counting sort
arr = [4, 2, 2, 8, 3, 3, 1]
print(counting_sort(arr))  # [1, 2, 2, 3, 3, 4, 8]
```

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Searching Algorithms

Surveys techniques for locating elements within data collections, covering linear and binary search, interpolation methods, search trees, and hash-based lookup. Discusses prerequisites such as data ordering and structure, and contrasts worst-case versus expected performance.

### Key Concepts
- Linear Search
- Binary Search
- Interpolation Search
- Hash-based Lookup
- Search Trees (BST, AVL, Red-Black)

### Python Examples

#### 1. Linear Search
```python
def linear_search(arr, target):
    """Simple linear search - O(n)."""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# Example: Find index of target
arr = [10, 20, 30, 40, 50]
print(linear_search(arr, 30))  # 2
```

#### 2. Binary Search (Iterative)
```python
def binary_search_iterative(arr, target):
    """Binary search on sorted array - O(log n)."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Example: Binary search
arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search_iterative(arr, 7))  # 3
```

#### 3. Binary Search (Recursive)
```python
def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive binary search."""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Example: Recursive binary search
arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search_recursive(arr, 9))  # 4
```

#### 4. Binary Search - Find Insertion Position
```python
def search_insert_position(nums, target):
    """Find position to insert target in sorted array."""
    left, right = 0, len(nums)
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

# Example: Find insertion position
nums = [1, 3, 5, 6]
print(search_insert_position(nums, 5))  # 2
print(search_insert_position(nums, 2))  # 1
print(search_insert_position(nums, 7))  # 4
```

#### 5. Binary Search on Answer
```python
def find_peak_element(nums):
    """Find peak element using binary search on answer."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left

# Example: Find peak in mountain array
nums = [1, 2, 3, 1]
print(find_peak_element(nums))  # 2 (peak at index 2)
```

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## String Processing and Pattern Matching

Covers the tools for manipulating textual data: substring search (KMP, Boyer–Moore), multi-pattern matching (Aho–Corasick), suffix structures, and preprocessing functions. Provides context on applications in text editors, bioinformatics, and compression pipelines.

### Key Concepts
- Pattern Matching (KMP, Boyer-Moore, Rabin-Karp)
- String Manipulation
- Regular Expressions
- Suffix Arrays and Trees

### Python Examples

#### 1. KMP Pattern Matching
```python
def build_lps(pattern):
    """Build Longest Proper Prefix which is also Suffix array."""
    lps = [0] * len(pattern)
    length = 0
    i = 1
    
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps

def kmp_search(text, pattern):
    """KMP string matching algorithm."""
    lps = build_lps(pattern)
    i = j = 0  # i for text, j for pattern
    matches = []
    
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

# Example: Find all occurrences of pattern
text = "ABABDABACDABABCABCAB"
pattern = "ABABCABAB"
print(kmp_search(text, pattern))  # [10]
```

#### 2. Rabin-Karp Algorithm
```python
def rabin_karp(text, pattern):
    """Rabin-Karp string matching using rolling hash."""
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    base = 256
    mod = 101  # Prime number for modulo
    
    # Calculate hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    h = 1
    
    for i in range(m - 1):
        h = (h * base) % mod
    
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
        window_hash = (base * window_hash + ord(text[i])) % mod
    
    matches = []
    for i in range(n - m + 1):
        if pattern_hash == window_hash:
            if text[i:i + m] == pattern:
                matches.append(i)
        
        if i < n - m:
            window_hash = (base * (window_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
            window_hash = (window_hash + mod) % mod
    
    return matches

# Example: Find pattern in text
text = "GEEKS FOR GEEKS"
pattern = "GEEK"
print(rabin_karp(text, pattern))  # [0, 10]
```

#### 3. String Reversal and Manipulation
```python
def reverse_string(s):
    """Reverse a string."""
    return s[::-1]

def reverse_words(s):
    """Reverse words in a string."""
    words = s.split()
    return ' '.join(reversed(words))

def is_palindrome(s):
    """Check if string is palindrome."""
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

# Examples
print(reverse_string("hello"))  # "olleh"
print(reverse_words("the sky is blue"))  # "blue is sky the"
print(is_palindrome("A man a plan a canal Panama"))  # True
```

#### 4. String Anagrams
```python
from collections import Counter

def is_anagram(s1, s2):
    """Check if two strings are anagrams."""
    return Counter(s1) == Counter(s2)

def group_anagrams(strs):
    """Group strings that are anagrams of each other."""
    groups = {}
    for s in strs:
        key = ''.join(sorted(s))
        groups.setdefault(key, []).append(s)
    return list(groups.values())

# Examples
print(is_anagram("listen", "silent"))  # True
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Graph Algorithms

Frames the critical problems on graphs—traversal, connectivity, shortest paths, minimum spanning trees, flows, and matchings. Presents the algorithmic toolkit necessary for reasoning about networks, scheduling, and numerous combinatorial optimization tasks.

### Key Concepts
- Graph Traversal (BFS, DFS)
- Shortest Paths (Dijkstra, Bellman-Ford, Floyd-Warshall)
- Minimum Spanning Trees (Kruskal, Prim)
- Topological Sorting
- Strongly Connected Components

### Python Examples

#### 1. Depth-First Search (DFS)
```python
def dfs(graph, start, visited=None):
    """DFS traversal of graph."""
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    
    return result

# Example: DFS on graph
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
print(dfs(graph, 2))  # [2, 0, 1, 3]
```

#### 2. Breadth-First Search (BFS)
```python
from collections import deque

def bfs(graph, start):
    """BFS traversal of graph."""
    visited = set()
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            queue.extend(graph.get(node, []))
    
    return result

# Example: BFS on graph
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
print(bfs(graph, 2))  # [2, 0, 3, 1]
```

#### 3. Dijkstra's Shortest Path
```python
import heapq

def dijkstra(graph, start):
    """Dijkstra's algorithm for shortest paths."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor, weight in graph.get(node, []):
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distances

# Example: Shortest paths from node 0
graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}
print(dijkstra(graph, 0))  # {0: 0, 1: 3, 2: 1, 3: 4}
```

#### 4. Topological Sort
```python
def topological_sort(graph):
    """Topological sorting using DFS."""
    visited = set()
    result = []
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return result[::-1]

# Example: Topological sort of DAG
graph = {
    5: [2, 0],
    4: [0, 1],
    2: [3],
    3: [1],
    1: [],
    0: []
}
print(topological_sort(graph))  # [5, 4, 2, 3, 1, 0]
```

#### 5. Union-Find (Disjoint Set)
```python
class UnionFind:
    """Union-Find data structure for connectivity queries."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

# Example: Union-Find operations
uf = UnionFind(5)
uf.union(0, 1)
uf.union(2, 3)
print(uf.find(0) == uf.find(1))  # True
print(uf.find(0) == uf.find(2))  # False
```

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Dynamic Programming

Explains the principle of optimal substructure and overlapping subproblems, illustrating how memoization and tabulation convert exponential recurrences into polynomial-time solutions. Highlights canonical problems such as knapsack, sequence alignment, and pathfinding in weighted DAGs.

### Key Concepts
- Optimal Substructure
- Overlapping Subproblems
- Memoization vs Tabulation
- State Space Reduction
- 1D, 2D, and Multi-dimensional DP

### Python Examples

#### 1. Memoization (Top-Down)
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memo(n):
    """Fibonacci using memoization."""
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)

# Example: Calculate 10th Fibonacci number
print(fibonacci_memo(10))  # 55
```

#### 2. Tabulation (Bottom-Up)
```python
def fibonacci_tab(n):
    """Fibonacci using tabulation."""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Example: Calculate 10th Fibonacci number
print(fibonacci_tab(10))  # 55
```

#### 3. 1D DP - Climbing Stairs
```python
def climb_stairs(n):
    """Number of ways to climb n stairs (1 or 2 steps at a time)."""
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Example: Ways to climb 5 stairs
print(climb_stairs(5))  # 8
```

#### 4. 2D DP - Longest Common Subsequence
```python
def longest_common_subsequence(text1, text2):
    """Find length of longest common subsequence."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Example: LCS of "abcde" and "ace"
print(longest_common_subsequence("abcde", "ace"))  # 3 ("ace")
```

#### 5. Knapsack Problem
```python
def knapsack(weights, values, capacity):
    """0/1 Knapsack problem using DP."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]

# Example: Knapsack with capacity 7
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
capacity = 7
print(knapsack(weights, values, capacity))  # 9
```

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Greedy Algorithms

Introduces algorithms that make locally optimal choices in hopes of achieving global optimality, detailing the conditions under which greediness succeeds (matroids, exchange properties). Addresses classic applications including minimum spanning trees, scheduling, and compression coding.

### Key Concepts
- Greedy Choice Property
- Optimal Substructure
- Activity Selection
- Interval Scheduling
- Huffman Coding

### Python Examples

#### 1. Activity Selection Problem
```python
def activity_selection(start, finish):
    """Select maximum number of non-overlapping activities."""
    # Sort by finish time
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    
    selected = [0]  # First activity always selected
    last_finish = activities[0][1]
    
    for i in range(1, len(activities)):
        if activities[i][0] >= last_finish:
            selected.append(i)
            last_finish = activities[i][1]
    
    return selected

# Example: Select maximum activities
start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
print(activity_selection(start, finish))  # [0, 1, 3, 4]
```

#### 2. Fractional Knapsack
```python
def fractional_knapsack(weights, values, capacity):
    """Greedy approach: take items with highest value/weight ratio."""
    items = [(v/w, w, v) for v, w in zip(values, weights)]
    items.sort(reverse=True)  # Sort by value/weight ratio
    
    total_value = 0
    remaining = capacity
    
    for ratio, weight, value in items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            total_value += ratio * remaining
            break
    
    return total_value

# Example: Fractional knapsack
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50
print(fractional_knapsack(weights, values, capacity))  # 240.0
```

#### 3. Coin Change (Greedy - when applicable)
```python
def coin_change_greedy(coins, amount):
    """Greedy coin change (works for canonical coin systems)."""
    coins.sort(reverse=True)
    count = 0
    
    for coin in coins:
        if amount >= coin:
            num_coins = amount // coin
            count += num_coins
            amount -= num_coins * coin
    
    return count if amount == 0 else -1

# Example: Make change for 67 cents
coins = [25, 10, 5, 1]
print(coin_change_greedy(coins, 67))  # 6 (2 quarters, 1 dime, 1 nickel, 2 pennies)
```

#### 4. Interval Scheduling
```python
def erase_overlap_intervals(intervals):
    """Remove minimum intervals to make non-overlapping."""
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[1])  # Sort by end time
    count = 0
    end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            count += 1  # Remove overlapping interval
        else:
            end = intervals[i][1]
    
    return count

# Example: Remove overlapping intervals
intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
print(erase_overlap_intervals(intervals))  # 1
```

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Numerical and Scientific Algorithms

Focuses on algorithms for continuous mathematics: root finding, integration, differential equations, linear algebraic systems, and spectral analysis. Clarifies stability, convergence rates, and error bounds essential for scientific computing.

### Key Concepts
- Root Finding (Newton-Raphson, Bisection)
- Numerical Integration
- Linear Algebra Operations
- Matrix Computations

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Optimization Techniques

Covers discrete and continuous optimization, from linear programming (simplex, interior-point) to gradient-based methods and metaheuristics. Discusses constraint handling, convergence properties, and practical considerations for large-scale or non-convex problems.

### Key Concepts
- Linear Programming
- Gradient Descent
- Simulated Annealing
- Genetic Algorithms

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Machine Learning and Data Analysis Algorithms

Provides a structured view of algorithms underpinning supervised, unsupervised, and reinforcement learning. Summarizes model families (regression, SVMs, decision trees, neural networks) and the optimization routines used to train them, along with dimensionality reduction and clustering strategies.

### Key Concepts
- Supervised Learning
- Unsupervised Learning
- Feature Engineering
- Model Evaluation

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Cryptographic Algorithms

Highlights the building blocks of secure communication: symmetric and asymmetric encryption, hashing, digital signatures, and zero-knowledge protocols. Outlines the mathematical assumptions and security models that validate each cryptographic primitive.

### Key Concepts
- Encryption/Decryption
- Hash Functions
- Digital Signatures
- Key Exchange Protocols

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Data Compression Algorithms

Describes techniques for reducing data redundancy, both lossless (Huffman, Lempel–Ziv variants, arithmetic coding) and lossy (transform coding for audio, image, video). Emphasizes entropy, rate–distortion trade-offs, and practical deployment scenarios.

### Key Concepts
- Lossless Compression
- Lossy Compression
- Entropy Encoding
- Transform Coding

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Computational Geometry Algorithms

Explores algorithms for spatial data: convex hulls, nearest neighbors, Voronoi diagrams, and intersection detection. Discusses geometric primitives, numerical robustness, and applications in graphics, CAD, and geographic information systems.

### Key Concepts
- Convex Hull
- Line Intersection
- Point-in-Polygon
- Closest Pair

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Probabilistic and Approximation Algorithms

Discusses algorithms that leverage randomness or provide guaranteed approximations for intractable problems. Includes Monte Carlo and Las Vegas frameworks, randomized cuts, and approximation schemes for NP-hard tasks.

### Key Concepts
- Monte Carlo Methods
- Las Vegas Algorithms
- Randomized Algorithms
- Approximation Schemes

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Parallel and Distributed Algorithms

Surveys algorithmic models and techniques for concurrent computation: shared-memory and message-passing paradigms, parallel graph processing, consensus protocols, and large-scale data processing frameworks. Emphasizes speedup, scalability, and fault tolerance.

### Key Concepts
- Parallel Processing
- Distributed Systems
- Consensus Algorithms
- MapReduce

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Constraint Solving and Logic-Based Algorithms

Examines algorithms dedicated to constraint satisfaction, SAT/SMT solving, and logical inference. Presents backtracking, propagation (AC-3), and conflict-driven clause learning as core techniques for reasoning about combinatorial constraints.

### Key Concepts
- Constraint Satisfaction
- SAT Solving
- Backtracking Search
- Constraint Propagation

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Fundamental Data Structures

Summarizes the key data structures that enable efficient algorithmic design: arrays, linked lists, stacks, queues, heaps, balanced search trees, tries, hash tables, segment trees, and probabilistic structures. Discusses operations, complexity, and typical use cases.

### Key Concepts
- Arrays and Dynamic Arrays
- Linked Lists (Singly, Doubly, Circular)
- Stacks and Queues
- Heaps (Min/Max)
- Trees (BST, AVL, Red-Black, B-Trees)
- Hash Tables
- Tries
- Segment Trees
- Union-Find (Disjoint Set)

### Python Examples

#### 1. Linked List Implementation
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        if not self.head:
            self.head = ListNode(val)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = ListNode(val)
    
    def reverse(self):
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

# Example: Linked list operations
ll = LinkedList()
for val in [1, 2, 3, 4]:
    ll.append(val)
ll.reverse()  # Now: 4 -> 3 -> 2 -> 1
```

#### 2. Stack Implementation
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0

# Example: Stack operations
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # 3
print(stack.peek())  # 2
```

#### 3. Queue Implementation
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        return None
    
    def is_empty(self):
        return len(self.items) == 0

# Example: Queue operations
queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.dequeue())  # 1
```

#### 4. Min Heap Implementation
```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        heapq.heappush(self.heap, val)
    
    def pop(self):
        return heapq.heappop(self.heap) if self.heap else None
    
    def peek(self):
        return self.heap[0] if self.heap else None

# Example: Min heap operations
heap = MinHeap()
heap.push(3)
heap.push(1)
heap.push(4)
heap.push(2)
print(heap.pop())  # 1 (minimum)
print(heap.pop())  # 2
```

#### 5. Binary Search Tree
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert(self.root, val)
    
    def _insert(self, root, val):
        if not root:
            return TreeNode(val)
        if val < root.val:
            root.left = self._insert(root.left, val)
        else:
            root.right = self._insert(root.right, val)
        return root
    
    def search(self, val):
        return self._search(self.root, val)
    
    def _search(self, root, val):
        if not root or root.val == val:
            return root
        if val < root.val:
            return self._search(root.left, val)
        return self._search(root.right, val)

# Example: BST operations
bst = BST()
for val in [5, 3, 7, 2, 4, 6, 8]:
    bst.insert(val)
node = bst.search(4)
print(node.val if node else None)  # 4
```

#### 6. Trie (Prefix Tree)
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Example: Trie operations
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))    # True
print(trie.search("app"))      # False
print(trie.starts_with("app")) # True
```

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Array Algorithms

A dedicated treatment of algorithms centered on array data structures. Arrays offer contiguous memory storage with constant-time random access, making them the underpinning of numerous computational techniques.

### Historical Background

Arrays trace their formal use back to the dawn of high-level programming languages in the 1950s (e.g., FORTRAN, ALGOL). Early computing hardware with limited memory favored contiguous storage to minimize pointer overhead, and compilers quickly adopted array indexing as a primitive operation. Over time, arrays became the default abstraction for linear collections in systems programming, scientific computing, and data processing.

### Theoretical Foundations

At the core lies the mapping between logical indices and physical memory offsets: `address(i) = base + i * element_size`. This arithmetic enables O(1) access to any index, shaping the design of array-based algorithms. Classic operations include:

- Traversal and aggregation (sums, minima, maxima) — O(n).
- In-place updates and assignment — O(1) per element.
- Prefix sums and difference arrays, enabling range queries/updates in O(1) post-preprocessing.
- Sliding windows and two-pointer techniques, exploiting contiguity for O(n) solutions.
- Convolution and FFT-based multiplication when arrays represent polynomials or signals.

Arrays interact closely with cache hierarchies: sequential access exhibits spatial locality, enabling hardware prefetching and reduced latency compared to pointer-chasing structures.

### Core Algorithmic Patterns

| Pattern / Technique         | Purpose                                 | Typical Complexity |
|-----------------------------|-----------------------------------------|--------------------|
| Two-pointer / Sliding window| Subarray problems (e.g., longest window with constraint) | O(n) |
| Binary lifting / Prefix sums| Range queries, offline calculations     | O(n) build + O(1) query |
| Difference arrays           | Efficient range updates                  | O(1) per update with O(n) reconstruction |
| Partitioning (Lomuto/Hoare) | In-place reordering (quickselect/sort)  | O(n) for selection |
| Scan algorithms (inclusive/exclusive) | Parallel-friendly prefix computations | O(n) sequential / O(log n) parallel depth |
| Block decomposition (sqrt decomposition) | Trade-off between update/query on static array | O(n) build, O(√n) per operation |

### Python Examples

#### 1. Two-Pointer Technique
```python
def two_sum_sorted(arr, target):
    """Find two numbers that add up to target in sorted array."""
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Example: Find indices of two numbers summing to 9
arr = [2, 7, 11, 15]
print(two_sum_sorted(arr, 9))  # [0, 1]
```

#### 2. Sliding Window
```python
def max_sum_subarray(arr, k):
    """Find maximum sum of subarray of length k."""
    if len(arr) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example: Maximum sum of subarray of length 3
arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
print(max_sum_subarray(arr, 3))  # 36 (from [10, 23, 3])
```

#### 3. Prefix Sums
```python
def prefix_sum(arr):
    """Build prefix sum array for O(1) range queries."""
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, left, right):
    """Get sum of elements from index left to right (inclusive)."""
    return prefix[right + 1] - prefix[left]

# Example: Range sum queries
arr = [1, 2, 3, 4, 5]
prefix = prefix_sum(arr)
print(range_sum(prefix, 1, 3))  # 9 (2 + 3 + 4)
print(range_sum(prefix, 0, 4))  # 15 (sum of all)
```

#### 4. Difference Array (Range Updates)
```python
def difference_array(arr):
    """Build difference array for efficient range updates."""
    diff = [0] * len(arr)
    diff[0] = arr[0]
    for i in range(1, len(arr)):
        diff[i] = arr[i] - arr[i - 1]
    return diff

def update_range(diff, left, right, value):
    """Add value to all elements from left to right."""
    diff[left] += value
    if right + 1 < len(diff):
        diff[right + 1] -= value

def reconstruct_array(diff):
    """Reconstruct original array from difference array."""
    arr = [0] * len(diff)
    arr[0] = diff[0]
    for i in range(1, len(diff)):
        arr[i] = arr[i - 1] + diff[i]
    return arr

# Example: Range updates
arr = [1, 2, 3, 4, 5]
diff = difference_array(arr)
update_range(diff, 1, 3, 10)  # Add 10 to indices 1-3
result = reconstruct_array(diff)
print(result)  # [1, 12, 13, 14, 5]
```

#### 5. Partitioning (Quick Select)
```python
def partition(arr, low, high):
    """Lomuto partition scheme for quickselect/quicksort."""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickselect(arr, low, high, k):
    """Find k-th smallest element using quickselect."""
    if low == high:
        return arr[low]
    
    pivot_idx = partition(arr, low, high)
    
    if k == pivot_idx:
        return arr[pivot_idx]
    elif k < pivot_idx:
        return quickselect(arr, low, pivot_idx - 1, k)
    else:
        return quickselect(arr, pivot_idx + 1, high, k)

# Example: Find 3rd smallest element (0-indexed: k=2)
arr = [7, 10, 4, 3, 20, 15]
k = 2
result = quickselect(arr.copy(), 0, len(arr) - 1, k)
print(f"{k+1}rd smallest: {result}")  # 7
```

#### 6. Basic Traversal and Aggregation
```python
def array_operations(arr):
    """Common array operations: sum, min, max."""
    total_sum = sum(arr)
    min_val = min(arr)
    max_val = max(arr)
    return total_sum, min_val, max_val

def find_max_subarray_sum(arr):
    """Kadane's algorithm: Maximum subarray sum."""
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# Example: Maximum subarray sum
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(find_max_subarray_sum(arr))  # 6 (from [4, -1, 2, 1])
```

### Applications and Use Cases

- **Scientific Computing**: Dense matrices and tensors are stored as flattened arrays, enabling BLAS/LAPACK operations with optimized locality.
- **Computer Graphics**: Framebuffers, textures, and vertex buffers rely on array indexing for rapid GPU processing.
- **Databases and Analytics**: Columnar storage formats (e.g., Apache Parquet) arrange values in arrays to accelerate scans, SIMD operations, and compression.
- **Signal Processing**: Time-series data stored in arrays feeds DSP algorithms (FFT, filtering).
- **Machine Learning**: Tensors in frameworks like NumPy, TensorFlow, and PyTorch are multi-dimensional arrays with rich broadcasting semantics.

### Real-World Case Studies

- **YouTube's Video Pipeline**: Encoded frames are handled as arrays of pixel values; vectorized array operations power transcoding and filtering routines.
- **High-Frequency Trading**: Market tick data stored in arrays allows rapid statistical computations (moving averages, volatility estimates) with minimized latency.
- **Astronomy Data Reduction**: Large CCD images are processed as arrays where convolution kernels detect celestial bodies against noisy backgrounds.
- **Embedded Systems**: Firmware often manipulates arrays of sensor readings or control signals to maintain deterministic timing.

### Implementation Considerations

- **Memory Layout**: Row-major vs. column-major order affects performance when interfacing with libraries or hardware; e.g., C uses row-major, Fortran uses column-major.
- **Bounds Checking**: Safety-critical systems often enforce bounds checks to prevent buffer overruns; languages like Rust provide safe array abstractions.
- **Dynamic Resizing**: While plain arrays are fixed-size, dynamic arrays (e.g., `std::vector`, `ArrayList`) combine contiguous storage with amortized O(1) append via capacity doubling.
- **Parallelism**: Arrays are well-suited for SIMD and GPU kernels, enabling data-parallel speedups through vector instructions and CUDA/OpenCL.

### Related Topics and Extensions

- Multidimensional arrays and tensor computation.
- Sparse array representations (compressed sparse row/column) for sparse matrices.
- Memory alignment, cache blocking, and tiling optimizations.
- Hyperdimensional array libraries (NumPy, BLAS, cuBLAS).

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Specialized Application Algorithms

Captures domain-specific methodologies spanning operating systems (scheduling, caching), computer graphics (rasterization, clipping), networking (routing, flow control), computational linguistics, blockchain consensus, and more. Serves as a catalog for niche yet impactful algorithmic solutions.

### Key Concepts
- Operating System Algorithms
- Network Algorithms
- Database Algorithms
- Game Development Algorithms

### Related Problems
*See [Solved Problems Index](#solved-problems-index) for implementations*

---

## Solved Problems Index

This section catalogs all solved problems with links to their implementations. Problems are organized by primary algorithm category and difficulty level.

### How to Add a New Problem

When adding a new problem solution:

1. **Create the solution file** following the naming convention: `{ProblemNumber}. {ProblemTitle}.py`
2. **Update this index** by adding an entry in the appropriate category below
3. **Include metadata**: Problem number, title, difficulty, tags, and file link

**Template for new entries:**
```markdown
- [Problem Number. Problem Title]({filename}) - Difficulty: Easy/Medium/Hard | Tags: tag1, tag2, tag3
```

---

### By Category

#### Arrays & Two Pointers
- [1. Two Sum](1.%20Two Sum.py) - Difficulty: Easy | Tags: Array, Hash Table
- [11. Container With Most Water](11.%20Container%20With%20Most%20Water.py) - Difficulty: Medium | Tags: Array, Two Pointers, Greedy
- [283. Move Zeroes](283.%20Move%20Zeroes.py) - Difficulty: Easy | Tags: Array, Two Pointers
- [448. Find All Numbers Disappeared in an Array](448.%20Find%20All%20Numbers%20Disappeared%20in%20an%20Array.py) - Difficulty: Easy | Tags: Array, Hash Table
- [238. Product of Array Except Self](238.%20Product%20of%20Array%20Except%20Self.py) - Difficulty: Medium | Tags: Array, Prefix Sum
- [41. First Missing Positive](41.%20First%20Missing%20Positive.py) - Difficulty: Hard | Tags: Array, Hash Table
- [54. Spiral Matrix](54.%20Spiral%20Matrix.py) - Difficulty: Medium | Tags: Array, Matrix, Simulation
- [73. Set Matrix Zeroes](73.%20Set%20Matrix%20Zeroes.py) - Difficulty: Medium | Tags: Array, Hash Table, Matrix
- [268. Missing Number](268.%20Missing%20Number.py) - Difficulty: Easy | Tags: Array, Hash Table, Math, Bit Manipulation
- [334. Increasing Triplet Subsequence](334.%20Increasing%20Triplet%20Subsequence.py) - Difficulty: Medium | Tags: Array, Greedy
- [645. Set Mismatch](645.%20Set%20Mismatch.py) - Difficulty: Easy | Tags: Array, Hash Table, Math, Bit Manipulation
- [724. Find Pivot Index](724.%20Find%20Pivot%20Index.py) - Difficulty: Easy | Tags: Array, Prefix Sum
- [867. Transpose Matrix](867.%20Transpose%20Matrix.py) - Difficulty: Easy | Tags: Array, Matrix, Simulation

#### String Manipulation
- [13. Roman to Integer](13.%20Roman%20to%20Integer.py) - Difficulty: Easy | Tags: Hash Table, Math, String
- [38. Count and Say](38.%20Count%20and%20Say.py) - Difficulty: Medium | Tags: String
- [43. Multiply Strings](43.%20Multiply%20Strings.py) - Difficulty: Medium | Tags: Math, String, Simulation
- [58. Length of Last Word](58.%20Length%20of%20Last%20Word.py) - Difficulty: Easy | Tags: String
- [151. Reverse Words in a String](151.%20Reverse%20Words%20in%20a%20String.py) - Difficulty: Medium | Tags: Two Pointers, String
- [345. Reverse Vowels of a String](345.%20Reverse%20Vowels%20of%20a%20String.py) - Difficulty: Easy | Tags: Two Pointers, String
- [387. First Unique Character in a String](387.%20First%20Unique%20Character%20in%20a%20String.py) - Difficulty: Easy | Tags: Hash Table, String, Queue
- [392. Is Subsequence](392.%20Is%20Subsequence.py) - Difficulty: Easy | Tags: Two Pointers, String, Dynamic Programming
- [443. String Compression](443.%20String%20Compression.py) - Difficulty: Medium | Tags: Two Pointers, String
- [459. Repeated Substring Pattern](459.%20Repeated%20Substring%20Pattern.py) - Difficulty: Easy | Tags: String, String Matching
- [482. License Key Formatting](482.%20License%20Key%20Formatting.py) - Difficulty: Easy | Tags: String
- [520. Detect Capital](520.%20Detect%20Capital.py) - Difficulty: Easy | Tags: String
- [686. Repeated String Match](686.%20Repeated%20String%20Match.py) - Difficulty: Medium | Tags: String, String Matching
- [709. To Lower Case](709.%20To%20Lower%20Case.py) - Difficulty: Easy | Tags: String
- [796. Rotate String](796.%20Rotate%20String.py) - Difficulty: Easy | Tags: String, String Matching
- [819. Most Common Word](819.%20Most%20Common%20Word.py) - Difficulty: Easy | Tags: Hash Table, String, Counting
- [831. Masking Personal Information](831.%20Masking%20Personal%20Information.py) - Difficulty: Medium | Tags: String
- [859. Buddy Strings](859.%20Buddy%20Strings.py) - Difficulty: Easy | Tags: Hash Table, String
- [868. Binary Gap](868.%20Binary%20Gap.py) - Difficulty: Easy | Tags: Bit Manipulation
- [Q1. Remove Duplicate Letters](Q1.%20Remove%20Duplicate%20Letters.py) - Difficulty: Medium | Tags: String, Stack, Greedy, Monotonic Stack

#### Linked Lists
- [2. Add Two Numbers](2.%20Add%20Two%20Numbers.py) - Difficulty: Medium | Tags: Linked List, Math, Recursion
- [21. Merge Two Sorted Lists](21.%20Merge%20Two%20Sorted%20Lists.py) - Difficulty: Easy | Tags: Linked List, Recursion
- [83. Remove Duplicates from Sorted List](83.%20Remove%20Duplicates%20from%20Sorted%20List.py) - Difficulty: Easy | Tags: Linked List
- [138. Copy List with Random Pointer](138.%20Copy%20List%20with%20Random%20Pointer.py) - Difficulty: Medium | Tags: Hash Table, Linked List
- [147. Insertion Sort List](147.%20Insertion%20Sort%20List.py) - Difficulty: Medium | Tags: Linked List, Sorting
- [203. Remove Linked List Elements](203.%20Remove%20Linked%20List%20Elements.py) - Difficulty: Easy | Tags: Linked List, Recursion
- [206. Reverse Linked List](206.%20Reverse%20Linked%20List.py) - Difficulty: Easy | Tags: Linked List, Recursion
- [328. Odd Even Linked List](328.%20Odd%20Even%20Linked%20List.py) - Difficulty: Medium | Tags: Linked List
- [445. Add Two Numbers II](445.%20Add%20Two%20Numbers%20II.py) - Difficulty: Medium | Tags: Linked List, Math, Stack
- [2095. Delete the Middle Node of a Linked List](2095.%20Delete%20the%20Middle%20Node%20of%20a%20Linked%20List.py) - Difficulty: Medium | Tags: Linked List, Two Pointers
- [2130. Maximum Twin Sum of a Linked List](2130.%20Maximum%20Twin%20Sum%20of%20a%20Linked%20List.py) - Difficulty: Medium | Tags: Linked List, Two Pointers, Stack

#### Trees & Binary Trees
- [104. Maximum Depth of Binary Tree](104.%20Maximum%20Depth%20of%20Binary%20Tree.py) - Difficulty: Easy | Tags: Tree, DFS, BFS, Binary Tree
- [199. Binary Tree Right Side View](199.%20Binary%20Tree%20Right%20Side%20View.py) - Difficulty: Medium | Tags: Tree, DFS, BFS, Binary Tree
- [222. Count Complete Tree Nodes](222.%20Count%20Complete%20Tree%20Nodes.py) - Difficulty: Easy | Tags: Binary Search, Tree, Binary Tree
- [236. Lowest Common Ancestor of a Binary Tree](236.%20Lowest%20Common%20Ancestor%20of%20a%20Binary%20Tree.py) - Difficulty: Medium | Tags: Tree, DFS, Binary Tree
- [437. Path Sum III](437.%20Path%20Sum%20III.py) - Difficulty: Medium | Tags: Tree, DFS, Binary Tree
- [450. Delete Node in a BST](450.%20Delete%20Node%20in%20a%20BST.py) - Difficulty: Medium | Tags: Tree, BST, Binary Tree
- [700. Search in a Binary Search Tree](700.%20Search%20in%20a%20Binary%20Search%20Tree.py) - Difficulty: Easy | Tags: Tree, BST, Binary Tree
- [872. Leaf-Similar Trees](872.%20Leaf-Similar%20Trees.py) - Difficulty: Easy | Tags: Tree, DFS, Binary Tree
- [1161. Maximum Level Sum of a Binary Tree](1161.%20Maximum%20Level%20Sum%20of%20a%20Binary%20Tree.py) - Difficulty: Medium | Tags: Tree, BFS, Binary Tree
- [1372. Longest ZigZag Path in a Binary Tree](1372.%20Longest%20ZigZag%20Path%20in%20a%20Binary%20Tree.py) - Difficulty: Medium | Tags: Tree, DFS, Binary Tree
- [1448. Count Good Nodes in Binary Tree](1448.%20Count%20Good%20Nodes%20in%20Binary%20Tree.py) - Difficulty: Medium | Tags: Tree, DFS, Binary Tree

#### Dynamic Programming
- [70. Climbing Stairs](70.%20Climbing%20Stairs.py) - Difficulty: Easy | Tags: Math, Dynamic Programming, Memoization
- [198. House Robber](198.%20House%20Robber.py) - Difficulty: Medium | Tags: Array, Dynamic Programming
- [746. Min Cost Climbing Stairs](746.%20Min%20Cost%20Climbing%20Stairs.py) - Difficulty: Easy | Tags: Array, Dynamic Programming
- [1137. N-th Tribonacci Number](1137.%20N-th%20Tribonacci%20Number.py) - Difficulty: Easy | Tags: Math, Dynamic Programming, Memoization

#### Stack & Queue
- [20. Valid Parentheses](20.%20Valid%20Parentheses.py) - Difficulty: Easy | Tags: String, Stack
- [150. Evaluate Reverse Polish Notation](150.%20Evaluate%20Reverse%20Polish%20Notation.py) - Difficulty: Medium | Tags: Array, Math, Stack
- [232. Implement Queue using Stacks](232.%20Implement%20Queue%20using%20Stacks.py) - Difficulty: Easy | Tags: Stack, Design, Queue
- [735. Asteroid Collision](735.%20Asteroid%20Collision.py) - Difficulty: Medium | Tags: Array, Stack, Simulation
- [739. Daily Temperatures](739.%20Daily%20Temperatures.py) - Difficulty: Medium | Tags: Array, Stack, Monotonic Stack
- [841. Keys and Rooms](841.%20Keys%20and%20Rooms.py) - Difficulty: Medium | Tags: DFS, BFS, Graph
- [933. Number of Recent Calls](933.%20Number%20of%20Recent%20Calls.py) - Difficulty: Easy | Tags: Design, Queue, Data Stream
- [1021. Remove Outermost Parentheses](1021.%20Remove%20Outermost%20Parentheses.py) - Difficulty: Easy | Tags: String, Stack
- [1441. Build an Array With Stack Operations](1441.%20Build%20an%20Array%20With%20Stack%20Operations.py) - Difficulty: Medium | Tags: Array, Stack, Simulation
- [1700. Number of Students Unable to Eat Lunch](1700.%20Number%20of%20Students%20Unable%20to%20Eat%20Lunch.py) - Difficulty: Easy | Tags: Array, Stack, Queue, Simulation
- [2390. Removing Stars From a String](2390.%20Removing%20Stars%20From%20a%20String.py) - Difficulty: Medium | Tags: String, Stack, Simulation

#### Graph Algorithms
- [547. Number of Provinces](547.%20Number%20of%20Provinces.py) - Difficulty: Medium | Tags: DFS, BFS, Union Find, Graph
- [1466. Reorder Routes to Make All Paths Lead to the City Zero](1466.%20Reorder%20Routes%20to%20Make%20All%20Paths%20Lead%20to%20the%20City%20Zero.py) - Difficulty: Medium | Tags: DFS, BFS, Graph

#### Binary Search
- [162. Find Peak Element](162.%20Find%20Peak%20Element.py) - Difficulty: Medium | Tags: Array, Binary Search
- [374. Guess Number Higher or Lower](374.%20Guess%20Number%20Higher%20or%20Lower.py) - Difficulty: Easy | Tags: Binary Search, Interactive
- [852. Peak Index in a Mountain Array](852.%20Peak%20Index%20in%20a%20Mountain%20Array.py) - Difficulty: Easy | Tags: Array, Binary Search
- [875. Koko Eating Bananas](875.%20Koko%20Eating%20Bananas.py) - Difficulty: Medium | Tags: Array, Binary Search

#### Sorting & Heap
- [215. Kth Largest Element in an Array](215.%20Kth%20Largest%20Element%20in%20an%20Array.py) - Difficulty: Medium | Tags: Array, Divide and Conquer, Sorting, Heap
- [912. Sort an Array](912.%20Sort%20an%20Array.py) - Difficulty: Medium | Tags: Array, Divide and Conquer, Sorting, Heap, Merge Sort, Quick Sort

#### Greedy Algorithms
- [56. Merge Intervals](56.%20Merge%20Intervals.py) - Difficulty: Medium | Tags: Array, Sorting, Greedy
- [392. Is Subsequence](392.%20Is%20Subsequence.py) - Difficulty: Easy | Tags: Two Pointers, String, Greedy, Dynamic Programming
- [605. Can Place Flowers](605.%20Can%20Place%20Flowers.py) - Difficulty: Easy | Tags: Array, Greedy
- [649. Dota2 Senate](649.%20Dota2%20Senate.py) - Difficulty: Medium | Tags: String, Greedy, Queue
- [860. Lemonade Change](860.%20Lemonade%20Change.py) - Difficulty: Easy | Tags: Array, Greedy
- [888. Fair Candy Swap](888.%20Fair%20Candy%20Swap.py) - Difficulty: Easy | Tags: Array, Hash Table, Binary Search, Sorting

#### Math & Bit Manipulation
- [50. Pow(x, n)](50.%20Pow(x,%20n).py) - Difficulty: Medium | Tags: Math, Recursion
- [326. Power of Three](326.%20Power%20of%20Three.py) - Difficulty: Easy | Tags: Math, Recursion
- [412. Fizz Buzz](412.%20Fizz%20Buzz.py) - Difficulty: Easy | Tags: Math, String, Simulation
- [461. Hamming Distance](461.%20Hamming%20Distance.py) - Difficulty: Easy | Tags: Bit Manipulation
- [492. Construct the Rectangle](492.%20Construct%20the%20Rectangle.py) - Difficulty: Easy | Tags: Math
- [507. Perfect Number](507.%20Perfect%20Number.py) - Difficulty: Easy | Tags: Math
- [594. Longest Harmonious Subsequence](594.%20Longest%20Harmonious%20Subsequence.py) - Difficulty: Easy | Tags: Array, Hash Table, Sorting
- [657. Robot Return to Origin](657.%20Robot%20Return%20to%20Origin.py) - Difficulty: Easy | Tags: String, Simulation
- [693. Binary Number with Alternating Bits](693.%20Binary%20Number%20with%20Alternating%20Bits.py) - Difficulty: Easy | Tags: Bit Manipulation
- [976. Largest Perimeter Triangle](976.%20Largest%20Perimeter%20Triangle.py) - Difficulty: Easy | Tags: Array, Math, Greedy, Sorting

#### Backtracking
- [17. Letter Combinations of a Phone Number](17.%20Letter%20Combinations%20of%20a%20Phone%20Number.py) - Difficulty: Medium | Tags: Hash Table, String, Backtracking

#### Hash Table & Counting
- [136. Single Number](136.%20Single%20Number.py) - Difficulty: Easy | Tags: Array, Bit Manipulation
- [496. Next Greater Element I](496.%20Next%20Greater%20Element%20I.py) - Difficulty: Easy | Tags: Array, Hash Table, Stack, Monotonic Stack
- [804. Unique Morse Code Words](804.%20Unique%20Morse%20Code%20Words.py) - Difficulty: Easy | Tags: Array, Hash Table, String
- [942. DI String Match](942.%20DI%20String%20Match.py) - Difficulty: Easy | Tags: Array, Two Pointers, String, Greedy

#### Sliding Window
- [643. Maximum Average Subarray I](643.%20Maximum%20Average%20Subarray%20I.py) - Difficulty: Easy | Tags: Array, Sliding Window
- [1456. Maximum Number of Vowels in a Substring of Given Length](1456.%20Maximum%20Number%20of%20Vowels%20in%20a%20Substring%20of%20Given%20Length.py) - Difficulty: Medium | Tags: String, Sliding Window

#### SQL Problems
- [175. Recyclable and Low Fat Products](175.%20Recyclable%20and%20Low%20Fat%20Products.py) - Difficulty: Easy | Tags: Database
- [183. Customers Who Never Order](183.%20Customers%20Who%20Never%20Order.py) - Difficulty: Easy | Tags: Database
- [595. Big Countries](595.%20Big%20Countries.py) - Difficulty: Easy | Tags: Database
- [1148. Article Views I](1148.%20Article%20Views%20I.py) - Difficulty: Easy | Tags: Database
- [1683. Invalid Tweets](1683.%20Invalid%20Tweets.py) - Difficulty: Easy | Tags: Database

#### Other Problems
- [66. Plus One](66.%20Plus%20One.py) - Difficulty: Easy | Tags: Array, Math
- [67. Add Binary](67.%20Add%20Binary.py) - Difficulty: Easy | Tags: Math, String, Bit Manipulation, Simulation
- [84. Largest Rectangle in Histogram](84.%20Largest%20Rectangle%20in%20Histogram.py) - Difficulty: Hard | Tags: Array, Stack, Monotonic Stack
- [292. Nim Game](292.%20Nim%20Game.py) - Difficulty: Easy | Tags: Math, Brainteaser, Game Theory
- [394. Decode String](394.%20Decode%20String.py) - Difficulty: Medium | Tags: String, Stack, Recursion
- [399. Evaluate Division](399.%20Evaluate%20Division.py) - Difficulty: Medium | Tags: Array, DFS, BFS, Union Find, Graph
- [636. Exclusive Time of Functions](636.%20Exclusive%20Time%20of%20Functions.py) - Difficulty: Medium | Tags: Array, Stack
- [682. Baseball Game](682.%20Baseball%20Game.py) - Difficulty: Easy | Tags: Array, Stack, Simulation
- [896. Monotonic Array](896.%20Monotonic%20Array.py) - Difficulty: Easy | Tags: Array
- [1041. Robot Bounded In Circle](1041.%20Robot%20Bounded%20In%20Circle.py) - Difficulty: Medium | Tags: Math, String, Simulation
- [1046. Last Stone Weight](1046.%20Last%20Stone%20Weight.py) - Difficulty: Easy | Tags: Array, Heap (Priority Queue)
- [1071. Greatest Common Divisor of Strings](1071.%20Greatest%20Common%20Divisor%20of%20Strings.py) - Difficulty: Easy | Tags: Math, String
- [1078. Occurrences After Bigram](1078.%20Occurrences%20After%20Bigram.py) - Difficulty: Easy | Tags: String
- [1200. Minimum Absolute Difference](1200.%20Minimum%20Absolute%20Difference.py) - Difficulty: Easy | Tags: Array, Sorting
- [1207. Unique Number of Occurrences](1207.%20Unique%20Number%20of%20Occurrences.py) - Difficulty: Easy | Tags: Array, Hash Table
- [1232. Check If It Is a Straight Line](1232.%20Check%20If%20It%20Is%20a%20Straight%20Line.py) - Difficulty: Easy | Tags: Array, Math, Geometry
- [1268. Search Suggestions System](1268.%20Search%20Suggestions%20System.py) - Difficulty: Medium | Tags: Array, String, Binary Search, Trie, Sorting
- [1275. Find Winner on a Tic Tac Toe Game](1275.%20Find%20Winner%20on%20a%20Tic%20Tac%20Toe%20Game.py) - Difficulty: Easy | Tags: Array, Hash Table, Matrix, Simulation
- [1354. Construct Target Array With Multiple Sums](1354.%20Construct%20Target%20Array%20With%20Multiple%20Sums.py) - Difficulty: Hard | Tags: Array, Heap (Priority Queue)
- [1365. How Many Numbers Are Smaller Than the Current Number](1365.%20How%20Many%20Numbers%20Are%20Smaller%20Than%20the%20Current%20Number.py) - Difficulty: Easy | Tags: Array, Hash Table, Sorting, Counting
- [1408. String Matching in an Array](1408.%20String%20Matching%20in%20an%20Array.py) - Difficulty: Easy | Tags: Array, String, String Matching
- [1431. Kids With the Greatest Number of Candies](1431.%20Kids%20With%20the%20Greatest%20Number%20of%20Candies.py) - Difficulty: Easy | Tags: Array
- [1470. Shuffle the Array](1470.%20Shuffle%20the%20Array.py) - Difficulty: Easy | Tags: Array
- [1475. Final Prices With a Special Discount in a Shop](1475.%20Final%20Prices%20With%20a%20Special%20Discount%20in%20a%20Shop.py) - Difficulty: Easy | Tags: Array, Stack, Monotonic Stack
- [1491. Average Salary Excluding the Minimum and Maximum Salary](1491.%20Average%20Salary%20Excluding%20the%20Minimum%20and%20Maximum%20Salary.py) - Difficulty: Easy | Tags: Array, Sorting
- [1493. Longest Subarray of 1's After Deleting One Element](1493.%20Longest%20Subarray%20of%201's%20After%20Deleting%20One%20Element.py) - Difficulty: Medium | Tags: Array, Dynamic Programming, Sliding Window
- [1502. Can Make Arithmetic Progression From Sequence](1502.%20Can%20Make%20Arithmetic%20Progression%20From%20Sequence.py) - Difficulty: Easy | Tags: Array, Sorting
- [1523. Count Odd Numbers in an Interval Range](1523.%20Count%20Odd%20Numbers%20in%20an%20Interval%20Range.py) - Difficulty: Easy | Tags: Math
- [1534. Count Good Triplets](1534.%20Count%20Good%20Triplets.py) - Difficulty: Easy | Tags: Array, Enumeration
- [1550. Three Consecutive Odds](1550.%20Three%20Consecutive%20Odds.py) - Difficulty: Easy | Tags: Array
- [1572. Matrix Diagonal Sum](1572.%20Matrix%20Diagonal%20Sum.py) - Difficulty: Easy | Tags: Array, Matrix
- [1590. Make Sum Divisible by P](1590.%20Make%20Sum%20Divisible%20by%20P.py) - Difficulty: Medium | Tags: Array, Hash Table, Prefix Sum
- [1657. Determine if Two Strings Are Close](1657.%20Determine%20if%20Two%20Strings%20Are%20Close.py) - Difficulty: Medium | Tags: Hash Table, String, Sorting, Counting
- [1664. Ways to Make a Fair Array](1664.%20Ways%20to%20Make%20a%20Fair%20Array.py) - Difficulty: Medium | Tags: Array, Prefix Sum
- [1672. Richest Customer Wealth](1672.%20Richest%20Customer%20Wealth.py) - Difficulty: Easy | Tags: Array, Matrix
- [1679. Max Number of K-Sum Pairs](1679.%20Max%20Number%20of%20K-Sum%20Pairs.py) - Difficulty: Medium | Tags: Array, Hash Table, Two Pointers, Sorting
- [1684. Count the Number of Consistent Strings](1684.%20Count%20the%20Number%20of%20Consistent%20Strings.py) - Difficulty: Easy | Tags: Array, Hash Table, String, Bit Manipulation
- [1716. Calculate Money in Leetcode Bank](1716.%20Calculate%20Money%20in%20Leetcode%20Bank.py) - Difficulty: Easy | Tags: Math
- [1726. Tuple with Same Product](1726.%20Tuple%20with%20Same%20Product.py) - Difficulty: Medium | Tags: Array, Hash Table
- [1732. Find the Highest Altitude](1732.%20Find%20the%20Highest%20Altitude.py) - Difficulty: Easy | Tags: Array, Prefix Sum
- [1752. Check if Array Is Sorted and Rotated](1752.%20Check%20if%20Array%20Is%20Sorted%20and%20Rotated.py) - Difficulty: Easy | Tags: Array
- [1758. Minimum Changes To Make Alternating Binary String](1758.%20Minimum%20Changes%20To%20Make%20Alternating%20Binary%20String.py) - Difficulty: Easy | Tags: String, Greedy
- [1768. Merge Strings Alternately](1768.%20Merge%20Strings%20Alternately.py) - Difficulty: Easy | Tags: Two Pointers, String
- [1773. Count Items Matching a Rule](1773.%20Count%20Items%20Matching%20a%20Rule.py) - Difficulty: Easy | Tags: Array, String
- [1780. Check if Number is a Sum of Powers of Three](1780.%20Check%20if%20Number%20is%20a%20Sum%20of%20Powers%20of%20Three.py) - Difficulty: Medium | Tags: Math
- [1790. Check if One String Swap Can Make Strings Equal](1790.%20Check%20if%20One%20String%20Swap%20Can%20Make%20Strings%20Equal.py) - Difficulty: Easy | Tags: Hash Table, String, Counting
- [1800. Maximum Ascending Subarray Sum](1800.%20Maximum%20Ascending%20Subarray%20Sum.py) - Difficulty: Easy | Tags: Array
- [1822. Sign of the Product of an Array](1822.%20Sign%20of%20the%20Product%20of%20an%20Array.py) - Difficulty: Easy | Tags: Array, Math
- [1863. Sum of All Subset XOR Totals](1863.%20Sum%20of%20All%20Subset%20XOR%20Totals.py) - Difficulty: Easy | Tags: Array, Math, Backtracking, Bit Manipulation, Enumeration
- [1887. Reduction Operations to Make the Array Elements Equal](1887.%20Reduction%20Operations%20to%20Make%20the%20Array%20Elements%20Equal.py) - Difficulty: Medium | Tags: Array, Sorting
- [1922. Count Good Numbers](1922.%20Count%20Good%20Numbers.py) - Difficulty: Medium | Tags: Math, Recursion
- [1925. Count Square Sum Triples](1925.%20Count%20Square%20Sum%20Triples.py) - Difficulty: Easy | Tags: Math, Enumeration
- [1926. Nearest Exit from Entrance in Maze](1926.%20Nearest%20Exit%20from%20Entrance%20in%20Maze.py) - Difficulty: Medium | Tags: Array, BFS, Matrix
- [1929. Concatenation of Array](1929.%20Concatenation%20of%20Array.py) - Difficulty: Easy | Tags: Array
- [1941. Check if All Characters Have Equal Number of Occurrences](1941.%20Check%20if%20All%20Characters%20Have%20Equal%20Number%20of%20Occurrences.py) - Difficulty: Easy | Tags: Hash Table, String, Counting
- [1952. Three Divisors](1952.%20Three%20Divisors.py) - Difficulty: Easy | Tags: Math
- [2016. Maximum Difference Between Increasing Elements](2016.%20Maximum%20Difference%20Between%20Increasing%20Elements.py) - Difficulty: Easy | Tags: Array
- [2047. Number of Valid Words in a Sentence](2047.%20Number%20of%20Valid%20Words%20in%20a%20Sentence.py) - Difficulty: Easy | Tags: String
- [2062. Count Vowel Substrings of a String](2062.%20Count%20Vowel%20Substrings%20of%20a%20String.py) - Difficulty: Easy | Tags: Hash Table, String
- [2073. Time Needed to Buy Tickets](2073.%20Time%20Needed%20to%20Buy%20Tickets.py) - Difficulty: Easy | Tags: Array, Queue, Simulation
- [2103. Rings and Rods](2103.%20Rings%20and%20Rods.py) - Difficulty: Easy | Tags: Hash Table, String
- [2114. Maximum Number of Words Found in Sentences](2114.%20Maximum%20Number%20of%20Words%20Found%20in%20Sentences.py) - Difficulty: Easy | Tags: Array, String
- [2119. A Number After a Double Reversal](2119.%20A%20Number%20After%20a%20Double%20Reversal.py) - Difficulty: Easy | Tags: Math
- [2131. Longest Palindrome by Concatenating Two Letter Words](2131.%20Longest%20Palindrome%20by%20Concatenating%20Two%20Letter%20Words.py) - Difficulty: Medium | Tags: Array, Hash Table, String, Greedy, Counting
- [2154. Keep Multiplying Found Values by Two](2154.%20Keep%20Multiplying%20Found%20Values%20by%20Two.py) - Difficulty: Easy | Tags: Array, Hash Table, Sorting
- [2161. Partition Array According to Given Pivot](2161.%20Partition%20Array%20According%20to%20Given%20Pivot.py) - Difficulty: Medium | Tags: Array, Two Pointers, Simulation
- [2176. Count Equal and Divisible Pairs in an Array](2176.%20Count%20Equal%20and%20Divisible%20Pairs%20in%20an%20Array.py) - Difficulty: Easy | Tags: Array
- [2206. Divide Array Into Equal Pairs](2206.%20Divide%20Array%20Into%20Equal%20Pairs.py) - Difficulty: Easy | Tags: Array, Hash Table, Bit Manipulation, Counting
- [2215. Find the Difference of Two Arrays](2215.%20Find%20the%20Difference%20of%20Two%20Arrays.py) - Difficulty: Easy | Tags: Array, Hash Table
- [2264. Largest 3-Same-Digit Number in String](2264.%20Largest%203-Same-Digit%20Number%20in%20String.py) - Difficulty: Easy | Tags: String
- [2273. Find Resultant Array After Removing Anagrams](2273.%20Find%20Resultant%20Array%20After%20Removing%20Anagrams.py) - Difficulty: Easy | Tags: Array, Hash Table, String, Sorting
- [2300. Successful Pairs of Spells and Potions](2300.%20Successful%20Pairs%20of%20Spells%20and%20Potions.py) - Difficulty: Medium | Tags: Array, Two Pointers, Binary Search, Sorting
- [2319. Check if Matrix Is X-Matrix](2319.%20Check%20if%20Matrix%20Is%20X-Matrix.py) - Difficulty: Easy | Tags: Array, Matrix
- [2336. Smallest Number in Infinite Set](2336.%20Smallest%20Number%20in%20Infinite%20Set.py) - Difficulty: Medium | Tags: Hash Table, Design, Heap (Priority Queue)
- [2342. Max Sum of a Pair With Equal Sum of Digits](2342.%20Max%20Sum%20of%20a%20Pair%20With%20Equal%20Sum%20of%20Digits.py) - Difficulty: Medium | Tags: Array, Hash Table, Sorting, Heap (Priority Queue)
- [2352. Equal Row and Column Pairs](2352.%20Equal%20Row%20and%20Column%20Pairs.py) - Difficulty: Medium | Tags: Array, Hash Table, Matrix, Simulation
- [2357. Make Array Zero by Subtracting Equal Amounts](2357.%20Make%20Array%20Zero%20by%20Subtracting%20Equal%20Amounts.py) - Difficulty: Easy | Tags: Array, Hash Table, Sorting, Greedy
- [2418. Sort the People](2418.%20Sort%20the%20People.py) - Difficulty: Easy | Tags: Array, Hash Table, Sorting
- [2423. Remove Letter To Equalize Frequency](2423.%20Remove%20Letter%20To%20Equalize%20Frequency.py) - Difficulty: Easy | Tags: Hash Table, String, Counting
- [2451. Odd String Difference](2451.%20Odd%20String%20Difference.py) - Difficulty: Easy | Tags: Hash Table, String, Math
- [2462. Total Cost to Hire K Workers](2462.%20Total%20Cost%20to%20Hire%20K%20Workers.py) - Difficulty: Medium | Tags: Array, Two Pointers, Heap (Priority Queue), Simulation
- [2475. Number of Unequal Triplets in Array](2475.%20Number%20of%20Unequal%20Triplets%20in%20Array.py) - Difficulty: Easy | Tags: Array, Hash Table
- [2529. Maximum Count of Positive Integer and Negative Integer](2529.%20Maximum%20Count%20of%20Positive%20Integer%20and%20Negative%20Integer.py) - Difficulty: Easy | Tags: Array, Binary Search, Counting
- [2542. Maximum Subsequence Score](2542.%20Maximum%20Subsequence%20Score.py) - Difficulty: Medium | Tags: Array, Greedy, Sorting, Heap (Priority Queue)
- [2570. Merge Two 2D Arrays by Summing Values](2570.%20Merge%20Two%202D%20Arrays%20by%20Summing%20Values.py) - Difficulty: Easy | Tags: Array, Hash Table, Two Pointers
- [2578. Split With Minimum Sum](2578.%20Split%20With%20Minimum%20Sum.py) - Difficulty: Easy | Tags: Math, Greedy, Sorting
- [2579. Count Total Number of Colored Cells](2579.%20Count%20Total%20Number%20of%20Colored%20Cells.py) - Difficulty: Medium | Tags: Math
- [2586. Count the Number of Vowel Strings in Range](2586.%20Count%20the%20Number%20of%20Vowel%20Strings%20in%20Range.py) - Difficulty: Easy | Tags: Array, String
- [2639. Find the Width of Columns of a Grid](2639.%20Find%20the%20Width%20of%20Columns%20of%20a%20Grid.py) - Difficulty: Easy | Tags: Array, Matrix
- [2651. Calculate Delayed Arrival Time](2651.%20Calculate%20Delayed%20Arrival%20Time.py) - Difficulty: Easy | Tags: Math
- [2682. Find the Losers of the Circular Game](2682.%20Find%20the%20Losers%20of%20the%20Circular%20Game.py) - Difficulty: Easy | Tags: Array, Hash Table, Simulation
- [2733. Neither Minimum nor Maximum](2733.%20Neither%20Minimum%20nor%20Maximum.py) - Difficulty: Easy | Tags: Array, Sorting
- [2788. Split Strings by Separator](2788.%20Split%20Strings%20by%20Separator.py) - Difficulty: Easy | Tags: Array, String
- [2824. Count Pairs Whose Sum is Less than Target](2824.%20Count%20Pairs%20Whose%20Sum%20is%20Less%20than%20Target.py) - Difficulty: Easy | Tags: Array, Two Pointers, Binary Search, Sorting
- [2828. Check if a String Is an Acronym of Words](2828.%20Check%20if%20a%20String%20Is%20an%20Acronym%20of%20Words.py) - Difficulty: Easy | Tags: Array, String
- [2843. Count Symmetric Integers](2843.%20Count%20Symmetric%20Integers.py) - Difficulty: Easy | Tags: Math, Enumeration
- [2873. Maximum Value of an Ordered Triplet I](2873.%20Maximum%20Value%20of%20an%20Ordered%20Triplet%20I.py) - Difficulty: Easy | Tags: Array, Enumeration
- [2877. Create a DataFrame from List](2877.%20Create%20a%20DataFrame%20from%20List.py) - Difficulty: Easy | Tags: Database
- [2878. Get the Size of a DataFrame](2878.%20Get%20the%20Size%20of%20a%20DataFrame.py) - Difficulty: Easy | Tags: Database
- [2879. Display the First Three Rows](2879.%20Display%20the%20First%20Three%20Rows.py) - Difficulty: Easy | Tags: Database
- [2880. Select Data](2880.%20Select%20Data.py) - Difficulty: Easy | Tags: Database
- [2881. Create a New Column](2881.%20Create%20a%20New%20Column.py) - Difficulty: Easy | Tags: Database
- [2882. Drop Duplicate Rows](2882.%20Drop%20Duplicate%20Rows.py) - Difficulty: Easy | Tags: Database
- [2883. Drop Missing Data](2883.%20Drop%20Missing%20Data.py) - Difficulty: Easy | Tags: Database
- [2884. Modify Columns](2884.%20Modify%20Columns.py) - Difficulty: Easy | Tags: Database
- [2885. Rename Columns](2885.%20Rename%20Columns.py) - Difficulty: Easy | Tags: Database
- [2886. Change Data Type](2886.%20Change%20Data%20Type.py) - Difficulty: Easy | Tags: Database
- [2887. Fill Missing Data](2887.%20Fill%20Missing%20Data.py) - Difficulty: Easy | Tags: Database
- [2888. Reshape Data: Concatenate](2888.%20Reshape%20Data:%20Concatenate.py) - Difficulty: Easy | Tags: Database
- [2889. Reshape Data: Pivot](2889.%20Reshape%20Data:%20Pivot.py) - Difficulty: Easy | Tags: Database
- [2890. Reshape Data: Melt](2890.%20Reshape%20Data:%20Melt.py) - Difficulty: Easy | Tags: Database
- [2891. Method Chaining](2891.%20Method%20Chaining.py) - Difficulty: Easy | Tags: Database
- [2894. Divisible and Non-divisible Sums Difference](2894.%20Divisible%20and%20Non-divisible%20Sums%20Difference.py) - Difficulty: Easy | Tags: Math
- [2903. Find Indices With Index and Value Difference I](2903.%20Find%20Indices%20With%20Index%20and%20Value%20Difference%20I.py) - Difficulty: Easy | Tags: Array
- [2923. Find Champion I](2923.%20Find%20Champion%20I.py) - Difficulty: Easy | Tags: Array, Matrix
- [2932. Maximum Strong Pair XOR I](2932.%20Maximum%20Strong%20Pair%20XOR%20I.py) - Difficulty: Easy | Tags: Array, Bit Manipulation, Trie
- [2942. Find Words Containing Character](2942.%20Find%20Words%20Containing%20Character.py) - Difficulty: Easy | Tags: Array, String
- [2965. Find Missing and Repeated Values](2965.%20Find%20Missing%20and%20Repeated%20Values.py) - Difficulty: Easy | Tags: Array, Hash Table, Matrix
- [2980. Check if Bitwise OR Has Trailing Zeros](2980.%20Check%20if%20Bitwise%20OR%20Has%20Trailing%20Zeros.py) - Difficulty: Easy | Tags: Array, Bit Manipulation
- [3010. Divide an Array Into Subarrays With Minimum Cost I](3010.%20Divide%20an%20Array%20Into%20Subarrays%20With%20Minimum%20Cost%20I.py) - Difficulty: Easy | Tags: Array, Greedy, Sorting
- [3033. Modify the Matrix](3033.%20Modify%20the%20Matrix.py) - Difficulty: Easy | Tags: Array, Matrix
- [3079. Find the Sum of Encrypted Integers](3079.%20Find%20the%20Sum%20of%20Encrypted%20Integers.py) - Difficulty: Easy | Tags: Array, Math
- [3105. Longest Strictly Increasing or Strictly Decreasing Subarray](3105.%20Longest%20Strictly%20Increasing%20or%20Strictly%20Decreasing%20Subarray.py) - Difficulty: Easy | Tags: Array
- [3127. Make a Square with the Same Color](3127.%20Make%20a%20Square%20with%20the%20Same%20Color.py) - Difficulty: Easy | Tags: Array, Matrix, Enumeration
- [3142. Check if Grid Satisfies Conditions](3142.%20Check%20if%20Grid%20Satisfies%20Conditions.py) - Difficulty: Easy | Tags: Array, Matrix
- [3151. Special Array I](3151.%20Special%20Array%20I.py) - Difficulty: Easy | Tags: Array
- [3160. Find the Number of Distinct Colors Among the Balls](3160.%20Find%20the%20Number%20of%20Distinct%20Colors%20Among%20the%20Balls.py) - Difficulty: Medium | Tags: Array, Hash Table, Simulation
- [3169. Count Days Without Meetings](3169.%20Count%20Days%20Without%20Meetings.py) - Difficulty: Medium | Tags: Array, Sorting, Prefix Sum
- [3174. Clear Digits](3174.%20Clear%20Digits.py) - Difficulty: Easy | Tags: String, Stack, Simulation
- [3194. Minimum Average of Smallest and Largest Elements](3194.%20Minimum%20Average%20of%20Smallest%20and%20Largest%20Elements.py) - Difficulty: Easy | Tags: Array, Sorting, Greedy
- [3210. Find the Encrypted String](3210.%20Find%20the%20Encrypted%20String.py) - Difficulty: Easy | Tags: String, Simulation
- [3289. The Two Sneaky Numbers of Digitville](3289.%20The%20Two%20Sneaky%20Numbers%20of%20Digitville.py) - Difficulty: Easy | Tags: Array, Hash Table, Counting
- [3370. Smallest Number With All Set Bits](3370.%20Smallest%20Number%20With%20All%20Set%20Bits.py) - Difficulty: Easy | Tags: Math, Bit Manipulation
- [3375. Minimum Operations to Make Array Values Equal to K](3375.%20Minimum%20Operations%20to%20Make%20Array%20Values%20Equal%20to%20K.py) - Difficulty: Medium | Tags: Array, Hash Table, Math, Greedy
- [3396. Minimum Number of Operations to Make Elements in Array Distinct](3396.%20Minimum%20Number%20of%20Operations%20to%20Make%20Elements%20in%20Array%20Distinct.py) - Difficulty: Easy | Tags: Array, Hash Table, Greedy, Sorting
- [3407. Substring Matching Pattern](3407.%20Substring%20Matching%20Pattern.py) - Difficulty: Easy | Tags: String, String Matching
- [3432. Count Partitions with Even Sum Difference](3432.%20Count%20Partitions%20with%20Even%20Sum%20Difference.py) - Difficulty: Easy | Tags: Array, Math, Counting
- [3487. Maximum Unique Subarray Sum After Deletion](3487.%20Maximum%20Unique%20Subarray%20Sum%20After%20Deletion.py) - Difficulty: Medium | Tags: Array, Hash Table, Sliding Window, Prefix Sum
- [3550. Smallest Index With Digit Sum Equal to Index](3550.%20Smallest%20Index%20With%20Digit%20Sum%20Equal%20to%20Index.py) - Difficulty: Easy | Tags: Math, Enumeration
- [373. Find K Pairs with Smallest Sums](373.%20Find%20K%20Pairs%20with%20Smallest%20Sums.py) - Difficulty: Medium | Tags: Array, Heap (Priority Queue)

---

### By Difficulty

#### Easy
*See individual categories above for Easy problems*

#### Medium
*See individual categories above for Medium problems*

#### Hard
- [41. First Missing Positive](41.%20First%20Missing%20Positive.py) - Tags: Array, Hash Table
- [84. Largest Rectangle in Histogram](84.%20Largest%20Rectangle%20in%20Histogram.py) - Tags: Array, Stack, Monotonic Stack
- [1354. Construct Target Array With Multiple Sums](1354.%20Construct%20Target%20Array%20With%20Multiple%20Sums.py) - Tags: Array, Heap (Priority Queue)

---

### Statistics

- **Total Problems Solved**: 300+
- **Easy**: ~150
- **Medium**: ~140
- **Hard**: ~10

*Last Updated: [Update this date when adding new problems]*

---

## Further Reading and Study Resources

Curates authoritative textbooks, monographs, lecture series, and online platforms that support in-depth study and practice. Provides guidance on sequencing topics and building expertise progressively.

### Recommended Textbooks
- "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein (CLRS)
- "Algorithm Design Manual" by Steven Skiena
- "Elements of Programming Interviews" by Adnan Aziz

### Online Platforms
- LeetCode
- Codeforces
- HackerRank
- AtCoder

### Study Paths
1. **Beginner**: Start with basic data structures and simple algorithms
2. **Intermediate**: Focus on dynamic programming, graph algorithms, and advanced data structures
3. **Advanced**: Tackle complex optimization problems and competitive programming challenges

---

## Notes

- This index is maintained manually. When adding new problems, update the appropriate category section.
- Problem difficulty and tags are based on LeetCode classifications.
- File naming convention: `{ProblemNumber}. {ProblemTitle}.py`
- Each solution file should include test cases and be executable.
