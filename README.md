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
6. [Stack Algorithms](#stack-algorithms)
7. [Queue Algorithms](#queue-algorithms)
8. [Heap Algorithms](#heap-algorithms)
9. [String Algorithms](#string-algorithms)
10. [String Matching Algorithm](#string-matching-algorithm)
11. [Linked List Algorithm](#linked-list-algorithm)
12. [Hash Algorithm](#hash-algorithm)
13. [Prefix Sum Algorithm](#prefix-sum-algorithm)
14. [Counting Sort Algorithm](#counting-sort-algorithm)
15. [Merge Sort Algorithm](#merge-sort-algorithm)
16. [Two Pointers Algorithm](#two-pointers-algorithm)
17. [Sliding Window Algorithm](#sliding-window-algorithm)
18. [Tree Algorithm](#tree-algorithm)
19. [Binary Tree Algorithm](#binary-tree-algorithm)
20. [DFS Algorithm](#dfs-algorithm)
21. [BFS Algorithm](#bfs-algorithm)
22. [Sorting Algorithms](#sorting-algorithms)
23. [Searching Algorithms](#searching-algorithms)
24. [String Processing and Pattern Matching](#string-processing-and-pattern-matching)
25. [Array Algorithms](#array-algorithms)
26. [Graph Algorithms](#graph-algorithms)

### Part III: Data Structures
27. [Fundamental Data Structures](#fundamental-data-structures)

### Part IV: Specialized Domains
28. [Numerical and Scientific Algorithms](#numerical-and-scientific-algorithms)
29. [Optimization Techniques](#optimization-techniques)
30. [Machine Learning and Data Analysis Algorithms](#machine-learning-and-data-analysis-algorithms)
31. [Cryptographic Algorithms](#cryptographic-algorithms)
32. [Data Compression Algorithms](#data-compression-algorithms)
33. [Computational Geometry Algorithms](#computational-geometry-algorithms)
34. [Parallel and Distributed Algorithms](#parallel-and-distributed-algorithms)
35. [Constraint Solving and Logic-Based Algorithms](#constraint-solving-and-logic-based-algorithms)
36. [Specialized Application Algorithms](#specialized-application-algorithms)

### Part V: Practice Problems
37. [Solved Problems Index](#solved-problems-index)

### Part VI: Resources
38. [Further Reading and Study Resources](#further-reading-and-study-resources)

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

## Stack Algorithms

**Stacks** are LIFO (last-in, first-out) structures. They support fast `push` (add to top) and `pop` (remove from top), and are ideal for problems that need **reversal**, **nesting / matching**, or **“undo”** operations.

Common time/space:
- `push`, `pop`, `peek`, `is_empty`: **O(1)** time.
- Space: O(n) for n elements.

In interview / competitive programming tasks, stacks appear in:
- Parentheses / bracket validation.
- Expression evaluation (e.g., Reverse Polish Notation).
- Monotonic stacks (next greater/smaller element, ranges, histograms).
- DFS (iterative).
- Parsing / decoding nested structures (e.g., `decodeString`).

### 1. Basic stack implementation (Python list)

```python
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            return None
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def __len__(self):
        return len(self._items)


# Example usage
s = Stack()
for x in [1, 2, 3]:
    s.push(x)

print(s.pop())   # 3
print(s.peek())  # 2
print(len(s))    # 2
```

### 2. Valid parentheses (classic stack problem)

Check if every opening bracket has a matching closing bracket in the correct order.

```python
def is_valid_parentheses(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []

    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()

    return not stack


# Examples
print(is_valid_parentheses("()[]{}"))      # True
print(is_valid_parentheses("(]"))          # False
print(is_valid_parentheses("([{}])"))      # True
print(is_valid_parentheses("((())"))       # False
```

### 3. Evaluate Reverse Polish Notation (RPN)

Given tokens like `["2","1","+","3","*"]`, compute the result using a stack.

```python
def eval_rpn(tokens: list[str]) -> int:
    stack: list[int] = []

    for token in tokens:
        if token in {"+", "-", "*", "/"}:
            b = stack.pop()
            a = stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            else:  # division truncates toward zero
                stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[-1]


# Examples
print(eval_rpn(["2", "1", "+", "3", "*"]))     # (2 + 1) * 3 = 9
print(eval_rpn(["4", "13", "5", "/", "+"]))    # 4 + 13/5 = 6
```

### 4. Monotonic stack – Next Greater Element

Maintain a stack that is **monotonic decreasing** in values to find, for each element, the next element to the right that is greater.

```python
def next_greater_elements(nums: list[int]) -> list[int]:
    """
    For each index i, find index j > i such that nums[j] > nums[i] and j is minimal.
    If no such j exists, answer is -1 for that position.
    """
    n = len(nums)
    res = [-1] * n
    stack: list[int] = []  # stack of indices, nums[stack] is decreasing

    for i, val in enumerate(nums):
        # Resolve all positions whose next greater is current val
        while stack and nums[stack[-1]] < val:
            idx = stack.pop()
            res[idx] = i
        stack.append(i)

    return res


# Example
nums = [2, 1, 2, 4, 3]
idxs = next_greater_elements(nums)
print(idxs)                     # [3, 2, 3, -1, -1]
print([nums[i] if i != -1 else -1 for i in idxs])
# [4, 2, 4, -1, -1]
```

### 5. Monotonic stack – Daily Temperatures style

Same pattern as above but returning **distance** instead of index.

```python
def days_until_warmer(temps: list[int]) -> list[int]:
    """
    For each day i, return how many days you would have to wait until a warmer temperature.
    If there is no future day with a warmer temperature, return 0 for that day.
    """
    n = len(temps)
    res = [0] * n
    stack: list[int] = []  # indices of days, temps[stack] is decreasing

    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)

    return res


# Example
temps = [73, 74, 75, 71, 69, 72, 76, 73]
print(days_until_warmer(temps))  # [1, 1, 4, 2, 1, 1, 0, 0]
```

### 6. Decode nested strings (e.g. \"3[a2[c]]\" → \"accaccacc\")

Use a stack to handle nested repetition and concatenation.

```python
def decode_string(s: str) -> str:
    num_stack: list[int] = []
    str_stack: list[str] = []
    current_num = 0
    current_str = []

    for ch in s:
        if ch.isdigit():
            current_num = current_num * 10 + int(ch)
        elif ch == '[':
            # Push current context
            num_stack.append(current_num)
            str_stack.append(''.join(current_str))
            current_num = 0
            current_str = []
        elif ch == ']':
            repeat = num_stack.pop()
            prev_str = str_stack.pop()
            current_str = [prev_str + ''.join(current_str) * repeat]
        else:
            current_str.append(ch)

    return ''.join(current_str)


# Examples
print(decode_string(\"3[a]2[bc]\"))      # \"aaabcbc\"
print(decode_string(\"3[a2[c]]\"))      # \"accaccacc\"
print(decode_string(\"2[abc]3[cd]ef\")) # \"abcabccdcdcdef\"
```

### 7. Implementation notes and patterns

| Pattern / Use case              | Idea                                                                 |\n|---------------------------------|----------------------------------------------------------------------|\n| Parentheses / bracket matching  | Push opening, pop when matching closing; invalid if mismatch/stack left. |\n| Expression evaluation (RPN)     | Push numbers, on operator pop 2 operands, compute, push result.     |\n| Monotonic stack                 | Maintain increasing/decreasing order to answer range queries in O(n). |\n| Iterative DFS                   | Use stack instead of recursion to explore graph or tree.            |\n| Undo operations                 | Push previous states onto stack; `undo` pops last state.            |\n\n### Related sections and problems\n\n- Data structure details: [Fundamental Data Structures](#fundamental-data-structures) (Stack subsection).\n- Typical LeetCode problems: Valid Parentheses, Evaluate Reverse Polish Notation, Daily Temperatures, Next Greater Element, Decode String (see [Solved Problems Index](#solved-problems-index)).\n*** End Patch```} ***!

---

## Queue Algorithms

**Queues** are FIFO (first-in, first-out) structures. Elements are inserted at the **back** (enqueue) and removed from the **front** (dequeue). Queues are ideal for **breadth-first traversal**, **task scheduling**, and **buffering** data streams.

In Python, `collections.deque` is the preferred implementation for O(1) appends/pops at both ends.

### Operations and complexity

- `enqueue` (push to back): **O(1)**
- `dequeue` (pop from front): **O(1)**
- `peek_front`, `peek_back`, `is_empty`: **O(1)**
- Space: **O(n)** for n elements

### 1. Basic queue implementation (using deque)

```python
from collections import deque


class Queue:
    def __init__(self):
        self._q = deque()

    def enqueue(self, item):
        self._q.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._q.popleft()

    def peek(self):
        if self.is_empty():
            return None
        return self._q[0]

    def is_empty(self):
        return len(self._q) == 0

    def __len__(self):
        return len(self._q)


# Example usage
q = Queue()
for x in [1, 2, 3]:
    q.enqueue(x)

print(q.dequeue())  # 1
print(q.peek())     # 2
print(len(q))       # 2
```

### 2. BFS (Breadth-First Search) on a graph

Breadth-first search uses a queue to explore nodes level by level.

```python
from collections import deque


def bfs_graph(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    BFS traversal of an unweighted graph represented as adjacency list.
    Returns nodes in the order they are visited.
    """
    visited = set([start])
    order: list[int] = []
    q: deque[int] = deque([start])

    while q:
        node = q.popleft()
        order.append(node)

        for nei in graph.get(node, []):
            if nei not in visited:
                visited.add(nei)
                q.append(nei)

    return order


# Example
graph = {
    0: [1, 2],
    1: [3],
    2: [3, 4],
    3: [5],
    4: [],
    5: [],
}
print(bfs_graph(graph, 0))  # [0, 1, 2, 3, 4, 5]
```

### 3. Shortest path in unweighted graph (BFS distance)

For unweighted graphs, BFS finds the shortest number of edges from a start node to all others.

```python
from collections import deque


def shortest_path_unweighted(graph: dict[int, list[int]], start: int) -> dict[int, int]:
    """
    Return distance (in edges) from start to each reachable node.
    Unreachable nodes are absent from the result.
    """
    dist: dict[int, int] = {start: 0}
    q: deque[int] = deque([start])

    while q:
        node = q.popleft()
        for nei in graph.get(node, []):
            if nei not in dist:
                dist[nei] = dist[node] + 1
                q.append(nei)

    return dist


# Example
graph = {
    0: [1, 2],
    1: [3],
    2: [3, 4],
    3: [5],
    4: [5],
    5: [],
}
print(shortest_path_unweighted(graph, 0))  # {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
```

### 4. Sliding window maximum with deque

A **monotonic deque** (double-ended queue) can maintain candidates for the maximum in a sliding window in overall O(n) time.

```python
from collections import deque


def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """
    Return list of maximums for each window of size k.
    Uses deque to store indices, maintaining decreasing values.
    """
    if not nums or k <= 0:
        return []

    dq: deque[int] = deque()  # indices, nums[dq] is decreasing
    res: list[int] = []

    for i, val in enumerate(nums):
        # Remove indices that are out of this window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Maintain decreasing order in deque
        while dq and nums[dq[-1]] <= val:
            dq.pop()

        dq.append(i)

        # The front of deque is max for window ending at i
        if i >= k - 1:
            res.append(nums[dq[0]])

    return res


# Example
nums = [1, 3, -1, -3, 5, 3, 6, 7]
print(sliding_window_max(nums, 3))  # [3, 3, 5, 5, 6, 7]
```

### 5. Implementing a stack using two queues

Classic interview question: simulate LIFO stack behavior using only queue operations.

```python
from collections import deque


class MyStack:
    def __init__(self):
        self.q1: deque[int] = deque()
        self.q2: deque[int] = deque()

    def push(self, x: int) -> None:
        # Push to q2, then move all from q1 -> q2, then swap
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self) -> int:
        return self.q1.popleft()

    def top(self) -> int:
        return self.q1[0]

    def empty(self) -> bool:
        return not self.q1


# Example
s = MyStack()
s.push(1)
s.push(2)
print(s.top())  # 2
print(s.pop())  # 2
print(s.empty())  # False
```

### 6. Implementation patterns and notes

| Pattern / Use case                    | Idea                                                                 |
|---------------------------------------|----------------------------------------------------------------------|
| BFS traversal                         | Use queue to process nodes level by level.                          |
| Shortest path in unweighted graphs    | BFS distance with queue; each edge adds 1 to distance.              |
| Sliding window (deque)                | Maintain candidates at ends; pop outdated or dominated elements.    |
| Producer–consumer / task scheduling   | Queue tasks; workers dequeue and process in order.                  |
| Simulating other structures           | Implement stack, priority queues, or schedulers using 1–2 queues.   |

### Related sections and problems

- Data structure details: [Fundamental Data Structures](#fundamental-data-structures) (Queue subsection).
- Typical LeetCode problems: Number of Recent Calls, Time Needed to Buy Tickets, Rotting Oranges, Binary Tree Level Order Traversal, Sliding Window Maximum (see [Solved Problems Index](#solved-problems-index)).

---

## Heap Algorithms

A **heap** (binary heap) is a complete binary tree where each node is **≥** (max-heap) or **≤** (min-heap) its children. The root is thus the maximum or minimum element. Heaps support fast insertion and extraction of the extremal value, making them ideal for **priority queues**, **top-k**, and **incremental ordering** problems.

In Python, `heapq` provides a **min-heap** only; for a max-heap, negate values or use a custom comparator.

### Operations and complexity

| Operation        | Time    | Notes |
|------------------|---------|--------|
| Push (insert)    | O(log n)| Bubble up to restore heap property. |
| Pop (extract min/max) | O(log n) | Replace root with last leaf, then bubble down. |
| Peek (min/max)   | O(1)    | Root of the heap. |
| Heapify (array → heap) | O(n) | Bottom-up heapify, not n × log n. |

Space: **O(n)** for n elements.

### When to use

- **Priority queue**: process items by priority (e.g. Dijkstra, scheduling).
- **Top K** (or K smallest/largest): keep a heap of size K.
- **Merge K sorted lists**: heap of front elements from each list.
- **Median / percentiles**: two heaps (max-heap for lower half, min-heap for upper half).
- **N-way merge** or **streaming** “best next” choices.

### 1. Basic min-heap with `heapq`

Python’s `heapq` uses a list; the smallest element is always at index 0.

```python
import heapq

# Build min-heap from list (in-place, O(n))
nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)
# nums is now a min-heap; nums[0] == 1

# Push and pop
heapq.heappush(nums, 0)
print(heapq.heappop(nums))  # 0
print(heapq.heappop(nums))  # 1

# Peek smallest without removing
if nums:
    print(nums[0])  # 1
```

### 2. Max-heap by negating values

`heapq` is min-heap only; negate keys for max-heap behavior.

```python
import heapq

def max_heap_push(heap, value):
    heapq.heappush(heap, -value)

def max_heap_pop(heap):
    return -heapq.heappop(heap)

def max_heap_peek(heap):
    return -heap[0] if heap else None

# Example: keep largest 3
arr = [4, 1, 7, 3, 9, 2, 6]
max_heap: list[int] = []
for x in arr:
    max_heap_push(max_heap, x)
    if len(max_heap) > 3:
        max_heap_pop(max_heap)

print([max_heap_pop(max_heap) for _ in range(3)])  # [9, 7, 6] (order may vary when popping)
# Or just read: top 3 are 9, 7, 6
```

### 3. Top K largest elements (min-heap of size K)

Keep only K elements in a min-heap; the root is the K-th largest. New elements larger than the root replace it.

```python
import heapq

def top_k_largest(nums: list[int], k: int) -> list[int]:
    if k <= 0 or not nums:
        return []
    if k >= len(nums):
        return sorted(nums, reverse=True)

    heap = nums[:k]
    heapq.heapify(heap)  # min-heap of first k

    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)  # pop smallest, push x

    return sorted(heap, reverse=True)

# Example
nums = [3, 2, 1, 5, 6, 4]
print(top_k_largest(nums, 2))   # [6, 5]
print(top_k_largest(nums, 3))   # [6, 5, 4]
```

### 4. Merge K sorted lists

Use a min-heap of (value, list_index, element_index). Each pop gives the next smallest; push the next element from that list.

```python
import heapq

def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    # Min-heap: (value, list_idx, element_idx)
    h: list[tuple[int, int, int]] = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(h, (lst[0], i, 0))

    result = []
    while h:
        val, li, ei = heapq.heappop(h)
        result.append(val)
        if ei + 1 < len(lists[li]):
            next_val = lists[li][ei + 1]
            heapq.heappush(h, (next_val, li, ei + 1))

    return result

# Example
lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(merge_k_sorted(lists))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 5. Kth largest element (quick-select style with heap)

Build a min-heap of size K from the first K elements; for the rest, if larger than the root, replace root. The root is the K-th largest.

```python
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
    if not nums or k < 1 or k > len(nums):
        raise ValueError("invalid k or empty nums")

    heap = nums[:k]
    heapq.heapify(heap)

    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)

    return heap[0]

# Example
nums = [3, 2, 1, 5, 6, 4]
print(find_kth_largest(nums, 2))  # 5
print(find_kth_largest(nums, 1))  # 6
```

### 6. Heap sort (max-heap, in-place idea)

Build a max-heap, then repeatedly swap root with last element, shrink heap, and sift down. Results in ascending order.

```python
def heapify_down(arr: list[int], n: int, i: int) -> None:
    """Max-heap: sift down node at index i in heap of size n."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_down(arr, n, largest)

def heap_sort(arr: list[int]) -> None:
    n = len(arr)
    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        heapify_down(arr, n, i)
    # Extract max repeatedly
    for size in range(n - 1, 0, -1):
        arr[0], arr[size] = arr[size], arr[0]
        heapify_down(arr, size, 0)

# Example
arr = [12, 11, 13, 5, 6, 7]
heap_sort(arr)
print(arr)  # [5, 6, 7, 11, 12, 13]
```

### 7. Find median from a data stream (two heaps)

Maintain a **max-heap** for the lower half and a **min-heap** for the upper half. Keep sizes balanced (or low has at most one more). Median is the max of the lower or the average of the two roots.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo: list[int] = []   # max-heap (negated values)
        self.hi: list[int] = []  # min-heap

    def add_num(self, num: int) -> None:
        heapq.heappush(self.lo, -num)
        # Balance: move max of lo to hi
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self) -> float:
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2.0

# Example
mf = MedianFinder()
for x in [1, 2, 3, 4, 5]:
    mf.add_num(x)
print(mf.find_median())  # 3.0
```

### 8. Implementation notes and patterns

| Pattern / Use case        | Idea                                                                 |
|---------------------------|----------------------------------------------------------------------|
| Min-heap in Python       | `heapq`: list as heap; `heapify`, `heappush`, `heappop`, `heapreplace`. |
| Max-heap                 | Store negated values in a min-heap; negate when reading.             |
| Top K / Kth largest      | Min-heap of size K; drop smallest when adding a larger element.      |
| Merge K sorted           | Heap of (value, list_id, index); pop min, push next from same list.  |
| Median from stream       | Max-heap for lower half, min-heap for upper; keep sizes balanced.    |
| Heap sort                | Build max-heap, then repeatedly extract max to the end.              |

### Related sections and problems

- Data structure details: [Fundamental Data Structures](#fundamental-data-structures) (Heap subsection).
- Sorting: [Sorting Algorithms](#sorting-algorithms) (heap sort).
- Typical LeetCode problems: Kth Largest Element, Merge K Sorted Lists, Find Median from Data Stream, Top K Frequent Elements, Last Stone Weight (see [Solved Problems Index](#solved-problems-index)).

---

## String Algorithms

**String algorithms** deal with sequences of characters: comparison, search, transformation, and pattern matching. In Python, strings are **immutable**; repeated concatenation is O(n²). Prefer building with a list and `''.join()` for O(n) when assembling large strings.

### Operations and complexity (typical)

| Operation              | Time (Python) | Notes |
|------------------------|---------------|--------|
| Index / slice          | O(1) / O(k)   | k = slice length. |
| Length                 | O(1)          | |
| Concatenate two        | O(n + m)      | New string allocated. |
| `in` (substring)       | O(n × m)      | Naive; use KMP for O(n + m). |
| Iteration              | O(n)          | |
| Build via list + join  | O(n)          | Prefer over repeated `+=`. |

### When to use

- **Two pointers**: palindromes, compare from both ends, valid strings.
- **Sliding window**: longest substring with at most K distinct, max vowels in window, etc.
- **Hashing / frequency**: anagrams, first unique character, character counts.
- **Stack**: nested structures, valid parentheses, decode string.
- **Pattern matching**: substring search (see [String Processing and Pattern Matching](#string-processing-and-pattern-matching) for KMP/Rabin–Karp).

### 1. Reversal and palindrome

```python
def reverse_string(s: str) -> str:
    return s[::-1]

def is_palindrome(s: str) -> bool:
    """Ignore non-alphanumeric, case-insensitive."""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def reverse_words(s: str) -> str:
    """Reverse order of words; keep spaces as in original spacing."""
    return ' '.join(s.split()[::-1])

# Examples
print(reverse_string("hello"))                    # "olleh"
print(is_palindrome("A man a plan a canal Panama"))  # True
print(reverse_words("the sky is blue"))           # "blue is sky the"
```

### 2. Anagrams (frequency-based)

```python
from collections import Counter

def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups: dict[tuple, list[str]] = {}
    for w in strs:
        key = tuple(sorted(w))
        groups.setdefault(key, []).append(w)
    return list(groups.values())

# Examples
print(is_anagram("listen", "silent"))   # True
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### 3. First unique character

```python
from collections import Counter

def first_uniq_char(s: str) -> int:
    """Return index of first non-repeating character, or -1."""
    freq = Counter(s)
    for i, c in enumerate(s):
        if freq[c] == 1:
            return i
    return -1

# Example
print(first_uniq_char("leetcode"))   # 0 ('l')
print(first_uniq_char("aabb"))       # -1
```

### 4. Longest substring without repeating characters (sliding window)

```python
def length_of_longest_substring(s: str) -> int:
    """Length of longest substring with all distinct characters."""
    seen: set[str] = set()
    left = 0
    best = 0
    for right, c in enumerate(s):
        while c in seen:
            seen.discard(s[left])
            left += 1
        seen.add(c)
        best = max(best, right - left + 1)
    return best

# Examples
print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
print(length_of_longest_substring("bbbbb"))     # 1
print(length_of_longest_substring("pwwkew"))    # 3 ("wke")
```

### 5. Maximum vowels in substring of length K (fixed window)

```python
VOWELS = set("aeiou")

def max_vowels(s: str, k: int) -> int:
    n = len(s)
    if k > n:
        k = n
    count = sum(1 for c in s[:k] if c in VOWELS)
    best = count
    for i in range(k, n):
        if s[i - k] in VOWELS:
            count -= 1
        if s[i] in VOWELS:
            count += 1
        best = max(best, count)
    return best

# Example
print(max_vowels("abciiidef", 3))   # 3 ("iii")
print(max_vowels("aeiou", 2))       # 2
```

### 6. Valid parentheses (stack)

```python
def is_valid_parens(s: str) -> bool:
    pair = {')': '(', ']': '[', '}': '{'}
    stack: list[str] = []
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack[-1] != pair[c]:
                return False
            stack.pop()
    return not stack

# Examples
print(is_valid_parens("()[]{}"))   # True
print(is_valid_parens("(]"))       # False
```

### 7. String building (efficient)

```python
def build_large_string(chars: list[str]) -> str:
    """O(n) instead of repeated += which would be O(n^2)."""
    return ''.join(chars)

def compress(chars: list[str]) -> int:
    """In-place run-length encode; return new length. Example: ['a','a','b','b','c'] -> ['a','2','b','2','c']."""
    n = len(chars)
    write = 0
    i = 0
    while i < n:
        c = chars[i]
        count = 0
        while i < n and chars[i] == c:
            count += 1
            i += 1
        chars[write] = c
        write += 1
        if count > 1:
            for d in str(count):
                chars[write] = d
                write += 1
    return write

# Example
arr = ['a', 'a', 'b', 'b', 'b', 'c']
new_len = compress(arr)
print(new_len, arr[:new_len])  # 5 ['a', '2', 'b', '3', 'c']
```

### 8. Two pointers: valid palindrome / compare

```python
def is_palindrome_two_pointers(s: str) -> bool:
    """O(1) extra space; ignore non-alphanumeric, case-insensitive."""
    i, j = 0, len(s) - 1
    while i < j:
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        if i < j and s[i].lower() != s[j].lower():
            return False
        i += 1
        j -= 1
    return True

# Example
print(is_palindrome_two_pointers("A man a plan a canal Panama"))  # True
```

### 9. Implementation notes and patterns

| Pattern / Use case              | Idea                                                                 |
|---------------------------------|----------------------------------------------------------------------|
| Building long strings           | Use list of parts then `''.join(parts)` for O(n).                    |
| Anagrams / character counts    | `Counter(s)` or fixed-size list for limited alphabet.               |
| Sliding window (variable)      | Two indices; expand right, shrink left to satisfy invariant.        |
| Sliding window (fixed length K)| Maintain window of size K; update count when shifting by 1.          |
| Nested / balanced              | Stack: push open, pop on close; invalid if mismatch or stack non-empty at end. |
| Two pointers                   | Start both ends or same end; move based on comparison.               |

### Related sections and problems

- Pattern matching (KMP, Rabin–Karp): [String Processing and Pattern Matching](#string-processing-and-pattern-matching).
- Data structures: [Fundamental Data Structures](#fundamental-data-structures).
- Typical LeetCode problems: Valid Palindrome, Valid Parentheses, Longest Substring Without Repeating Characters, Group Anagrams, First Unique Character, String Compression, Decode String (see [Solved Problems Index](#solved-problems-index)).

---

## String Matching Algorithm

**String matching** (substring search) is the problem of finding all (or the first) occurrences of a **pattern** string `P` in a **text** string `T`. We assume `n = len(T)` and `m = len(P)`.

### Algorithm comparison

| Algorithm      | Time (preprocessing) | Time (search) | Space | Notes |
|----------------|----------------------|---------------|-------|--------|
| Naive          | —                    | O(n × m)      | O(1)  | Simple; no skip. |
| KMP            | O(m)                 | O(n + m)      | O(m)  | No backtrack in text. |
| Rabin–Karp     | O(m)                 | O(n + m) avg  | O(1)* | Rolling hash; verify with compare. |
| Python `in`/`str.find` | —              | Often O(n + m) | —    | Implementations vary (e.g. mix of ideas). |

\* Excluding hash storage; rolling hash uses O(1) extra if implemented with a single value.

### When to use which

- **Naive**: Short patterns, one-off search, or when simplicity matters.
- **KMP**: Need guaranteed linear time; avoid backtracking in text; good for repeated searches with same pattern.
- **Rabin–Karp**: Multiple patterns of same length; plagiarism detection; rolling hash useful for “same substring” checks.

### 1. Naive (brute force) matching

Check every starting position in `T`; for each, compare `P` character by character. Worst case O(n × m) (e.g. `T = "aaa…a"`, `P = "aa…ab"`).

```python
def naive_search(text: str, pattern: str) -> list[int]:
    """Return list of starting indices where pattern occurs in text."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return [] if m > n else list(range(n + 1))
    indices = []
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            indices.append(i)
    return indices

# Examples
print(naive_search("ABABDABACDABABCABAB", "ABABCABAB"))  # [10]
print(naive_search("aaaa", "aa"))                        # [0, 1, 2]
print(naive_search("abc", "d"))                          # []
```

### 2. KMP (Knuth–Morris–Pratt)

KMP precomputes a **failure / LPS** (longest proper prefix that is also a suffix) array for `P`, then scans `T` once without backtracking. Time O(n + m), space O(m).

**Idea**: After a mismatch at `T[i]` and `P[j]`, instead of moving `i` back, we move `j` to `lps[j-1]` (or 0) and keep `i` unchanged.

```python
def build_lps(pattern: str) -> list[int]:
    """Build longest proper prefix which is also suffix for each prefix of pattern."""
    m = len(pattern)
    lps = [0] * m
    length = 0  # length of current longest prefix-suffix
    i = 1
    while i < m:
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

def kmp_search(text: str, pattern: str) -> list[int]:
    """Return list of starting indices of pattern in text using KMP."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return [] if m > n else list(range(n + 1))

    lps = build_lps(pattern)
    indices = []
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                indices.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return indices

# Examples
print(kmp_search("ABABDABACDABABCABAB", "ABABCABAB"))  # [10]
print(kmp_search("aaaa", "aa"))                        # [0, 1, 2]
print(kmp_search("abcabc", "abc"))                      # [0, 3]
```

### 3. Rabin–Karp (rolling hash)

Compute a hash of `P` and a hash of each length-`m` window of `T`. If hashes match, verify with a direct compare to avoid false positives. With a good hash and modulus, average time O(n + m); worst case O(n × m) if many hash collisions.

```python
def rabin_karp_search(text: str, pattern: str, base: int = 256, mod: int = 10**9 + 7) -> list[int]:
    """Return list of starting indices using Rabin–Karp rolling hash."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return [] if m > n else list(range(n + 1))

    # Compute hash of pattern and of first window
    def hash_str(s: str) -> int:
        h = 0
        for c in s:
            h = (h * base + ord(c)) % mod
        return h

    pattern_hash = hash_str(pattern)
    window_hash = hash_str(text[:m])
    high = pow(base, m - 1, mod)  # base^(m-1) % mod

    indices = []
    for i in range(n - m + 1):
        if pattern_hash == window_hash and text[i:i + m] == pattern:
            indices.append(i)
        if i < n - m:
            # Roll: remove text[i], add text[i+m]
            window_hash = (window_hash - ord(text[i]) * high) % mod
            window_hash = (window_hash * base + ord(text[i + m])) % mod
            window_hash = (window_hash + mod) % mod

    return indices

# Examples
print(rabin_karp_search("GEEKS FOR GEEKS", "GEEK"))   # [0, 10]
print(rabin_karp_search("abcabc", "abc"))             # [0, 3]
```

### 4. Check if pattern is substring (first occurrence only)

For “does `P` occur in `T`?” or “first index of `P`”, you can use any of the above and take the first result, or use Python’s built-in.

```python
def contains_substring(text: str, pattern: str) -> bool:
    return pattern in text  # Python uses an efficient algorithm internally

def first_occurrence(text: str, pattern: str) -> int:
    return text.find(pattern)  # -1 if not found

# Or with KMP: return kmp_search(text, pattern)[0] if kmp_search(text, pattern) else -1
```

### 5. Repeated substring pattern (e.g. “abab” → True, “aba” → False)

Check if `T` is built by repeating a proper substring. Concatenate `T + T`, remove first and last character, then check if `T` appears in the middle.

```python
def repeated_substring_pattern(s: str) -> bool:
    """True if s = (substring) repeated 2+ times."""
    if len(s) <= 1:
        return False
    doubled = (s + s)[1:-1]
    return s in doubled

# Examples
print(repeated_substring_pattern("abab"))   # True  (ab repeated)
print(repeated_substring_pattern("aba"))    # False
print(repeated_substring_pattern("abcabc")) # True
```

### 6. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **LPS / failure array** | Build from left to right; on mismatch, back up `length` to `lps[length-1]`, not to 0. |
| **Rabin–Karp modulus** | Use a large prime to reduce collisions; always verify with `text[i:i+m] == pattern` when hashes match. |
| **Overlapping matches** | Naive and Rabin–Karp naturally report overlapping occurrences; KMP can be adjusted (e.g. after a match set `j = lps[j-1]` to find next overlap). |
| **Empty pattern** | Define behavior: often “match at every index” or return `[]`; document it. |

### Related sections and problems

- General string tricks: [String Algorithms](#string-algorithms).
- More pattern matching (e.g. Aho–Corasick): [String Processing and Pattern Matching](#string-processing-and-pattern-matching).
- Typical LeetCode problems: Implement strStr(), Repeated Substring Pattern, Find the Index of the First Occurrence (see [Solved Problems Index](#solved-problems-index)).

---

## Linked List Algorithm

A **linked list** is a linear structure of **nodes**. Each node holds a value and a **next** (and optionally **prev**) pointer. There is no random access by index; traversal is sequential. **Singly linked**: one pointer per node. **Doubly linked**: `next` and `prev`. **Circular**: last node points back to head (or self in single-node case).

### Operations and complexity (singly linked)

| Operation | Time | Notes |
|-----------|------|--------|
| Access by index | O(k) | Must traverse k steps. |
| Search by value | O(n) | Linear scan. |
| Insert after known node | O(1) | Just relink pointers. |
| Delete node (if you have ref to predecessor or use trick) | O(1) | Relink. |
| Insert/delete at head | O(1) | Update head. |
| Reverse entire list | O(n) | One pass with pointer manipulation. |

### When to use

- **In-place** reordering without shifting (reverse, partition, reorder).
- **Merge** two sorted lists, **split** lists (e.g. middle).
- **Cycle detection** (Floyd’s hare and tortoise).
- **Dummy node** pattern to simplify head handling and “previous” pointer.

### 1. Node class and building a list

```python
from __future__ import annotations

class ListNode:
    def __init__(self, val: int = 0, next: ListNode | None = None):
        self.val = val
        self.next = next

def build_list(values: list[int]) -> ListNode | None:
    """Build a singly linked list from a list of values."""
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

def list_to_array(head: ListNode | None) -> list[int]:
    """Convert linked list to list for printing/testing."""
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

# Example
head = build_list([1, 2, 3, 4])
print(list_to_array(head))  # [1, 2, 3, 4]
```

### 2. Reverse linked list (iterative)

```python
def reverse_list(head: ListNode | None) -> ListNode | None:
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev

# Example
head = build_list([1, 2, 3, 4])
rev = reverse_list(head)
print(list_to_array(rev))  # [4, 3, 2, 1]
```

### 3. Reverse linked list (recursive)

```python
def reverse_list_rec(head: ListNode | None) -> ListNode | None:
    if not head or not head.next:
        return head
    new_head = reverse_list_rec(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

### 4. Find middle node (slow/fast pointers)

```python
def middle_node(head: ListNode | None) -> ListNode | None:
    """Return the middle node (second of two if even length)."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# Example: [1,2,3,4,5] -> 3; [1,2,3,4] -> 3 (second middle)
head = build_list([1, 2, 3, 4, 5])
print(middle_node(head).val)  # 3
```

### 5. Merge two sorted lists

```python
def merge_two_lists(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    dummy = ListNode(0)
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next

# Example
a = build_list([1, 3, 5])
b = build_list([2, 4, 6])
print(list_to_array(merge_two_lists(a, b)))  # [1, 2, 3, 4, 5, 6]
```

### 6. Detect cycle (Floyd’s cycle detection)

```python
def has_cycle(head: ListNode | None) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def cycle_entry(head: ListNode | None) -> ListNode | None:
    """Return the node where cycle begins, or None if no cycle."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

### 7. Remove Nth node from end (one pass)

```python
def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    """Remove the nth node from the end (1-indexed)."""
    dummy = ListNode(0, head)
    fast = head
    for _ in range(n):
        fast = fast.next
    slow = dummy
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next

# Example: [1,2,3,4,5], n=2 -> [1,2,3,5]
head = build_list([1, 2, 3, 4, 5])
print(list_to_array(remove_nth_from_end(head, 2)))  # [1, 2, 3, 5]
```

### 8. Reorder list (e.g. L0 → Ln → L1 → Ln−1 → …)

Split at middle, reverse second half, then merge alternately.

```python
def reorder_list(head: ListNode | None) -> None:
    """In-place: L0->Ln->L1->Ln-1->..."""
    if not head or not head.next:
        return
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    # Reverse second half
    second = slow.next
    slow.next = None
    prev = None
    while second:
        nxt = second.next
        second.next = prev
        prev = second
        second = nxt
    # Merge
    first, second = head, prev
    while second:
        t1, t2 = first.next, second.next
        first.next = second
        second.next = t1
        first, second = t1, t2

# Example
head = build_list([1, 2, 3, 4])
reorder_list(head)
print(list_to_array(head))  # [1, 4, 2, 3]
```

### 9. Dummy node pattern

Using a **dummy** node before the head avoids special-casing the head when building a new list or deleting the first node.

```python
def delete_all_with_value(head: ListNode | None, val: int) -> ListNode | None:
    """Remove all nodes with value val."""
    dummy = ListNode(0, head)
    cur = dummy
    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next
```

### 10. Implementation notes and patterns

| Pattern | Idea |
|--------|------|
| **Dummy node** | `dummy = ListNode(0, head)`; build or modify from `dummy`; return `dummy.next`. |
| **Slow/fast pointers** | Middle: advance fast 2 steps, slow 1. Cycle: same until they meet. |
| **Reverse in place** | Three pointers: `prev`, `cur`, `nxt`; reverse link then advance. |
| **Merge two sorted** | Dummy + one `cur`; attach smaller of `l1.val` / `l2.val`; attach remainder. |
| **Delete node** | If you have predecessor: `prev.next = node.next`. Without predecessor: copy `node.next.val` into `node`, then `node.next = node.next.next` (if not tail). |

### Related sections and problems

- Data structure details: [Fundamental Data Structures](#fundamental-data-structures) (Linked List subsection).
- Typical LeetCode problems: Reverse Linked List, Merge Two Sorted Lists, Linked List Cycle, Remove Nth Node From End of List, Reorder List, Add Two Numbers (see [Solved Problems Index](#solved-problems-index)).

---

## Hash Algorithm

**Hash tables** (dictionaries, maps) store key–value pairs. A **hash function** maps keys to bucket indices; **collisions** (two keys to the same bucket) are handled by chaining (list per bucket) or open addressing. In Python, `dict` and `defaultdict` are hash-based; `Counter` is a dict subclass for counting. Average-time **O(1)** insert, delete, and lookup; worst case O(n) if many collisions.

### Operations and complexity (typical)

| Operation | Average | Worst | Notes |
|-----------|--------|--------|--------|
| Insert / set | O(1) | O(n) | Rehash can occur. |
| Lookup / get | O(1) | O(n) | |
| Delete | O(1) | O(n) | |
| Iterate keys/values | O(n) | O(n) | n = number of entries. |

### When to use

- **Two Sum / complement**: store “value → index” or “value → count”; check for `target - x` in the map.
- **Counting / frequency**: `Counter(iterable)` or `defaultdict(int)`.
- **Grouping**: key = derived value (e.g. sorted tuple for anagrams); value = list of items.
- **Deduplication / seen set**: track visited nodes, indices, or states.
- **Prefix/suffix or subarray**: store prefix sums or state and look up “complement” (e.g. subarray sum equals K).
- **Caching / memoization**: key = arguments, value = result.

### 1. Two Sum (complement in hash map)

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two numbers that add up to target. Exactly one solution assumed."""
    seen: dict[int, int] = {}  # value -> index
    for i, x in enumerate(nums):
        comp = target - x
        if comp in seen:
            return [seen[comp], i]
        seen[x] = i
    return []

# Example
print(two_sum([2, 7, 11, 15], 9))   # [0, 1]
print(two_sum([3, 2, 4], 6))       # [1, 2]
```

### 2. Frequency count with Counter

```python
from collections import Counter

def frequency_count(items: list) -> dict:
    return dict(Counter(items))

# Example
nums = [1, 2, 2, 3, 2, 1, 3]
print(frequency_count(nums))  # {1: 2, 2: 3, 3: 2}

# Most common
words = ["a", "b", "a", "c", "a", "b"]
print(Counter(words).most_common(2))  # [('a', 3), ('b', 2)]
```

### 3. Group by key (e.g. anagrams)

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups: dict[tuple, list[str]] = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())

# Example
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### 4. Subarray sum equals K (prefix sum + hash)

```python
def subarray_sum_k(nums: list[int], k: int) -> int:
    """Count of contiguous subarrays with sum equal to k."""
    prefix = 0
    count = 0
    seen: dict[int, int] = {0: 1}  # prefix_sum -> frequency
    for x in nums:
        prefix += x
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count

# Example
print(subarray_sum_k([1, 1, 1], 2))     # 2
print(subarray_sum_k([1, 2, 3], 3))     # 2  ([1,2] and [3])
```

### 5. First non-repeating character (count then scan)

```python
from collections import Counter

def first_uniq_char(s: str) -> int:
    freq = Counter(s)
    for i, c in enumerate(s):
        if freq[c] == 1:
            return i
    return -1

# Example
print(first_uniq_char("leetcode"))  # 0
print(first_uniq_char("aabb"))      # -1
```

### 6. Deduplication / seen set

```python
def remove_duplicates_preserve_order(items: list) -> list:
    seen: set = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# Example
print(remove_duplicates_preserve_order([1, 2, 2, 3, 1, 4]))  # [1, 2, 3, 4]
```

### 7. Two arrays: intersection or difference

```python
def intersection(nums1: list[int], nums2: list[int]) -> list[int]:
    """Distinct elements that appear in both."""
    set1 = set(nums1)
    return list(set(x for x in nums2 if x in set1))

def find_difference(arr1: list[int], arr2: list[int]) -> list[list[int]]:
    """Elements in arr1 but not arr2, and in arr2 but not arr1."""
    s1, s2 = set(arr1), set(arr2)
    return [list(s1 - s2), list(s2 - s1)]

# Example
print(intersection([1, 2, 2, 1], [2, 2]))           # [2]
print(find_difference([1, 2, 3], [2, 4, 6]))        # [[1, 3], [4, 6]]
```

### 8. Caching / memoization (hash as cache)

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# Or manual cache
def fib_manual(n: int, cache: dict[int, int] | None = None) -> int:
    cache = cache or {}
    if n <= 1:
        return n
    if n not in cache:
        cache[n] = fib_manual(n - 1, cache) + fib_manual(n - 2, cache)
    return cache[n]
```

### 9. defaultdict for accumulators

```python
from collections import defaultdict

def group_by_key(pairs: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Group values by key: [(a,1), (b,2), (a,3)] -> {a: [1,3], b: [2]}."""
    d: dict[str, list[int]] = defaultdict(list)
    for k, v in pairs:
        d[k].append(v)
    return dict(d)

# Example
print(group_by_key([("a", 1), ("b", 2), ("a", 3)]))  # {'a': [1, 3], 'b': [2]}
```

### 10. Implementation notes and patterns

| Pattern | Idea |
|--------|------|
| **Complement / Two Sum** | Store “value → index” (or count); for each `x`, check `target - x` in map. |
| **Prefix sum + hash** | Store prefix sums; for each prefix `p`, count how many earlier prefixes = `p - k`. |
| **Group by key** | Key = normalized form (e.g. sorted tuple); value = list of items. |
| **Seen set** | Before processing, check `if x in seen`; then `seen.add(x)`. |
| **Counter** | `Counter(iterable)` for frequency; `most_common(k)` for top k. |
| **defaultdict** | Use when “missing key” should default to 0, [], etc., to avoid key checks. |

### Related sections and problems

- Data structure details: [Fundamental Data Structures](#fundamental-data-structures) (Hash Table subsection).
- Typical LeetCode problems: Two Sum, Subarray Sum Equals K, Group Anagrams, First Unique Character, Intersection of Two Arrays, LRU Cache (see [Solved Problems Index](#solved-problems-index)).

---

## Prefix Sum Algorithm

**Prefix sums** (cumulative sums) let you answer **range-sum** queries in **O(1)** after an **O(n)** build. Define `prefix[i]` = sum of `arr[0]` through `arr[i-1]` (so `prefix[0] = 0`). Then the sum of `arr[left]` to `arr[right]` (inclusive) is `prefix[right+1] - prefix[left]`. The same idea extends to **2D** (rectangle sums) and to **difference arrays** (efficient range updates).

### Operations and complexity

| Operation | Time | Notes |
|-----------|------|--------|
| Build 1D prefix | O(n) | One pass: `prefix[i+1] = prefix[i] + arr[i]`. |
| Range sum [l, r] | O(1) | `prefix[r+1] - prefix[l]`. |
| Build 2D prefix | O(rows × cols) | Two passes (row then column) or nested loop. |
| Rectangle sum | O(1) | Inclusion–exclusion with four prefix values. |
| Difference array (range add) | O(1) per update | Then O(n) to reconstruct array. |

### When to use

- **Range sum** on a static array (many queries).
- **Subarray sum equals K**: combine prefix sum with a hash map of “prefix value → count”.
- **Pivot index** / equilibrium: compare left sum and right sum (derived from prefix).
- **Range updates**: difference array (add `d` on [l, r] with two updates, then prefix-sum to get final array).
- **2D**: sum of any rectangle in a matrix.

### 1. Build 1D prefix and range sum query

```python
def build_prefix(arr: list[int]) -> list[int]:
    """prefix[i] = sum(arr[0..i-1]), so prefix[0]=0, prefix[1]=arr[0], ..."""
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix: list[int], left: int, right: int) -> int:
    """Sum of arr[left]..arr[right] inclusive. 0-indexed."""
    return prefix[right + 1] - prefix[left]

# Example
arr = [1, 2, 3, 4, 5]
prefix = build_prefix(arr)
print(range_sum(prefix, 1, 3))   # 2+3+4 = 9
print(range_sum(prefix, 0, 4))   # 15
```

### 2. Subarray sum equals K (prefix + hash)

Count contiguous subarrays with sum equal to K. For each ending index, count how many prefix sums equal `current_prefix - K`.

```python
def subarray_sum_k(nums: list[int], k: int) -> int:
    prefix = 0
    count = 0
    seen: dict[int, int] = {0: 1}
    for x in nums:
        prefix += x
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count

# Example
print(subarray_sum_k([1, 1, 1], 2))     # 2
print(subarray_sum_k([1, 2, 3], 3))     # 2  ([1,2] and [3])
```

### 3. Find pivot index (left sum == right sum)

Pivot index: sum of elements to the left equals sum of elements to the right (no elements = 0).

```python
def pivot_index(nums: list[int]) -> int:
    total = sum(nums)
    left_sum = 0
    for i, x in enumerate(nums):
        if left_sum == total - left_sum - x:
            return i
        left_sum += x
    return -1

# Example
print(pivot_index([1, 7, 3, 6, 5, 6]))  # 3  (1+7+3 == 5+6)
print(pivot_index([2, 1, -1]))          # 0  (right sum = 0)
```

### 4. Difference array (range update, then reconstruct)

Add `value` to all elements in `[left, right]` with O(1) per update. Then reconstruct the array with one prefix-sum pass.

```python
def build_diff_array(arr: list[int]) -> list[int]:
    """diff[i] = arr[i] - arr[i-1], diff[0] = arr[0]."""
    n = len(arr)
    diff = [0] * n
    diff[0] = arr[0]
    for i in range(1, n):
        diff[i] = arr[i] - arr[i - 1]
    return diff

def range_add(diff: list[int], left: int, right: int, value: int) -> None:
    """Add value to arr[left..right] (0-indexed, inclusive)."""
    diff[left] += value
    if right + 1 < len(diff):
        diff[right + 1] -= value

def reconstruct(diff: list[int]) -> list[int]:
    """Convert difference array back to original array (prefix sum of diff)."""
    arr = [0] * len(diff)
    arr[0] = diff[0]
    for i in range(1, len(diff)):
        arr[i] = arr[i - 1] + diff[i]
    return arr

# Example
arr = [1, 2, 3, 4, 5]
diff = build_diff_array(arr)
range_add(diff, 1, 3, 10)
print(reconstruct(diff))  # [1, 12, 13, 14, 5]
```

### 5. 2D prefix sum (rectangle sum)

`prefix[i][j]` = sum of all elements in the rectangle from `(0,0)` to `(i-1, j-1)`. Then sum of rectangle `(r1,c1)` to `(r2,c2)` = inclusion–exclusion.

```python
def build_prefix_2d(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows):
        for j in range(cols):
            prefix[i + 1][j + 1] = (
                grid[i][j]
                + prefix[i][j + 1]
                + prefix[i + 1][j]
                - prefix[i][j]
            )
    return prefix

def rectangle_sum(
    prefix: list[list[int]],
    r1: int, c1: int, r2: int, c2: int
) -> int:
    """Sum of rectangle (r1,c1) to (r2,c2) inclusive. 0-indexed."""
    return (
        prefix[r2 + 1][c2 + 1]
        - prefix[r1][c2 + 1]
        - prefix[r2 + 1][c1]
        + prefix[r1][c1]
    )

# Example
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
p = build_prefix_2d(grid)
print(rectangle_sum(p, 0, 0, 1, 1))  # 1+2+4+5 = 12
print(rectangle_sum(p, 1, 1, 2, 2))  # 5+6+8+9 = 28
```

### 6. Running sum (in-place prefix for simple cases)

When you only need the prefix array and not the original, you can overwrite.

```python
def running_sum(nums: list[int]) -> list[int]:
    """nums[i] becomes sum of nums[0..i]. In-place."""
    for i in range(1, len(nums)):
        nums[i] += nums[i - 1]
    return nums

# Example
arr = [1, 2, 3, 4]
print(running_sum(arr))  # [1, 3, 6, 10]
```

### 7. Product of array except self (prefix × suffix)

Using “prefix product” and “suffix product” (or one pass with a running product), compute output[i] = product of all elements except self.

```python
def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    out = [1] * n
    left = 1
    for i in range(n):
        out[i] = left
        left *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        out[i] *= right
        right *= nums[i]
    return out

# Example
print(product_except_self([1, 2, 3, 4]))  # [24, 12, 8, 6]
```

### 8. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Index convention** | Use `prefix[0]=0`, `prefix[i]=sum(arr[0..i-1])` so range [l,r] = `prefix[r+1]-prefix[l]`. |
| **Bounds** | Ensure `left`, `right` are in [0, n-1] and `left <= right`. |
| **Difference array** | Range add [l, r]: add at l, subtract at r+1; then reconstruct with prefix sum. |
| **2D** | Build row-by-row then use inclusion–exclusion for rectangle; watch 1-based prefix indices. |
| **Subarray sum K** | Initialize `seen = {0: 1}` so subarrays starting at index 0 are counted. |

### Related sections and problems

- Hash + prefix: [Hash Algorithm](#hash-algorithm) (subarray sum K).
- Array tricks: [Array Algorithms](#array-algorithms).
- Typical LeetCode problems: Subarray Sum Equals K, Find Pivot Index, Product of Array Except Self, Range Sum Query (see [Solved Problems Index](#solved-problems-index)).

---

## Counting Sort Algorithm

**Counting sort** is a **non-comparison** sorting algorithm. It counts how many times each key (value) appears, then places elements in order by iterating over the key range. It runs in **O(n + k)** time and **O(k)** extra space, where **n** = number of elements and **k** = range of keys (max − min + 1). It is **stable** if implemented with a cumulative count (prefix sum) and a backward pass when writing the output.

### When to use

- Keys are **integers** (or map to integers) in a **small range** (e.g. 0..max or min..max).
- You need a **stable** sort and the range is small.
- **Not** suitable when k is very large (e.g. full 32-bit integers with few elements); use comparison-based or radix sort instead.

### Complexity

| Step | Time | Space |
|------|------|--------|
| Count occurrences | O(n) | O(k) |
| Build output (simple) | O(n + k) | O(n) for output |
| Stable: prefix counts + backward pass | O(n + k) | O(n + k) |

### 1. Basic counting sort (non-negative integers, small range)

Assume elements are in `[0, max_val]`. Count, then write each value the right number of times in order.

```python
def counting_sort_basic(arr: list[int], max_val: int | None = None) -> list[int]:
    """Sort non-negative integers in [0, max_val]. If max_val is None, use max(arr)."""
    if not arr:
        return []
    if max_val is None:
        max_val = max(arr)
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    result = []
    for val in range(max_val + 1):
        result.extend([val] * count[val])
    return result

# Example
arr = [4, 2, 2, 8, 3, 3, 1]
print(counting_sort_basic(arr))       # [1, 2, 2, 3, 3, 4, 8]
print(counting_sort_basic(arr, 10))  # same
```

### 2. Stable counting sort (preserve order of equal elements)

Use a **cumulative count** (prefix sum) so each key gets a range of output indices. Then iterate the **original array backwards** and place each element at the end of its range, then decrement the count. This keeps equal elements in their original order.

```python
def counting_sort_stable(arr: list[int], max_val: int | None = None) -> list[int]:
    """Stable counting sort for non-negative integers in [0, max_val]."""
    if not arr:
        return []
    if max_val is None:
        max_val = max(arr)
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    # Prefix sum: count[i] = number of elements <= i (so last index for value i)
    for i in range(1, max_val + 1):
        count[i] += count[i - 1]
    result = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        val = arr[i]
        pos = count[val] - 1
        result[pos] = arr[i]
        count[val] -= 1
    return result

# Example: stability matters when values have satellite data
arr = [2, 1, 2, 0, 1]
print(counting_sort_stable(arr))  # [0, 1, 1, 2, 2]  (order of 1s and 2s preserved)
```

### 3. Handling negative numbers (shift by min)

Map values to `[0, max - min]` by subtracting `min`, sort with counting sort, then add `min` back.

```python
def counting_sort_with_negatives(arr: list[int]) -> list[int]:
    """Counting sort for any integers (shift by min to get non-negative indices)."""
    if not arr:
        return []
    lo, hi = min(arr), max(arr)
    k = hi - lo + 1
    count = [0] * k
    for x in arr:
        count[x - lo] += 1
    result = []
    for i in range(k):
        val = lo + i
        result.extend([val] * count[i])
    return result

# Example
arr = [4, -1, 2, -1, 3, 0]
print(counting_sort_with_negatives(arr))  # [-1, -1, 0, 2, 3, 4]
```

### 4. Sort by key (e.g. sort pairs by first element)

When each element has a **key** and optional satellite data, build count by key, then prefix sum, then place by key in a stable way. Here we sort (key, value) pairs by key.

```python
def counting_sort_by_key(pairs: list[tuple[int, str]], max_key: int | None = None) -> list[tuple[int, str]]:
    """Stable sort of (key, value) pairs by key. Keys non-negative in [0, max_key]."""
    if not pairs:
        return []
    keys = [p[0] for p in pairs]
    if max_key is None:
        max_key = max(keys)
    count = [0] * (max_key + 1)
    for k in keys:
        count[k] += 1
    for i in range(1, max_key + 1):
        count[i] += count[i - 1]
    result = [None] * len(pairs)
    for i in range(len(pairs) - 1, -1, -1):
        k, v = pairs[i]
        pos = count[k] - 1
        result[pos] = (k, v)
        count[k] -= 1
    return result

# Example
pairs = [(2, "b"), (1, "a"), (2, "c"), (0, "d")]
print(counting_sort_by_key(pairs))  # [(0, 'd'), (1, 'a'), (2, 'b'), (2, 'c')]
```

### 5. In-place style (overwrite original using count)

You can write sorted values back into the original array if you use the stable variant and then copy from the output buffer, or rebuild in place from the count array (simple version below overwrites in order).

```python
def counting_sort_inplace(arr: list[int], max_val: int | None = None) -> None:
    """Overwrite arr with sorted order. Non-negative integers only."""
    if not arr:
        return
    if max_val is None:
        max_val = max(arr)
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    i = 0
    for val in range(max_val + 1):
        for _ in range(count[val]):
            arr[i] = val
            i += 1

# Example
arr = [3, 1, 2, 3, 1]
counting_sort_inplace(arr)
print(arr)  # [1, 1, 2, 3, 3]
```

### 6. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Range k** | Ensure `max_val` (or `max - min`) is known and not huge; otherwise counting sort can be worse than O(n log n) comparison sort. |
| **Stability** | Use cumulative counts and iterate **backwards** over the input when placing; otherwise stability is lost. |
| **Negative / any range** | Shift by `min` so indices are in `[0, k-1]`; add `min` back when building output. |
| **Zero elements** | Handle empty array to avoid index errors. |
| **Sorting objects** | Use key extraction (e.g. key = object.id) and stable counting sort by key; keep objects with their keys. |

### Related sections and problems

- Other sorts: [Sorting Algorithms](#sorting-algorithms) (comparison-based and counting sort mention).
- Radix sort often uses counting sort as a stable subroutine for each digit.
- Typical LeetCode problems: Sort an Array (when range is small), custom sort orders with integer keys (see [Solved Problems Index](#solved-problems-index)).

---

## Merge Sort Algorithm

**Merge sort** is a **divide-and-conquer** comparison sort. It splits the array into two halves, recursively sorts each half (until length ≤ 1), then **merges** the two sorted halves into one sorted array. It runs in **O(n log n)** time in all cases and is **stable** if the merge step compares with `≤`. Extra space is **O(n)** for the merge buffer (or for left/right copies in a typical implementation).

### Complexity

| Metric | Value | Notes |
|--------|--------|--------|
| Time (best / average / worst) | O(n log n) | No worst-case slowdown. |
| Space | O(n) | For temporary arrays during merge (or O(log n) stack if in-place merge is used). |
| Stable | Yes | When merge uses `<=` for choosing from left. |

### When to use

- You need **stable** sorting with guaranteed O(n log n).
- **Linked lists**: merge sort is well-suited (O(1) extra space per node for split/merge with pointers).
- **Inversion count** and other “while merging, count cross-half pairs” problems.
- **External sort**: merge large files in passes.

### 1. Recursive merge sort (classic)

Split into left/right, sort recursively, then merge into a new list (or back into the original with an auxiliary buffer).

```python
def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list. Stable if left elements preferred on tie."""
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

def merge_sort(arr: list[int]) -> list[int]:
    """Return a new sorted list. Original unchanged."""
    if len(arr) <= 1:
        return arr.copy()
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# Example
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # [3, 9, 10, 27, 38, 43, 82]
```

### 2. In-place style (sort in a buffer, copy back)

To sort the array in place, use an auxiliary buffer of size n; merge into the buffer, then copy back. Below: sort ranges of `arr` using `temp` as scratch.

```python
def merge_into(arr: list[int], temp: list[int], left: int, mid: int, right: int) -> None:
    """Merge arr[left:mid] and arr[mid:right] into temp[left:right], then copy back to arr."""
    i, j, k = left, mid, left
    while i < mid and j < right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1
    while i < mid:
        temp[k] = arr[i]
        i += 1
        k += 1
    while j < right:
        temp[k] = arr[j]
        j += 1
        k += 1
    for k in range(left, right):
        arr[k] = temp[k]

def merge_sort_inplace(arr: list[int], temp: list[int] | None = None, left: int = 0, right: int | None = None) -> None:
    """Sort arr[left:right] in place using temp as buffer."""
    if right is None:
        right = len(arr)
    if temp is None:
        temp = [0] * len(arr)
    if right - left <= 1:
        return
    mid = (left + right) // 2
    merge_sort_inplace(arr, temp, left, mid)
    merge_sort_inplace(arr, temp, mid, right)
    merge_into(arr, temp, left, mid, right)

# Example
arr = [38, 27, 43, 3, 9, 82, 10]
merge_sort_inplace(arr)
print(arr)  # [3, 9, 10, 27, 38, 43, 82]
```

### 3. Merge two sorted arrays (two-pointer)

Same as the `merge` subroutine: two pointers, append the smaller (or equal from first to keep stability), then append the remainder.

```python
def merge_two_sorted(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out

# Example
print(merge_two_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
```

### 4. Count inversions (while merging)

An **inversion** is a pair (i, j) with i < j and arr[i] > arr[j]. During merge, when we take an element from the right half, it is smaller than all remaining elements in the left half—count them.

```python
def merge_and_count(left: list[int], right: list[int]) -> tuple[list[int], int]:
    """Merge two sorted lists and count inversions (right elements before left elements)."""
    result = []
    i = j = 0
    inversions = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            inversions += len(left) - i  # all remaining left are greater than right[j]
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result, inversions

def count_inversions(arr: list[int]) -> tuple[list[int], int]:
    """Return sorted array and total inversion count."""
    if len(arr) <= 1:
        return arr.copy(), 0
    mid = len(arr) // 2
    left_sorted, left_inv = count_inversions(arr[:mid])
    right_sorted, right_inv = count_inversions(arr[mid:])
    merged, merge_inv = merge_and_count(left_sorted, right_sorted)
    return merged, left_inv + right_inv + merge_inv

# Example
arr = [2, 4, 1, 3, 5]
sorted_arr, inv = count_inversions(arr)
print(sorted_arr, inv)  # [1, 2, 3, 4, 5] 3
```

### 5. Bottom-up (iterative) merge sort

No recursion: merge subarrays of size 1, then 2, then 4, etc., until the whole array is sorted.

```python
def merge_sort_bottom_up(arr: list[int]) -> None:
    """Sort arr in place using iterative merge (double the width each pass)."""
    n = len(arr)
    temp = [0] * n
    width = 1
    while width < n:
        for start in range(0, n, 2 * width):
            left = start
            mid = min(start + width, n)
            right = min(start + 2 * width, n)
            if mid >= right:
                continue
            i, j, k = left, mid, left
            while i < mid and j < right:
                if arr[i] <= arr[j]:
                    temp[k] = arr[i]
                    i += 1
                else:
                    temp[k] = arr[j]
                    j += 1
                k += 1
            while i < mid:
                temp[k] = arr[i]
                i += 1
                k += 1
            while j < right:
                temp[k] = arr[j]
                j += 1
                k += 1
            for k in range(left, right):
                arr[k] = temp[k]
        width *= 2

# Example
arr = [38, 27, 43, 3, 9, 82, 10]
merge_sort_bottom_up(arr)
print(arr)  # [3, 9, 10, 27, 38, 43, 82]
```

### 6. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Stability** | In merge, use `left[i] <= right[j]` (take from left on tie) so equal elements from the left half stay before those from the right. |
| **Mid index** | Use `mid = (left + right) // 2` for split; merge `arr[left:mid]` and `arr[mid:right]` to avoid off-by-one. |
| **Auxiliary space** | Recursive version that returns new lists uses O(n log n) total allocations; in-place style with one buffer uses O(n). |
| **Base case** | Length 0 or 1: return (or do nothing) to avoid infinite recursion. |
| **Bottom-up** | Loop by `width`; merge pairs of segments of size `width`; double `width` each pass. |

### Related sections and problems

- Other sorts: [Sorting Algorithms](#sorting-algorithms), [Counting Sort Algorithm](#counting-sort-algorithm).
- Divide and conquer: [Algorithm Design Paradigms](#algorithm-design-paradigms).
- Typical LeetCode problems: Sort an Array, Merge Two Sorted Lists, Count of Smaller Numbers After Self (merge-sort idea), Reverse Pairs (see [Solved Problems Index](#solved-problems-index)).

---

## Two Pointers Algorithm

The **two pointers** technique uses two indices (or pointers) that move through a sequence—often in one pass—to satisfy a condition or scan a range. Typical setups: **opposite ends** (left/right toward the center), **same direction** (both advance, possibly at different speeds), or **fast/slow** (e.g. cycle detection, middle of list). Many problems become **O(n)** time and **O(1)** extra space.

### Common variants

| Variant | Movement | Typical use |
|---------|----------|-------------|
| **Opposite ends** | `left` from start, `right` from end; move toward center | Sorted array two sum, palindrome, pair with target |
| **Same direction** | Both advance; one may “lag” to skip or collect | Remove duplicates in place, move zeroes, partition |
| **Fast/slow** | Slow +1, fast +2 (or similar) | Middle of list, cycle detection |
| **Sliding window** | Two indices define a window; expand/shrink by moving one or both | Subarray with sum, longest substring (see [String Algorithms](#string-algorithms)) |

### When to use

- **Sorted array** or **sorted structure**: opposite-end pointers to find pairs or triples.
- **In-place** removal or partition: same-direction write pointer + read pointer.
- **Palindrome** or **symmetry**: left/right from both ends.
- **Linked list** middle or cycle: fast/slow pointers.
- **Merge** two sorted sequences: two pointers (one per sequence).

### 1. Two sum in sorted array (opposite ends)

Find two indices such that `arr[i] + arr[j] == target`. Move `left` right if sum is too small, `right` left if too large.

```python
def two_sum_sorted(arr: list[int], target: int) -> list[int]:
    """Return [i, j] such that arr[i] + arr[j] == target. 0-indexed. Exactly one solution assumed."""
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return [left, right]
        if s < target:
            left += 1
        else:
            right -= 1
    return []

# Example
print(two_sum_sorted([2, 7, 11, 15], 9))   # [0, 1]
print(two_sum_sorted([1, 2, 3, 4, 6], 6))   # [1, 3]
```

### 2. Remove duplicates in place (same direction)

Keep a **write** pointer; only write when the current value is different from the last written (or use a “last seen” value). Return new length.

```python
def remove_duplicates_inplace(nums: list[int]) -> int:
    """Remove duplicates in place (sorted non-decreasing). Return new length."""
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1
    return write

# Example
arr = [1, 1, 2, 2, 3, 4, 4, 4]
n = remove_duplicates_inplace(arr)
print(n, arr[:n])  # 4 [1, 2, 3, 4]
```

### 3. Move zeroes to the end (same direction)

One pointer scans; one points to the next position to place a non-zero. Swap or assign, then advance.

```python
def move_zeroes(nums: list[int]) -> None:
    """Move all 0s to the end while preserving relative order of non-zero elements."""
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1

# Example
arr = [0, 1, 0, 3, 12]
move_zeroes(arr)
print(arr)  # [1, 3, 12, 0, 0]
```

### 4. Valid palindrome (opposite ends)

Ignore non-alphanumeric; compare from both ends. Move pointers past non-alphanumeric, then compare (case-insensitive).

```python
def is_palindrome(s: str) -> bool:
    i, j = 0, len(s) - 1
    while i < j:
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        if i < j and s[i].lower() != s[j].lower():
            return False
        i += 1
        j -= 1
    return True

# Example
print(is_palindrome("A man a plan a canal Panama"))  # True
print(is_palindrome("race a car"))                   # False
```

### 5. Container with most water (opposite ends)

Height at indices `i` and `j`; width = `j - i`. Move the pointer at the **shorter** height inward (greedy: we can’t improve by moving the taller one).

```python
def max_area(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        w = right - left
        h = min(height[left], height[right])
        best = max(best, w * h)
        if height[left] <= height[right]:
            left += 1
        else:
            right -= 1
    return best

# Example
print(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))  # 49
```

### 6. Is subsequence (two sequences, same direction)

One pointer per string. Advance the text pointer every step; advance the pattern pointer only when characters match. True if pattern pointer reaches the end.

```python
def is_subsequence(pattern: str, text: str) -> bool:
    """True if pattern is a subsequence of text."""
    p = 0
    for t in range(len(text)):
        if p < len(pattern) and text[t] == pattern[p]:
            p += 1
    return p == len(pattern)

# Example
print(is_subsequence("ace", "abcde"))   # True
print(is_subsequence("aec", "abcde"))   # False
```

### 7. Merge two sorted arrays (two pointers, one per array)

Same idea as merge in merge sort: two indices, append the smaller (or equal from first for stability), then append the rest.

```python
def merge_sorted(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out

# Example
print(merge_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
```

### 8. Fast/slow: find middle of linked list

Slow advances by 1, fast by 2. When fast reaches the end, slow is at the middle (or second middle for even length).

```python
# Assume ListNode with .next
def middle_node(head: "ListNode | None") -> "ListNode | None":
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### 9. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Bounds** | For opposite ends use `left < right` (or `left <= right` if same index can be valid). |
| **Order of moves** | In “move zeroes” / partition, advance write only after placing; in two sum, move the pointer that improves the condition. |
| **Stability** | When merging or choosing “which pointer to move,” use `<=` to prefer the left/first sequence for stability. |
| **Fast/slow** | Check `fast and fast.next` before advancing to avoid None access. |
| **Palindrome** | Skip non-alphanumeric inside the loop; compare case-insensitively. |

### Related sections and problems

- Sliding window (variable/fixed): [String Algorithms](#string-algorithms), [Array Algorithms](#array-algorithms).
- Linked list: [Linked List Algorithm](#linked-list-algorithm).
- Typical LeetCode problems: Two Sum II, Remove Duplicates from Sorted Array, Move Zeroes, Valid Palindrome, Container With Most Water, Is Subsequence, Merge Two Sorted Lists (see [Solved Problems Index](#solved-problems-index)).

---

## Sliding Window Algorithm

The **sliding window** technique uses two indices (left and right) that define a **contiguous** subarray or substring. The “window” slides by moving one or both pointers while keeping a **constraint** satisfied (e.g. sum ≤ K, at most K distinct characters) or by maintaining a **fixed size** K. Many problems are solved in **O(n)** time with one pass.

### Fixed vs variable window

| Type | Window size | Typical approach |
|------|-------------|------------------|
| **Fixed (size K)** | Always K | Compute for first window; then for each step drop left element and add right element (update aggregate). |
| **Variable** | Grows/shrinks | Expand right until constraint is violated; then shrink left until valid again (or expand/shrink depending on “max window” vs “min window”). |

### When to use

- **Subarray/substring** problems: max sum, min size with sum ≥ target, longest/shortest satisfying a condition.
- **At most K distinct** (or exactly K): maintain a frequency map and window bounds.
- **Fixed-length** stats: max in each window of size K (use deque for O(n); see [Queue Algorithms](#queue-algorithms)).

### 1. Max sum subarray of fixed size K (fixed window)

Compute sum of the first K elements; then slide: subtract `arr[left]`, add `arr[right]`, update best.

```python
def max_sum_subarray_k(arr: list[int], k: int) -> int:
    """Maximum sum of any contiguous subarray of length k."""
    if not arr or k <= 0 or k > len(arr):
        return 0
    window_sum = sum(arr[:k])
    best = window_sum
    for right in range(k, len(arr)):
        window_sum += arr[right] - arr[right - k]
        best = max(best, window_sum)
    return best

# Example
arr = [2, 1, 5, 1, 3, 2]
print(max_sum_subarray_k(arr, 3))  # 9  (subarray [5, 1, 3])
```

### 2. Longest substring without repeating characters (variable window)

Expand right; when a duplicate appears, shrink left until the duplicate is removed (use a set or frequency map).

```python
def longest_substring_no_repeat(s: str) -> int:
    """Length of longest substring with all distinct characters."""
    seen = set()
    left = 0
    best = 0
    for right, c in enumerate(s):
        while c in seen:
            seen.discard(s[left])
            left += 1
        seen.add(c)
        best = max(best, right - left + 1)
    return best

# Example
print(longest_substring_no_repeat("abcabcbb"))  # 3 ("abc")
print(longest_substring_no_repeat("pwwkew"))    # 3 ("wke")
```

### 3. Max vowels in substring of length K (fixed window)

Count vowels in first K chars; slide and update: subtract vowel at left, add vowel at right.

```python
VOWELS = set("aeiou")

def max_vowels(s: str, k: int) -> int:
    """Maximum number of vowels in any substring of length k."""
    if k <= 0 or k > len(s):
        return 0
    count = sum(1 for c in s[:k] if c in VOWELS)
    best = count
    for right in range(k, len(s)):
        if s[right - k] in VOWELS:
            count -= 1
        if s[right] in VOWELS:
            count += 1
        best = max(best, count)
    return best

# Example
print(max_vowels("abciiidef", 3))  # 3  ("iii")
```

### 4. Minimum size subarray with sum ≥ target (variable window)

Expand right (add to sum); when sum ≥ target, update best and shrink left until sum < target, then repeat.

```python
def min_subarray_sum_target(nums: list[int], target: int) -> int:
    """Minimum length of contiguous subarray with sum >= target. 0 if none."""
    left = 0
    total = 0
    best = len(nums) + 1
    for right in range(len(nums)):
        total += nums[right]
        while total >= target:
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return best if best <= len(nums) else 0

# Example
print(min_subarray_sum_target([2, 3, 1, 2, 4, 3], 7))   # 2  ([4, 3])
print(min_subarray_sum_target([1, 4, 4], 4))             # 1
```

### 5. Longest substring with at most K distinct characters (variable window)

Expand right; when distinct count > K, shrink left until count ≤ K. Track character frequencies in a dict (or Counter).

```python
def longest_substring_k_distinct(s: str, k: int) -> int:
    """Length of longest substring containing at most k distinct characters."""
    if k <= 0:
        return 0
    freq = {}
    left = 0
    best = 0
    for right, c in enumerate(s):
        freq[c] = freq.get(c, 0) + 1
        while len(freq) > k:
            lc = s[left]
            freq[lc] -= 1
            if freq[lc] == 0:
                del freq[lc]
            left += 1
        best = max(best, right - left + 1)
    return best

# Example
print(longest_substring_k_distinct("eceba", 2))   # 3  ("ece")
print(longest_substring_k_distinct("aa", 1))     # 2
```

### 6. Sliding window maximum (deque; fixed window)

For each window of size K, report the maximum. Use a **monotonic deque** (indices of elements in decreasing order). See [Queue Algorithms](#queue-algorithms) for full code; idea: drop indices that are out of window or smaller than the new element.

```python
from collections import deque

def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """Max in each window of size k."""
    if not nums or k <= 0 or k > len(nums):
        return []
    dq = deque()
    result = []
    for i, val in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= val:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

# Example
print(sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3))  # [3, 3, 5, 5, 6, 7]
```

### 7. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Fixed window** | Initialize with first K; loop `right` from K to n-1; update by removing `arr[right-K]` and adding `arr[right]`. |
| **Variable window** | Expand right in outer loop; shrink left in inner `while` until constraint holds again. |
| **Off-by-one** | Window [left, right] inclusive has length `right - left + 1`. |
| **Empty / k > n** | Return 0 or [] as appropriate. |
| **Max in window** | Use monotonic deque; indices in deque have decreasing values; drop outdated from front, dominated from back. |

### Related sections and problems

- Two pointers: [Two Pointers Algorithm](#two-pointers-algorithm).
- Deque for max/min in window: [Queue Algorithms](#queue-algorithms).
- String windows: [String Algorithms](#string-algorithms).
- Typical LeetCode problems: Maximum Average Subarray I, Longest Substring Without Repeating Characters, Minimum Size Subarray Sum, Sliding Window Maximum, Longest Repeating Character Replacement (see [Solved Problems Index](#solved-problems-index)).

---

## Tree Algorithm

**Tree algorithms** work on hierarchical structures: each **node** has a value and zero or more **children**. In a **binary tree** each node has at most two children (left and right). **Binary search trees (BST)** impose an ordering (left < root < right). Common operations are **traversal** (visit every node), **height/depth**, **path sum**, and **lowest common ancestor (LCA)**. Traversals can be **DFS** (depth-first: preorder, inorder, postorder) or **BFS** (level-order).

### Complexity (typical)

| Operation | Time | Space |
|-----------|------|--------|
| Traverse all nodes | O(n) | O(h) stack (DFS) or O(w) queue (BFS); h = height, w = max level width. |
| Search in BST | O(h) | O(h) recursive, O(1) iterative with a pointer. |
| Insert/Delete in BST | O(h) | O(h). |

### When to use

- **DFS (recursive or stack)**: path problems, subtree checks, pre/in/post order.
- **BFS (queue)**: level-order, shortest path in unweighted tree, “level by level” output.
- **BST**: search, insert, delete, inorder (sorted order), range queries.

### 1. TreeNode and basic structure

```python
from __future__ import annotations

class TreeNode:
    def __init__(self, val: int = 0, left: TreeNode | None = None, right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right
```

### 2. DFS: Preorder, Inorder, Postorder

```python
def preorder(root: TreeNode | None) -> list[int]:
    """Root -> Left -> Right."""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root: TreeNode | None) -> list[int]:
    """Left -> Root -> Right. For BST this gives sorted order."""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root: TreeNode | None) -> list[int]:
    """Left -> Right -> Root."""
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# Example: tree    1
#                 / \
#                2   3
#               / \
#              4   5
# preorder: [1, 2, 4, 5, 3]; inorder: [4, 2, 5, 1, 3]; postorder: [4, 5, 2, 3, 1]
```

### 3. BFS (level-order)

```python
from collections import deque

def level_order(root: TreeNode | None) -> list[list[int]]:
    """Return list of levels (each level is a list of values)."""
    if not root:
        return []
    result = []
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result
```

### 4. Maximum depth (height)

```python
def max_depth(root: TreeNode | None) -> int:
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### 5. Same tree (structural and value equality)

```python
def is_same_tree(p: TreeNode | None, q: TreeNode | None) -> bool:
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    return p.val == q.val and is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
```

### 6. Invert (mirror) binary tree

```python
def invert_tree(root: TreeNode | None) -> TreeNode | None:
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

### 7. Lowest common ancestor (LCA) of two nodes

```python
def lowest_common_ancestor(root: TreeNode | None, p: TreeNode, q: TreeNode) -> TreeNode | None:
    """Assume p and q exist in tree. LCA is the deepest node that has both p and q as descendants."""
    if not root or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```

### 8. Path sum (root-to-leaf sum equals target)

```python
def has_path_sum(root: TreeNode | None, target: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target
    rem = target - root.val
    return has_path_sum(root.left, rem) or has_path_sum(root.right, rem)
```

### 9. Validate BST (inorder or min/max range)

```python
def is_valid_bst(root: TreeNode | None, lo: int | None = None, hi: int | None = None) -> bool:
    """Check if tree is a valid BST. Pass optional lo, hi to bound node values."""
    if not root:
        return True
    if lo is not None and root.val <= lo:
        return False
    if hi is not None and root.val >= hi:
        return False
    return is_valid_bst(root.left, lo, root.val) and is_valid_bst(root.right, root.val, hi)
```

### 10. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Null checks** | Always handle `root is None` (or `not root`) before accessing `root.left`/`root.right`. |
| **Leaf node** | Often defined as `not root.left and not root.right`. |
| **BST property** | For “valid BST” use range (lo, hi) per node; or do inorder and check strictly increasing. |
| **LCA** | If one of p, q is the root, root is LCA; if p and q are in different subtrees, root is LCA. |
| **Iterative DFS/BFS** | Use an explicit stack for DFS or queue for BFS to avoid recursion stack overflow on deep trees. |

### Related sections and problems

- Data structures: [Fundamental Data Structures](#fundamental-data-structures) (Trees, BST).
- Graph traversal: [Graph Algorithms](#graph-algorithms) (BFS/DFS).
- Typical LeetCode problems: Maximum Depth of Binary Tree, Same Tree, Invert Binary Tree, Lowest Common Ancestor of a Binary Tree, Path Sum, Validate Binary Search Tree, Binary Tree Level Order Traversal (see [Solved Problems Index](#solved-problems-index)).

---

## Binary Tree Algorithm

A **binary tree** is a tree in which each node has at most **two children** (left and right). This section covers binary-tree-specific structure, properties, and classic algorithms beyond basic traversal (see [Tree Algorithm](#tree-algorithm) for DFS/BFS, depth, LCA, path sum).

### Definitions and properties

| Type | Definition |
|------|------------|
| **Full binary tree** | Every node has 0 or 2 children. |
| **Complete binary tree** | All levels fully filled except possibly the last, which is filled left to right. |
| **Perfect binary tree** | All leaves at same depth; exactly 2^h − 1 nodes for height h. |
| **Balanced** | Height is O(log n); e.g. AVL, Red-Black. |
| **Binary search tree (BST)** | For every node, left subtree keys < root < right subtree keys. |

### When to use

- **Recursive structure**: most algorithms recurse on `root.left` and `root.right` and combine results.
- **Path / diameter**: compute per-node “height” or “longest path through this node” and take max.
- **Construction**: build from preorder+inorder or from level-order; use ranges or queues.
- **Serialization**: encode tree to string (e.g. preorder with null markers) for storage or comparison.

### 1. Diameter of binary tree (longest path between any two nodes)

The diameter is the maximum number of **edges** on any path. For each node, the longest path **through** that node = 1 + left_height + right_height. Return the max over all nodes; recursion can return (diameter_so_far, height).

```python
from __future__ import annotations

class TreeNode:
    def __init__(self, val: int = 0, left: TreeNode | None = None, right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right

def diameter_of_binary_tree(root: TreeNode | None) -> int:
    """Return the length (number of edges) of the longest path between any two nodes."""
    best = 0

    def height(node: TreeNode | None) -> int:
        nonlocal best
        if not node:
            return 0
        left_h = height(node.left)
        right_h = height(node.right)
        best = max(best, left_h + right_h)
        return 1 + max(left_h, right_h)

    height(root)
    return best
```

### 2. Symmetric tree (mirror of itself around root)

Left subtree should be mirror of right subtree: compare `left.left` with `right.right` and `left.right` with `right.left`.

```python
def is_symmetric(root: TreeNode | None) -> bool:
    def mirror(a: TreeNode | None, b: TreeNode | None) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None or a.val != b.val:
            return False
        return mirror(a.left, b.right) and mirror(a.right, b.left)

    return mirror(root, root) if root else True
```

### 3. Build tree from preorder and inorder

Preorder gives root; find root in inorder to split into left and right inorder ranges; recurse.

```python
def build_tree_pre_in(preorder: list[int], inorder: list[int]) -> TreeNode | None:
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    i = inorder.index(root_val)
    root.left = build_tree_pre_in(preorder[1 : 1 + i], inorder[:i])
    root.right = build_tree_pre_in(preorder[1 + i :], inorder[i + 1 :])
    return root
```

### 4. Count nodes in a complete binary tree (O((log n)^2))

For a complete tree, leftmost and rightmost paths from root give left and right heights. If equal, tree is perfect (2^h − 1 nodes); else 1 + count(left) + count(right).

```python
def count_nodes_complete(root: TreeNode | None) -> int:
    if not root:
        return 0
    left_h = right_h = 0
    p = root
    while p:
        left_h += 1
        p = p.left
    p = root
    while p:
        right_h += 1
        p = p.right
    if left_h == right_h:
        return (1 << left_h) - 1
    return 1 + count_nodes_complete(root.left) + count_nodes_complete(root.right)
```

### 5. Serialize and deserialize (preorder with null markers)

Use a delimiter and a sentinel for null (e.g. `"#"`) so the string uniquely defines the tree; preorder is simple to parse.

```python
def serialize(root: TreeNode | None) -> str:
    if not root:
        return "#"
    return str(root.val) + "," + serialize(root.left) + "," + serialize(root.right)

def deserialize(data: str) -> TreeNode | None:
    it = iter(data.split(","))

    def parse() -> TreeNode | None:
        val = next(it, "#")
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = parse()
        node.right = parse()
        return node

    return parse()
```

### 6. Maximum path sum (any path; node values can be negative)

For each node, max path **through** this node = node.val + max(0, left_gain) + max(0, right_gain). Recurse returning the max **single-side** gain (node + max(0, left, right)) for the parent.

```python
def max_path_sum(root: TreeNode | None) -> int:
    best = float("-inf")

    def gain(node: TreeNode | None) -> int:
        nonlocal best
        if not node:
            return 0
        left_g = max(0, gain(node.left))
        right_g = max(0, gain(node.right))
        best = max(best, node.val + left_g + right_g)
        return node.val + max(left_g, right_g)

    gain(root)
    return best if root else 0
```

### 7. Iterative traversals (stack-based)

Preorder: push root; pop, process, push right then left. Inorder: go left to bottom, pop and process, then go right.

```python
def preorder_iter(root: TreeNode | None) -> list[int]:
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result

def inorder_iter(root: TreeNode | None) -> list[int]:
    stack = []
    result = []
    node = root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        result.append(node.val)
        node = node.right
    return result
```

### 8. Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Diameter** | Length = number of **edges**; path through node = left_height + right_height (not +1 for node count). |
| **Symmetric** | Compare left subtree with mirror of right (left.left ↔ right.right, left.right ↔ right.left). |
| **Build from pre+in** | Preorder[0] is root; find in inorder to split; use same length for preorder left/right. |
| **Complete tree count** | Use leftmost/rightmost height; if equal, subtree is perfect. |
| **Path sum / max path** | Decide whether path must be root-to-leaf or any path; handle negative values (e.g. max(0, gain)). |

### Related sections and problems

- Basic traversal and LCA: [Tree Algorithm](#tree-algorithm).
- Data structures: [Fundamental Data Structures](#fundamental-data-structures).
- Typical LeetCode problems: Diameter of Binary Tree, Symmetric Tree, Construct Binary Tree from Preorder and Inorder, Count Complete Tree Nodes, Serialize and Deserialize Binary Tree, Binary Tree Maximum Path Sum (see [Solved Problems Index](#solved-problems-index)).

---

## DFS Algorithm

**Depth-First Search (DFS)** explores as far as possible along each branch before backtracking. It is used on **trees**, **graphs**, and **implicit state spaces** (e.g. permutations, subsets). DFS can be implemented **recursively** (call stack) or **iteratively** (explicit stack).

### When to use

| Scenario | Use DFS when |
|----------|----------------|
| **Traversal** | You need preorder, inorder, or postorder on trees; or any order on graphs. |
| **Path / connectivity** | Find a path, detect cycles, or check if two nodes are connected. |
| **Backtracking** | Enumerate combinations, permutations, or constrained choices (undo state after recurse). |
| **Flood fill / components** | Label connected components on a grid or graph. |
| **Topological order** | Post-order DFS on a DAG yields reverse topological order. |

### Complexity (typical)

| Setting | Time | Space |
|---------|------|--------|
| Tree, n nodes | O(n) | O(h) recursion or stack, h = height |
| Graph, V vertices, E edges | O(V + E) | O(V) visited + stack |
| Backtracking (e.g. subsets) | O(2^n) or similar | O(n) recursion depth |

### 1. Recursive DFS on graph (adjacency list)

Visit a node, then recurse on each unvisited neighbor. Use a `visited` set to avoid cycles.

```python
def dfs_graph_recursive(graph: dict, start: int, visited: set | None = None) -> list[int]:
    """DFS traversal of graph; returns list of nodes in discovery order."""
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs_graph_recursive(graph, neighbor, visited))
    return result

# Example: graph as adjacency list
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
print(dfs_graph_recursive(graph, 2))  # [2, 0, 1, 3]
```

### 2. Iterative DFS on graph (explicit stack)

Same discovery order as recursive DFS when you push neighbors in reverse order (so first neighbor is popped first).

```python
def dfs_graph_iterative(graph: dict, start: int) -> list[int]:
    visited = set()
    stack = [start]
    result = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)
    return result

print(dfs_graph_iterative(graph, 2))  # [2, 0, 1, 3]
```

### 3. DFS on 2D grid (flood fill / connected components)

Explore all cells reachable from (r, c) that match a condition (e.g. same color). Mark visited in place or with a set.

```python
def flood_fill(grid: list[list[int]], sr: int, sc: int, new_color: int) -> list[list[int]]:
    """Replace connected component containing (sr, sc) with new_color."""
    R, C = len(grid), len(grid[0])
    old = grid[sr][sc]
    if old == new_color:
        return grid

    def dfs(r: int, c: int) -> None:
        if r < 0 or r >= R or c < 0 or c >= C or grid[r][c] != old:
            return
        grid[r][c] = new_color
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc)

    dfs(sr, sc)
    return grid
```

### 4. DFS with path (find any path from start to target)

Pass current path down the recursion; backtrack by popping after the recursive call.

```python
def find_path_graph(graph: dict, start: int, target: int) -> list[int] | None:
    visited = set()
    path: list[int] = []

    def dfs(node: int) -> bool:
        visited.add(node)
        path.append(node)
        if node == target:
            return True
        for neighbor in graph.get(node, []):
            if neighbor not in visited and dfs(neighbor):
                return True
        path.pop()
        return False

    return path if dfs(start) else None
```

### 5. Cycle detection in directed graph (DFS with three states)

Use three states: unvisited, in current path (gray), finished (black). Cycle exists iff we hit a node that is in the current path.

```python
def has_cycle_directed(graph: dict) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    state = {v: WHITE for v in graph}

    def dfs(node: int) -> bool:
        state[node] = GRAY
        for neighbor in graph.get(node, []):
            if state[neighbor] == GRAY:
                return True
            if state[neighbor] == WHITE and dfs(neighbor):
                return True
        state[node] = BLACK
        return False

    return any(state[v] == WHITE and dfs(v) for v in graph)
```

### 6. Backtracking: generate all subsets

Classic DFS over “include / exclude” choices; each call corresponds to one index.

```python
def subsets(nums: list[int]) -> list[list[int]]:
    result: list[list[int]] = []

    def dfs(i: int, path: list[int]) -> None:
        if i == len(nums):
            result.append(path[:])
            return
        dfs(i + 1, path)
        path.append(nums[i])
        dfs(i + 1, path)
        path.pop()

    dfs(0, [])
    return result

print(subsets([1, 2, 3]))  # [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```

### 7. Backtracking: generate all permutations

Swap or use a “used” mask; backtrack by undoing the choice.

```python
def permutations(nums: list[int]) -> list[list[int]]:
    result: list[list[int]] = []

    def dfs(path: list[int], used: set[int]) -> None:
        if len(path) == len(nums):
            result.append(path[:])
            return
        for x in nums:
            if x in used:
                continue
            used.add(x)
            path.append(x)
            dfs(path, used)
            path.pop()
            used.remove(x)

    dfs([], set())
    return result
```

### 8. Topological sort (post-order DFS on DAG)

Run DFS; after finishing each node, append it to the result. Reverse the result for topological order.

```python
def topological_sort(graph: dict) -> list[int]:
    """Returns topological order (reverse of DFS post-order). Assumes DAG."""
    result: list[int] = []
    visited = set()

    def dfs(node: int) -> None:
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)

    for v in graph:
        if v not in visited:
            dfs(v)
    result.reverse()
    return result
```

### 9. Tree DFS (preorder / inorder / postorder)

Same idea as [Tree Algorithm](#tree-algorithm): recurse on left and right; process node before, between, or after children.

```python
def preorder(root: "TreeNode | None") -> list[int]:
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root: "TreeNode | None") -> list[int]:
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root: "TreeNode | None") -> list[int]:
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

### Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Graph vs tree** | In graphs, always track `visited` to avoid infinite loops; in trees no need (no cycles). |
| **Iterative DFS** | Push neighbors in **reverse** order if you want the same order as recursive DFS. |
| **Backtracking** | Restore state (e.g. `path.pop()`, `used.remove(x)`) after the recursive call. |
| **Directed cycle** | Use three states (unvisited / in stack / done); cycle iff edge to “in stack” node. |
| **Topological sort** | Only valid on DAGs; run post-order DFS and reverse, or use in-degree queue (BFS). |

### Related sections and problems

- Tree traversals: [Tree Algorithm](#tree-algorithm), [Binary Tree Algorithm](#binary-tree-algorithm).
- BFS and shortest paths: [Graph Algorithms](#graph-algorithms).
- Backtracking paradigms: [Algorithm Design Paradigms](#algorithm-design-paradigms).
- LeetCode-style problems: Number of Islands, Clone Graph, Course Schedule, Subsets, Permutations, Word Search (see [Solved Problems Index](#solved-problems-index)).

---

## BFS Algorithm

**Breadth-First Search (BFS)** explores all nodes at the current depth (or distance) before moving to the next level. It is implemented with a **queue** (FIFO): process a node, then enqueue its neighbors. BFS naturally gives **shortest path** (in number of edges) in unweighted graphs and **level-order** traversal in trees.

### When to use

| Scenario | Use BFS when |
|----------|----------------|
| **Shortest path (unweighted)** | Minimum number of edges from source to target. |
| **Level order** | Process tree/graph by layers (e.g. binary tree level-order). |
| **Nearest reachable** | Find closest node satisfying a condition (e.g. nearest 0 in a matrix). |
| **Multi-source** | Start from multiple nodes (e.g. all 0s); first time a node is reached is shortest. |
| **Topological sort (Kahn)** | Process nodes by in-degree; no DFS stack. |

### Complexity (typical)

| Setting | Time | Space |
|---------|------|--------|
| Tree, n nodes | O(n) | O(w) queue, w = max level width |
| Graph, V vertices, E edges | O(V + E) | O(V) visited + queue |
| 2D grid R×C | O(R·C) | O(R·C) or O(min(R,C)) for queue |

### 1. BFS on graph (adjacency list)

Enqueue start; while queue not empty, dequeue, mark visited, enqueue unvisited neighbors.

```python
from collections import deque

def bfs_graph(graph: dict, start: int) -> list[int]:
    """BFS traversal; returns nodes in level order."""
    visited = set()
    queue = deque([start])
    result = []
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return result

graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
print(bfs_graph(graph, 2))  # [2, 0, 3, 1]
```

### 2. Shortest path (unweighted): distance from source

Run BFS; when we first enqueue a node, that is a shortest path. Track distances by adding 1 when pushing neighbors.

```python
def shortest_path_lengths(graph: dict, start: int) -> dict[int, int]:
    """Distance (number of edges) from start to each reachable node."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        d = dist[node]
        for neighbor in graph.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = d + 1
                queue.append(neighbor)
    return dist
```

### 3. Shortest path: reconstruct path to target

Keep a parent (or predecessor) map; when you first reach the target, backtrack via parents to build the path.

```python
def shortest_path_to_target(graph: dict, start: int, target: int) -> list[int] | None:
    """One shortest path from start to target (list of nodes), or None."""
    if start == target:
        return [start]
    parent: dict[int, int] = {}
    queue = deque([start])
    parent[start] = -1
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == target:
                path = []
                cur = target
                while cur != -1:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            queue.append(neighbor)
    return None
```

### 4. Level-order traversal of binary tree

Process level by level: for each level, take current queue length, process that many nodes and enqueue their children.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: TreeNode | None) -> list[list[int]]:
    """Return list of levels (each level is a list of values)."""
    if not root:
        return []
    result: list[list[int]] = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### 5. BFS on 2D grid (shortest path in maze)

Move in 4 (or 8) directions; state is (r, c). Track visited and distance; first time reaching (tr, tc) gives shortest path.

```python
def shortest_path_grid(grid: list[list[int]], start: tuple[int, int], target: tuple[int, int]) -> int:
    """Shortest path length (steps) in grid; -1 if blocked (1) or unreachable. 0 = allowed."""
    R, C = len(grid), len(grid[0])
    sr, sc = start
    tr, tc = target
    if grid[sr][sc] == 1 or grid[tr][tc] == 1:
        return -1
    visited = {(sr, sc)}
    queue = deque([(sr, sc, 0)])
    while queue:
        r, c, steps = queue.popleft()
        if (r, c) == (tr, tc):
            return steps
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))
    return -1
```

### 6. Multi-source BFS (e.g. distance from nearest 0)

Initialize queue with all “source” cells (e.g. all 0s); run BFS. First time a cell is reached, that is its distance to the nearest source.

```python
def update_matrix(mat: list[list[int]]) -> list[list[int]]:
    """For each cell, distance to nearest 0. 0s are sources."""
    R, C = len(mat), len(mat[0])
    queue = deque()
    for r in range(R):
        for c in range(C):
            if mat[r][c] == 0:
                queue.append((r, c))
            else:
                mat[r][c] = -1
    while queue:
        r, c = queue.popleft()
        d = mat[r][c] if mat[r][c] >= 0 else 0
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and mat[nr][nc] == -1:
                mat[nr][nc] = d + 1
                queue.append((nr, nc))
    return mat
```

### 7. Topological sort (Kahn's algorithm)

Repeatedly remove nodes with in-degree 0; add to order and reduce in-degree of neighbors. Use a queue for current in-degree-0 nodes.

```python
def topological_sort_kahn(graph: dict) -> list[int]:
    """Topological order via in-degree; returns empty list if cycle."""
    in_degree = {v: 0 for v in graph}
    for v in graph:
        for u in graph[v]:
            in_degree[u] = in_degree.get(u, 0) + 1
    queue = deque([v for v in graph if in_degree[v] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return order if len(order) == len(graph) else []
```

### 8. Number of levels / depth (tree or graph from source)

BFS with level counting: each “wave” is one level; increment depth when finishing a level.

```python
def max_depth_bfs(root: TreeNode | None) -> int:
    """Max depth of binary tree via BFS (number of levels)."""
    if not root:
        return 0
    depth = 0
    queue = deque([root])
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return depth
```

### Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Queue** | Use `collections.deque` and `popleft()`; avoid list.pop(0) (O(n)). |
| **Visited** | Mark when enqueueing (or when dequeuing, but then allow duplicates in queue); be consistent. |
| **Level-by-level** | For “level” semantics, use `for _ in range(len(queue))` in the inner loop. |
| **Shortest path** | Only in **unweighted** graphs; for weights use [Graph Algorithms](#graph-algorithms) (e.g. Dijkstra). |
| **Multi-source** | Initialize queue with all sources; same BFS gives “distance to nearest source”. |

### Related sections and problems

- DFS: [DFS Algorithm](#dfs-algorithm). Trees: [Tree Algorithm](#tree-algorithm), [Binary Tree Algorithm](#binary-tree-algorithm).
- Weighted shortest paths: [Graph Algorithms](#graph-algorithms) (Dijkstra, etc.).
- LeetCode-style problems: Binary Tree Level Order Traversal, Shortest Path in Binary Matrix, 01 Matrix, Course Schedule II, Rotting Oranges (see [Solved Problems Index](#solved-problems-index)).

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

Graph algorithms address **traversal**, **connectivity**, **shortest paths**, **minimum spanning trees**, **topological order**, and **strongly connected components**. Graphs are typically represented as **adjacency lists** (list of neighbors per node) or **adjacency matrices**; for sparse graphs adjacency list is preferred.

### Representations

| Representation | Space | Edge lookup | Iterate neighbors |
|----------------|--------|-------------|-------------------|
| Adjacency list | O(V + E) | O(degree) | O(degree) |
| Adjacency matrix | O(V²) | O(1) | O(V) |

**Directed vs undirected:** For undirected, store each edge in both endpoints’ lists (or use a single list and treat as bidirectional).

### When to use

| Goal | Algorithm / technique |
|------|------------------------|
| Traversal, explore all nodes | [BFS](#bfs-algorithm), [DFS](#dfs-algorithm) |
| Shortest path (unweighted) | BFS |
| Shortest path (non-negative weights) | Dijkstra |
| Shortest path (negative weights allowed) | Bellman-Ford |
| All-pairs shortest paths | Floyd-Warshall |
| Minimum spanning tree | Kruskal (Union-Find), Prim |
| Dependency / schedule order | Topological sort (DFS or Kahn) |
| Strongly connected components | Tarjan or Kosaraju (DFS) |
| Connectivity / cycle in undirected | Union-Find, DFS |

### Complexity overview

| Algorithm | Time | Space |
|-----------|------|--------|
| BFS / DFS | O(V + E) | O(V) |
| Dijkstra (binary heap) | O((V + E) log V) | O(V) |
| Bellman-Ford | O(V·E) | O(V) |
| Floyd-Warshall | O(V³) | O(V²) |
| Kruskal | O(E log E) | O(V) |
| Prim (binary heap) | O(E log V) | O(V) |
| Topological sort (DFS/Kahn) | O(V + E) | O(V) |

### 1. Dijkstra's shortest path (non-negative weights)

Single-source shortest paths using a min-heap. When a node is popped, its distance is finalized; relax each outgoing edge.

```python
import heapq

def dijkstra(graph: dict, start: int) -> dict[int, float]:
    """graph: node -> list of (neighbor, weight). Returns dist from start to each node."""
    dist = {node: float("inf") for node in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

# Weighted directed graph: node -> [(neighbor, weight)]
graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
print(dijkstra(graph, 0))  # {0: 0, 1: 3, 2: 1, 3: 4}
```

### 2. Bellman-Ford (negative weights, negative cycle detection)

Relax all edges V−1 times; one more pass to detect nodes reachable from a negative cycle.

```python
def bellman_ford(edges: list[tuple[int, int, float]], n: int, start: int) -> tuple[dict[int, float], bool]:
    """edges: (u, v, w). Returns (dist, has_negative_cycle_reachable_from_start)."""
    dist = {i: float("inf") for i in range(n)}
    dist[start] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    has_neg_cycle = False
    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            has_neg_cycle = True
            break
    return dist, has_neg_cycle
```

### 3. Floyd-Warshall (all-pairs shortest paths)

Initialize with direct edges; for each middle node k, try improving i→j via k.

```python
def floyd_warshall(n: int, edges: list[tuple[int, int, float]]) -> list[list[float]]:
    """edges: (u, v, w). Returns n×n dist matrix (inf = no path)."""
    dist = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
```

### 4. Kruskal's minimum spanning tree (Union-Find)

Sort edges by weight; add each edge if it connects two different components (no cycle).

```python
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

def kruskal(n: int, edges: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """edges: (u, v, weight). Returns MST edge list."""
    edges = sorted(edges, key=lambda e: e[2])
    uf = UnionFind(n)
    mst: list[tuple[int, int, float]] = []
    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
    return mst
```

### 5. Prim's minimum spanning tree (priority queue)

Start from a vertex; repeatedly add the lightest edge from the current cut (tree vs rest). Similar to Dijkstra but key = edge weight, not path length.

```python
def prim(n: int, adj: dict[int, list[tuple[int, float]]]) -> list[tuple[int, int, float]]:
    """adj: node -> [(neighbor, weight)]. Returns MST edges (u, v, w)."""
    if not adj:
        return []
    start = next(iter(adj))
    in_mst = {start}
    pq: list[tuple[float, int, int]] = []  # (w, u, v) where v not in mst
    for v, w in adj.get(start, []):
        heapq.heappush(pq, (w, start, v))
    mst: list[tuple[int, int, float]] = []
    while pq and len(in_mst) < n:
        w, u, v = heapq.heappop(pq)
        if v in in_mst:
            continue
        in_mst.add(v)
        mst.append((u, v, w))
        for nxt, nw in adj.get(v, []):
            if nxt not in in_mst:
                heapq.heappush(pq, (nw, v, nxt))
    return mst
```

### 6. Topological sort (DFS post-order)

Valid only on DAGs. Run DFS; append node to result when finishing; reverse for topological order.

```python
def topological_sort_dfs(graph: dict[int, list[int]]) -> list[int]:
    visited = set()
    result: list[int] = []

    def dfs(u: int) -> None:
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                dfs(v)
        result.append(u)

    for u in graph:
        if u not in visited:
            dfs(u)
    result.reverse()
    return result

graph = {5: [2, 0], 4: [0, 1], 2: [3], 3: [1], 1: [], 0: []}
print(topological_sort_dfs(graph))  # e.g. [5, 4, 2, 3, 1, 0]
```

### 7. Topological sort (Kahn's algorithm, in-degree)

Process nodes with in-degree 0; remove them and update in-degrees of neighbors.

```python
from collections import deque

def topological_sort_kahn(graph: dict[int, list[int]]) -> list[int]:
    """Returns topological order, or empty if cycle."""
    in_deg = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_deg[v] = in_deg.get(v, 0) + 1
    q = deque([u for u in graph if in_deg[u] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    return order if len(order) == len(graph) else []
```

### 8. Union-Find (disjoint set) for connectivity

Path compression and union by rank give near-constant time per operation.

```python
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

# Example: count components after adding edges
uf = UnionFind(5)
uf.union(0, 1)
uf.union(2, 3)
print(uf.find(0) == uf.find(1))  # True
print(uf.find(0) == uf.find(2))  # False
```

### 9. Cycle detection in undirected graph (DFS)

If you see an edge to an already-visited node that is not the parent, there is a cycle.

```python
def has_cycle_undirected(graph: dict[int, list[int]]) -> bool:
    visited = set()

    def dfs(u: int, parent: int | None) -> bool:
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                if dfs(v, u):
                    return True
            elif v != parent:
                return True
        return False

    for u in graph:
        if u not in visited and dfs(u, None):
            return True
    return False
```

### 10. Number of connected components (undirected)

Run DFS or BFS from each unvisited node; count starts.

```python
def count_components(n: int, edges: list[tuple[int, int]]) -> int:
    """n nodes, edges as (u, v). Returns number of connected components."""
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()

    def dfs(u: int) -> None:
        visited.add(u)
        for v in adj[u]:
            if v not in visited:
                dfs(v)

    count = 0
    for i in range(n):
        if i not in visited:
            count += 1
            dfs(i)
    return count
```

### Implementation notes and pitfalls

| Topic | Recommendation |
|--------|-----------------|
| **Dijkstra** | Only for non-negative weights; use a heap and skip stale entries (d > dist[u]). |
| **Bellman-Ford** | Use for negative weights or to detect negative cycles; relax V−1 times. |
| **Floyd-Warshall** | O(V³); use when you need all-pairs or when V is small. |
| **Kruskal vs Prim** | Kruskal: sort edges, Union-Find. Prim: like Dijkstra with edge weight as key. |
| **Topological sort** | Graph must be a DAG; Kahn gives order and detects cycle (order length < V). |
| **Directed cycle** | Use DFS with three states (see [DFS Algorithm](#dfs-algorithm)). |

### Related sections and problems

- Traversal: [DFS Algorithm](#dfs-algorithm), [BFS Algorithm](#bfs-algorithm).
- Trees: [Tree Algorithm](#tree-algorithm), [Binary Tree Algorithm](#binary-tree-algorithm).
- Data structures: [Fundamental Data Structures](#fundamental-data-structures).
- LeetCode-style problems: Number of Connected Components, Course Schedule, Network Delay Time, Cheapest Flights Within K Stops, Min Cost to Connect All Points, Critical Connections (see [Solved Problems Index](#solved-problems-index)).

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
