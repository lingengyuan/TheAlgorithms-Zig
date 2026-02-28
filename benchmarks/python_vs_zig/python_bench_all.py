#!/usr/bin/env python3
"""Python vs Zig benchmark harness (Python side, alignable algorithms)."""

from __future__ import annotations

import heapq
import math
import os
import time
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable

MASK_64 = (1 << 64) - 1
BASE = 256
MOD = 1_000_003


def signed_i64_to_u64(value: int) -> int:
    return value & MASK_64


def checksum_ints(values: list[int]) -> int:
    if not values:
        return 0
    return (
        signed_i64_to_u64(values[0])
        + signed_i64_to_u64(values[len(values) // 2])
        + signed_i64_to_u64(values[-1])
        + len(values)
    ) & MASK_64


def checksum_bytes(values: str | bytes) -> int:
    if not values:
        return 0
    if isinstance(values, str):
        values = values.encode("utf-8")
    mid = values[len(values) // 2]
    return (values[0] + mid + values[-1] + len(values)) & MASK_64


def checksum_bool(value: bool) -> int:
    return 1 if value else 0


def checksum_u(value: int) -> int:
    return value & MASK_64


def generate_int_data(n: int) -> list[int]:
    return [((i * 48_271 + 12_345) % 1_000_003) - 500_000 for i in range(n)]


def generate_non_negative_data(n: int) -> list[int]:
    return [((i * 48_271 + 12_345) % 100_000) for i in range(n)]


def generate_sorted_data(n: int) -> list[int]:
    return [i * 2 for i in range(n)]


def generate_search_queries(query_count: int, n: int) -> list[int]:
    return [(((i * 97) + 31) % n) * 2 for i in range(query_count)]


def generate_u64_data(n: int) -> list[int]:
    return [((i * 73) + 19) % 1_000_000 for i in range(n)]


def generate_ascii_string(n: int, mul: int, add: int) -> str:
    base = ord("a")
    return "".join(chr(base + (((i * mul) + add) % 26)) for i in range(n))


def generate_matrix_data(n: int, mul: int, add: int, mod_val: int, shift: int) -> list[int]:
    return [(((i * mul) + add) % mod_val) - shift for i in range(n)]


def bubble_sort(arr: list[int]) -> None:
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def insertion_sort(arr: list[int]) -> None:
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    out: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def quick_sort(arr: list[int]) -> None:
    def partition(low: int, high: int) -> int:
        pivot = arr[high]
        i = low
        for j in range(low, high):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[high] = arr[high], arr[i]
        return i

    def quick(low: int, high: int) -> None:
        if low >= high:
            return
        p = partition(low, high)
        quick(low, p - 1)
        quick(p + 1, high)

    if arr:
        quick(0, len(arr) - 1)


def heap_sort(arr: list[int]) -> None:
    n = len(arr)

    def heapify(size: int, root: int) -> None:
        largest = root
        left = 2 * root + 1
        right = 2 * root + 2
        if left < size and arr[left] > arr[largest]:
            largest = left
        if right < size and arr[right] > arr[largest]:
            largest = right
        if largest != root:
            arr[root], arr[largest] = arr[largest], arr[root]
            heapify(size, largest)

    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(i, 0)


def radix_sort(arr: list[int]) -> list[int]:
    if not arr:
        return []
    negatives = [-x for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]

    def radix_sort_unsigned(nums: list[int]) -> list[int]:
        if len(nums) <= 1:
            return nums[:]
        out = nums[:]
        exp = 1
        max_val = max(out)
        while exp <= max_val:
            buckets = [0] * 10
            for v in out:
                buckets[(v // exp) % 10] += 1
            for i in range(1, 10):
                buckets[i] += buckets[i - 1]
            tmp = [0] * len(out)
            for i in range(len(out) - 1, -1, -1):
                digit = (out[i] // exp) % 10
                buckets[digit] -= 1
                tmp[buckets[digit]] = out[i]
            out = tmp
            exp *= 10
        return out

    neg_sorted = radix_sort_unsigned(negatives)
    pos_sorted = radix_sort_unsigned(positives)
    return [-x for x in reversed(neg_sorted)] + pos_sorted


def bucket_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr[:]
    lo = min(arr)
    hi = max(arr)
    if lo == hi:
        return arr[:]
    bucket_count = max(1, int(math.isqrt(len(arr))))
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    denominator = (hi - lo) + 1
    for v in arr:
        idx = ((v - lo) * bucket_count) // denominator
        if idx >= bucket_count:
            idx = bucket_count - 1
        buckets[idx].append(v)
    out: list[int] = []
    for b in buckets:
        insertion_sort(b)
        out.extend(b)
    return out


def selection_sort(arr: list[int]) -> None:
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]


def shell_sort(arr: list[int]) -> None:
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2


def counting_sort(arr: list[int]) -> list[int]:
    if not arr:
        return []
    lo = min(arr)
    hi = max(arr)
    counts = [0] * (hi - lo + 1)
    for v in arr:
        counts[v - lo] += 1
    out: list[int] = []
    for i, c in enumerate(counts):
        if c:
            out.extend([i + lo] * c)
    return out


def cocktail_shaker_sort(arr: list[int]) -> None:
    start = 0
    end = len(arr) - 1
    swapped = True
    while swapped:
        swapped = False
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        start += 1


def gnome_sort(arr: list[int]) -> None:
    i = 0
    n = len(arr)
    while i < n:
        if i == 0 or arr[i] >= arr[i - 1]:
            i += 1
        else:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i -= 1


def linear_search(items: list[int], target: int) -> int:
    for i, v in enumerate(items):
        if v == target:
            return i
    return -1


def binary_search(items: list[int], target: int) -> int:
    low = 0
    high = len(items) - 1
    while low <= high:
        mid = low + (high - low) // 2
        v = items[mid]
        if v == target:
            return mid
        if v < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def exponential_search(items: list[int], target: int) -> int:
    if not items:
        return -1
    if items[0] == target:
        return 0
    bound = 1
    while bound < len(items) and items[bound] < target:
        bound *= 2
    left = bound // 2
    right = min(bound, len(items) - 1)
    low = left
    high = right
    while low <= high:
        mid = low + (high - low) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def interpolation_search(items: list[int], target: int) -> int:
    if not items:
        return -1
    low = 0
    high = len(items) - 1
    while low <= high and items[low] <= target <= items[high]:
        if items[high] == items[low]:
            return low if items[low] == target else -1
        pos = low + ((high - low) * (target - items[low])) // (items[high] - items[low])
        if pos < low or pos > high:
            return -1
        if items[pos] == target:
            return pos
        if items[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    return -1


def jump_search(items: list[int], target: int) -> int:
    n = len(items)
    if n == 0:
        return -1
    step = int(math.isqrt(n))
    prev = 0
    while items[min(step, n) - 1] < target:
        prev = step
        step += int(math.isqrt(n))
        if prev >= n:
            return -1
    while prev < min(step, n):
        if items[prev] == target:
            return prev
        prev += 1
    return -1


def ternary_search(items: list[int], target: int) -> int:
    if not items:
        return -1
    left = 0
    right = len(items) - 1
    while left <= right:
        if right - left < 3:
            for i in range(left, right + 1):
                if items[i] == target:
                    return i
            return -1
        third = (right - left) // 3
        mid1 = left + third
        mid2 = right - third
        if items[mid1] == target:
            return mid1
        if items[mid2] == target:
            return mid2
        if target < items[mid1]:
            right = mid1 - 1
        elif target > items[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1
    return -1


def gcd(a: int, b: int) -> int:
    x = abs(a)
    y = abs(b)
    while y:
        x, y = y, x % y
    return x


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a) // gcd(a, b) * abs(b)


def fibonacci(n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def power(base: int, exponent: int) -> int:
    result = 1
    b = base
    e = exponent
    while e > 0:
        if e & 1:
            result *= b
        b *= b
        e >>= 1
    return result


def power_mod(base: int, exponent: int, modulus: int) -> int:
    if modulus == 0:
        raise ValueError("invalid modulus")
    if modulus == 1:
        return 0
    result = 1
    b = base % modulus
    e = exponent
    while e > 0:
        if e & 1:
            result = (result * b) % modulus
        b = (b * b) % modulus
        e >>= 1
    return result


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    flags = [True] * (limit + 1)
    flags[0] = flags[1] = False
    i = 2
    while i * i <= limit:
        if flags[i]:
            j = i * i
            while j <= limit:
                flags[j] = False
                j += i
        i += 1
    return [i for i, v in enumerate(flags) if v]


def collatz_steps(n: int) -> int:
    if n == 0:
        raise ValueError("invalid input")
    steps = 0
    current = n
    while current != 1:
        if current % 2 == 0:
            current //= 2
        else:
            current = 3 * current + 1
        steps += 1
    return steps


def extended_euclidean(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def modular_inverse(a: int, m: int) -> int:
    if m <= 1:
        raise ValueError("invalid modulus")
    g, x, _ = extended_euclidean(a, m)
    if g != 1:
        raise ValueError("no inverse")
    return x % m


def eulers_totient(n: int) -> int:
    if n == 0:
        return 0
    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1
    if x > 1:
        result -= result // x
    return result


def chinese_remainder_theorem(remainders: list[int], moduli: list[int]) -> int:
    if len(remainders) != len(moduli):
        raise ValueError("length mismatch")
    if not remainders:
        raise ValueError("empty")
    for m in moduli:
        if m <= 0:
            raise ValueError("invalid modulus")
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if math.gcd(moduli[i], moduli[j]) != 1:
                raise ValueError("not pairwise coprime")

    prod = 1
    for m in moduli:
        prod *= m

    total = 0
    for a, m in zip(remainders, moduli):
        ni = prod // m
        inv = modular_inverse(ni % m, m)
        total += (a % m) * ni * inv
    return total % prod


def binomial_coefficient(n: int, k: int) -> int:
    if k > n:
        return 0
    kk = min(k, n - k)
    result = 1
    for i in range(1, kk + 1):
        result = result * (n - kk + i) // i
    return result


def integer_square_root(n: int) -> int:
    if n < 2:
        return n
    x = n
    y = (x + n // x) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def encode_base26_word(value: int, length: int) -> str:
    chars = ["a"] * length
    x = value
    for i in range(length - 1, -1, -1):
        chars[i] = chr(ord("a") + (x % 26))
        x //= 26
    return "".join(chars)


class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end = False


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                return False
            node = nxt
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            nxt = node.children.get(ch)
            if nxt is None:
                return False
            node = nxt
        return True

    def delete(self, word: str) -> bool:
        def dfs(node: TrieNode, idx: int) -> tuple[bool, bool]:
            if idx == len(word):
                if not node.is_end:
                    return False, False
                node.is_end = False
                return True, not node.children

            ch = word[idx]
            child = node.children.get(ch)
            if child is None:
                return False, False

            removed, prune_child = dfs(child, idx + 1)
            if prune_child:
                del node.children[ch]
            return removed, (not node.is_end and not node.children)

        removed, _ = dfs(self.root, 0)
        return removed


def trie_workload(words: list[str]) -> int:
    trie = Trie()
    for word in words:
        trie.insert(word)

    present = sum(1 for word in words if trie.search(word))
    prefix_hits = sum(1 for word in words if trie.starts_with(word[:3]))
    removed = sum(1 for word in words[::5] if trie.delete(word))
    remain = sum(1 for word in words if trie.search(word))
    return (present + (prefix_hits * 3) + (removed * 5) + (remain * 7)) & MASK_64


class DisjointSetBench:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


def disjoint_set_workload(n: int) -> int:
    ds = DisjointSetBench(n)
    for i in range(0, n - 1, 2):
        ds.union(i, i + 1)
    for i in range(0, n - 3, 3):
        ds.union(i, i + 3)

    connected_hits = 0
    for i in range(20_000):
        a = ((i * 97) + 31) % n
        b = ((i * 53) + 17) % n
        if ds.connected(a, b):
            connected_hits += 1

    root_sum = 0
    for i in range(10_000):
        idx = ((i * 193) + 7) % n
        root_sum = (root_sum + ds.find(idx)) & MASK_64

    return (connected_hits + root_sum + (ds.components * 11)) & MASK_64


class AvlNode:
    __slots__ = ("key", "height", "left", "right")

    def __init__(self, key: int) -> None:
        self.key = key
        self.height = 1
        self.left: AvlNode | None = None
        self.right: AvlNode | None = None


class AvlTreeBench:
    def __init__(self) -> None:
        self.root: AvlNode | None = None
        self.size = 0

    @staticmethod
    def _h(node: AvlNode | None) -> int:
        return node.height if node else 0

    @staticmethod
    def _update(node: AvlNode) -> None:
        node.height = max(AvlTreeBench._h(node.left), AvlTreeBench._h(node.right)) + 1

    @staticmethod
    def _balance(node: AvlNode) -> int:
        return AvlTreeBench._h(node.left) - AvlTreeBench._h(node.right)

    @staticmethod
    def _rotate_left(x: AvlNode) -> AvlNode:
        y = x.right
        assert y is not None
        t2 = y.left
        y.left = x
        x.right = t2
        AvlTreeBench._update(x)
        AvlTreeBench._update(y)
        return y

    @staticmethod
    def _rotate_right(y: AvlNode) -> AvlNode:
        x = y.left
        assert x is not None
        t2 = x.right
        x.right = y
        y.left = t2
        AvlTreeBench._update(y)
        AvlTreeBench._update(x)
        return x

    @staticmethod
    def _rebalance(node: AvlNode) -> AvlNode:
        bf = AvlTreeBench._balance(node)
        if bf > 1:
            assert node.left is not None
            if AvlTreeBench._balance(node.left) < 0:
                node.left = AvlTreeBench._rotate_left(node.left)
            return AvlTreeBench._rotate_right(node)
        if bf < -1:
            assert node.right is not None
            if AvlTreeBench._balance(node.right) > 0:
                node.right = AvlTreeBench._rotate_right(node.right)
            return AvlTreeBench._rotate_left(node)
        return node

    def insert(self, key: int) -> bool:
        inserted = False

        def rec(node: AvlNode | None) -> AvlNode:
            nonlocal inserted
            if node is None:
                inserted = True
                return AvlNode(key)
            if key < node.key:
                node.left = rec(node.left)
            elif key > node.key:
                node.right = rec(node.right)
            else:
                return node
            self._update(node)
            return self._rebalance(node)

        self.root = rec(self.root)
        if inserted:
            self.size += 1
        return inserted

    def contains(self, key: int) -> bool:
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False

    def remove(self, key: int) -> bool:
        removed = False

        def min_node(node: AvlNode) -> AvlNode:
            cur = node
            while cur.left is not None:
                cur = cur.left
            return cur

        def rec(node: AvlNode | None) -> AvlNode | None:
            nonlocal removed
            if node is None:
                return None
            if key < node.key:
                node.left = rec(node.left)
            elif key > node.key:
                node.right = rec(node.right)
            else:
                removed = True
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                succ = min_node(node.right)
                node.key = succ.key
                node.right = rec_delete_key(node.right, succ.key)

            self._update(node)
            return self._rebalance(node)

        def rec_delete_key(node: AvlNode | None, target: int) -> AvlNode | None:
            if node is None:
                return None
            if target < node.key:
                node.left = rec_delete_key(node.left, target)
            elif target > node.key:
                node.right = rec_delete_key(node.right, target)
            else:
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                succ = min_node(node.right)
                node.key = succ.key
                node.right = rec_delete_key(node.right, succ.key)
            self._update(node)
            return self._rebalance(node)

        self.root = rec(self.root)
        if removed:
            self.size -= 1
        return removed

    def inorder(self) -> list[int]:
        out: list[int] = []

        def walk(node: AvlNode | None) -> None:
            if node is None:
                return
            walk(node.left)
            out.append(node.key)
            walk(node.right)

        walk(self.root)
        return out


def avl_tree_workload(values: list[int], queries: list[int]) -> int:
    tree = AvlTreeBench()
    for v in values:
        tree.insert(v)

    hits_before = sum(1 for q in queries if tree.contains(q))
    removed = sum(1 for v in values[::4] if tree.remove(v))
    hits_after = sum(1 for q in queries if tree.contains(q))

    ordered = tree.inorder()
    if not ordered:
        inorder_checksum = 0
    else:
        inorder_checksum = (
            signed_i64_to_u64(ordered[0])
            + signed_i64_to_u64(ordered[len(ordered) // 2])
            + signed_i64_to_u64(ordered[-1])
            + len(ordered)
        ) & MASK_64

    return (hits_before + (hits_after * 3) + (removed * 5) + (inorder_checksum * 7)) & MASK_64


def max_heap_workload(values: list[int]) -> int:
    heap = [-v for v in values]
    heapq.heapify(heap)
    out = [-heapq.heappop(heap) for _ in range(len(heap))]
    return checksum_ints(out)


def priority_queue_workload(n: int) -> int:
    heap: list[tuple[int, int]] = []
    for i in range(n):
        heapq.heappush(heap, ((((i * 97) + 31) % 1000), i))
    out = [heapq.heappop(heap)[1] for _ in range(len(heap))]
    return checksum_ints(out)


def climbing_stairs(n: int) -> int:
    if n <= 0:
        raise ValueError("invalid input")
    if n == 1:
        return 1
    prev2 = 1
    prev1 = 1
    i = 0
    while i < n - 1:
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
        i += 1
    return prev1


def fibonacci_dp(n: int) -> list[int]:
    memo = [2**64 - 1] * (n + 1)
    memo[0] = 0
    if n >= 1:
        memo[1] = 1

    def fm(x: int) -> int:
        if memo[x] != 2**64 - 1:
            return memo[x]
        memo[x] = fm(x - 1) + fm(x - 2)
        return memo[x]

    return [fm(i) for i in range(n + 1)]


def coin_change_ways(coins: list[int], amount: int) -> int:
    if amount < 0:
        return 0
    if amount == 0:
        return 1
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for v in range(c, amount + 1):
            dp[v] += dp[v - c]
    return dp[amount]


def max_subarray_sum(items: list[int], allow_empty_subarrays: bool) -> int:
    if not items:
        return 0
    best = 0 if allow_empty_subarrays else -(1 << 63)
    current = 0
    for v in items:
        current = max(0 if allow_empty_subarrays else v, current + v)
        best = max(best, current)
    return best


def lcs_length(a: str, b: str) -> int:
    rows = len(a) + 1
    cols = len(b) + 1
    table = [0] * (rows * cols)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            idx = i * cols + j
            up = (i - 1) * cols + j
            left = i * cols + (j - 1)
            diag = (i - 1) * cols + (j - 1)
            if a[i - 1] == b[j - 1]:
                table[idx] = table[diag] + 1
            else:
                table[idx] = table[up] if table[up] >= table[left] else table[left]
    return table[len(a) * cols + len(b)]


def edit_distance(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    cols = n + 1
    dp = [0] * ((m + 1) * cols)
    for i in range(m + 1):
        dp[i * cols] = i
    for j in range(n + 1):
        dp[j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i * cols + j] = dp[(i - 1) * cols + (j - 1)]
            else:
                ins = dp[i * cols + (j - 1)]
                dele = dp[(i - 1) * cols + j]
                rep = dp[(i - 1) * cols + (j - 1)]
                dp[i * cols + j] = 1 + min(ins, dele, rep)
    return dp[m * cols + n]


def knapsack(capacity: int, weights: list[int], values: list[int]) -> int:
    n = len(weights)
    cols = capacity + 1
    dp = [0] * ((n + 1) * cols)
    for i in range(1, n + 1):
        wi = weights[i - 1]
        vi = values[i - 1]
        cur = i * cols
        prev = (i - 1) * cols
        for w in range(1, capacity + 1):
            if wi <= w:
                take = vi + dp[prev + (w - wi)]
                skip = dp[prev + w]
                dp[cur + w] = take if take > skip else skip
            else:
                dp[cur + w] = dp[prev + w]
    return dp[n * cols + capacity]


def longest_increasing_subsequence(arr: list[int]) -> int:
    tails: list[int] = []
    for value in arr:
        idx = bisect_left(tails, value)
        if idx == len(tails):
            tails.append(value)
        else:
            tails[idx] = value
    return len(tails)


def rod_cutting(prices: list[int], length: int) -> int:
    if length == 0:
        return 0
    dp = [0] * (length + 1)
    for rod_len in range(1, length + 1):
        best = None
        for cut in range(1, rod_len + 1):
            if cut > len(prices):
                continue
            cand = prices[cut - 1] + dp[rod_len - cut]
            if best is None or cand > best:
                best = cand
        dp[rod_len] = 0 if best is None else best
    return dp[length]


def matrix_chain_multiplication(dims: list[int]) -> int:
    if len(dims) <= 2:
        return 0
    n = len(dims) - 1
    inf = (1 << 63) - 1
    dp = [0] * (n * n)
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            best = inf
            for k in range(i, j):
                cost = (
                    dp[i * n + k]
                    + dp[(k + 1) * n + j]
                    + (dims[i] * dims[k + 1] * dims[j + 1])
                )
                if cost < best:
                    best = cost
            dp[i * n + j] = best
    return dp[n - 1]


def palindrome_partition_min_cuts(text: str) -> int:
    n = len(text)
    if n == 0:
        return 0

    is_pal = [False] * (n * n)
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if text[i] == text[j] and (j - i <= 1 or is_pal[(i + 1) * n + (j - 1)]):
                is_pal[i * n + j] = True

    cuts = [0] * n
    for end in range(n):
        if is_pal[end]:
            cuts[end] = 0
            continue
        best = end
        for prev in range(end):
            if is_pal[(prev + 1) * n + end]:
                cand = cuts[prev] + 1
                if cand < best:
                    best = cand
        cuts[end] = best
    return cuts[-1]


def word_break(text: str, dictionary: list[str]) -> bool:
    n = len(text)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for word in dictionary:
            wl = len(word)
            if wl <= i and dp[i - wl] and text[i - wl : i] == word:
                dp[i] = True
                break
    return dp[n]


def catalan_number(n: int) -> int:
    if n == 0:
        return 1
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        total = 0
        for j in range(i):
            total += dp[j] * dp[i - 1 - j]
        dp[i] = total
    return dp[n]


def bfs(adj: list[list[int]], start: int) -> list[int]:
    n = len(adj)
    if start >= n:
        return []
    visited = [False] * n
    q = [start]
    head = 0
    visited[start] = True
    out: list[int] = []
    while head < len(q):
        cur = q[head]
        head += 1
        out.append(cur)
        for nb in adj[cur]:
            if 0 <= nb < n and not visited[nb]:
                visited[nb] = True
                q.append(nb)
    return out


def dfs(adj: list[list[int]], start: int) -> list[int]:
    n = len(adj)
    if start >= n:
        return []
    visited = [False] * n
    stack = [start]
    out: list[int] = []
    while stack:
        cur = stack.pop()
        if visited[cur]:
            continue
        visited[cur] = True
        out.append(cur)
        neighbors = adj[cur]
        for nb in reversed(neighbors):
            if 0 <= nb < n and not visited[nb]:
                stack.append(nb)
    return out


def dijkstra(adj: list[list[tuple[int, int]]], start: int) -> list[int]:
    n = len(adj)
    if start >= n:
        return []
    inf = (1 << 63) - 1
    dist = [inf] * n
    visited = [False] * n
    dist[start] = 0

    for _ in range(n):
        best = inf
        best_idx = -1
        for i, d in enumerate(dist):
            if not visited[i] and d < best:
                best = d
                best_idx = i
        if best_idx < 0:
            break

        visited[best_idx] = True
        base = dist[best_idx]
        for nb, weight in adj[best_idx]:
            if 0 <= nb < n and not visited[nb]:
                cand = base + weight
                if cand < dist[nb]:
                    dist[nb] = cand

    return dist


def bellman_ford(vertex_count: int, edges: list[tuple[int, int, int]], start: int) -> list[int]:
    if start >= vertex_count:
        return []
    inf = (1 << 63) - 1
    dist = [inf] * vertex_count
    dist[start] = 0

    for _ in range(vertex_count - 1):
        changed = False
        for u, v, w in edges:
            if not (0 <= u < vertex_count and 0 <= v < vertex_count):
                continue
            if dist[u] == inf:
                continue
            cand = dist[u] + w
            if cand < dist[v]:
                dist[v] = cand
                changed = True
        if not changed:
            break

    for u, v, w in edges:
        if not (0 <= u < vertex_count and 0 <= v < vertex_count):
            continue
        if dist[u] == inf:
            continue
        if dist[u] + w < dist[v]:
            raise ValueError("negative cycle")

    return dist


def topological_sort(adj: list[list[int]]) -> list[int]:
    n = len(adj)
    if n == 0:
        return []
    indegree = [0] * n
    for neighbors in adj:
        for nb in neighbors:
            if 0 <= nb < n:
                indegree[nb] += 1

    queue = [i for i, deg in enumerate(indegree) if deg == 0]
    head = 0
    out: list[int] = []
    while head < len(queue):
        cur = queue[head]
        head += 1
        out.append(cur)
        for nb in adj[cur]:
            if 0 <= nb < n:
                indegree[nb] -= 1
                if indegree[nb] == 0:
                    queue.append(nb)

    if len(out) != n:
        raise ValueError("cycle")
    return out


def floyd_warshall(matrix: list[int], n: int, inf: int) -> list[int]:
    if n == 0:
        return []
    if len(matrix) != n * n:
        raise ValueError("invalid matrix size")
    dist = matrix.copy()
    for k in range(n):
        k_base = k * n
        for i in range(n):
            i_base = i * n
            ik = dist[i_base + k]
            if ik == inf:
                continue
            for j in range(n):
                kj = dist[k_base + j]
                if kj == inf:
                    continue
                cand = ik + kj
                idx = i_base + j
                if cand < dist[idx]:
                    dist[idx] = cand
    return dist


def detect_cycle(adj: list[list[int]]) -> bool:
    n = len(adj)
    state = [0] * n  # 0=unvisited, 1=visiting, 2=done

    def dfs(node: int) -> bool:
        if state[node] == 1:
            return True
        if state[node] == 2:
            return False
        state[node] = 1
        for nb in adj[node]:
            if 0 <= nb < n and dfs(nb):
                return True
        state[node] = 2
        return False

    for i in range(n):
        if state[i] == 0 and dfs(i):
            return True
    return False


def connected_components(adj: list[list[int]]) -> int:
    n = len(adj)
    visited = [False] * n
    count = 0
    for i in range(n):
        if visited[i]:
            continue
        count += 1
        visited[i] = True
        stack = [i]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if 0 <= nb < n and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
    return count


def kruskal_mst_weight(vertex_count: int, edges: list[tuple[int, int, int]]) -> int:
    if vertex_count == 0:
        return 0

    parent = list(range(vertex_count))
    rank = [0] * vertex_count

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    valid = sorted((e for e in edges if 0 <= e[0] < vertex_count and 0 <= e[1] < vertex_count), key=lambda e: e[2])
    total = 0
    used = 0
    for u, v, w in valid:
        if union(u, v):
            total += w
            used += 1
            if used == vertex_count - 1:
                break

    if used != vertex_count - 1:
        raise ValueError("disconnected")
    return total


def prim_mst_weight(adj: list[list[tuple[int, int]]], start: int = 0) -> int:
    n = len(adj)
    if n == 0:
        return 0
    if start >= n:
        raise ValueError("invalid start")

    inf = (1 << 63) - 1
    key = [inf] * n
    used = [False] * n
    key[start] = 0
    total = 0

    for _ in range(n):
        best = inf
        u = -1
        for i in range(n):
            if not used[i] and key[i] < best:
                best = key[i]
                u = i
        if u < 0 or key[u] == inf:
            raise ValueError("disconnected")
        used[u] = True
        total += key[u]

        for nb, weight in adj[u]:
            if 0 <= nb < n and not used[nb] and weight < key[nb]:
                key[nb] = weight

    return total


def is_power_of_two(n: int) -> bool:
    return n != 0 and (n & (n - 1)) == 0


def count_set_bits(n: int) -> int:
    count = 0
    x = n
    while x != 0:
        x &= x - 1
        count += 1
    return count


def find_unique_number(arr: list[int]) -> int:
    if not arr:
        return 0
    r = 0
    for v in arr:
        r ^= v
    return r


def reverse_bits(n: int) -> int:
    r = 0
    x = n & 0xFFFFFFFF
    for _ in range(32):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def missing_number(nums: list[int]) -> int:
    r = len(nums)
    for i, v in enumerate(nums):
        r ^= i ^ v
    return r


def is_power_of_four(n: int) -> bool:
    return n != 0 and (n & (n - 1)) == 0 and (n & 0x5555555555555555) != 0


def decimal_to_binary(n: int) -> str:
    if n == 0:
        return "0"
    out: list[str] = []
    x = n
    while x > 0:
        out.append("1" if (x & 1) else "0")
        x >>= 1
    return "".join(reversed(out))


def binary_to_decimal(bin_s: str) -> int:
    if not bin_s:
        raise ValueError("empty")
    negative = bin_s[0] == "-"
    s = bin_s[1:] if negative else bin_s
    if not s:
        raise ValueError("invalid")
    result = 0
    for c in s:
        if c not in ("0", "1"):
            raise ValueError("invalid")
        result = result * 2 + (ord(c) - ord("0"))
    return -result if negative else result


def decimal_to_hex(n: int) -> str:
    if n == 0:
        return "0"
    return format(n, "x")


def binary_to_hex(bin_s: str) -> str:
    if not bin_s:
        raise ValueError("empty")
    for c in bin_s:
        if c not in ("0", "1"):
            raise ValueError("invalid")
    rem = len(bin_s) % 4
    padding = 0 if rem == 0 else 4 - rem
    padded = ("0" * padding) + bin_s
    out = []
    for i in range(0, len(padded), 4):
        nibble = int(padded[i : i + 4], 2)
        out.append("0123456789abcdef"[nibble])
    return "".join(out)


def is_palindrome(s: str) -> bool:
    lo = 0
    hi = len(s) - 1
    while lo < hi:
        if s[lo] != s[hi]:
            return False
        lo += 1
        hi -= 1
    return True


def reverse_words(sentence: str) -> str:
    words = [w for w in sentence.split(" ") if w]
    return " ".join(reversed(words))


def is_anagram(a: str, b: str) -> bool:
    counts = [0] * 128
    for c in a:
        if c == " ":
            continue
        x = c.lower()
        counts[ord(x)] += 1
    for c in b:
        if c == " ":
            continue
        x = c.lower()
        counts[ord(x)] -= 1
    return all(v == 0 for v in counts)


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("len mismatch")
    return sum(1 for x, y in zip(a, b) if x != y)


def naive_search(text: str, pattern: str) -> list[int]:
    if not pattern or len(pattern) > len(text):
        return []
    out = []
    limit = len(text) - len(pattern) + 1
    for i in range(limit):
        if text[i : i + len(pattern)] == pattern:
            out.append(i)
    return out


def kmp_search(text: str, pattern: str) -> int:
    if len(pattern) == 0:
        return 0
    if len(pattern) > len(text):
        return -1
    f = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[k] != pattern[i]:
            k = f[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        f[i] = k
    j = 0
    for i, c in enumerate(text):
        while j > 0 and pattern[j] != c:
            j = f[j - 1]
        if pattern[j] == c:
            j += 1
        if j == len(pattern):
            return i + 1 - len(pattern)
    return -1


def rabin_karp(text: str, pattern: str) -> bool:
    n = len(text)
    m = len(pattern)
    if m > n:
        return False
    if m == 0:
        return True
    base_pow = 1
    for _ in range(m - 1):
        base_pow = (base_pow * BASE) % MOD
    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (p_hash * BASE + ord(pattern[i])) % MOD
        t_hash = (t_hash * BASE + ord(text[i])) % MOD
    for i in range(n - m + 1):
        if t_hash == p_hash and text[i : i + m] == pattern:
            return True
        if i < n - m:
            t_hash = (t_hash + MOD - (ord(text[i]) * base_pow) % MOD) % MOD
            t_hash = (t_hash * BASE + ord(text[i + m])) % MOD
    return False


def z_function(s: str) -> list[int]:
    n = len(s)
    z = [0] * n
    l = 0
    r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l = i
            r = i + z[i]
    return z


def levenshtein_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    m = len(a)
    n = len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(m):
        curr[0] = i + 1
        ai = a[i]
        for j in range(n):
            if ai == b[j]:
                curr[j + 1] = prev[j]
            else:
                x = prev[j]
                y = curr[j]
                z = prev[j + 1]
                best = y if y < z else z
                if x < best:
                    best = x
                curr[j + 1] = 1 + best
        prev, curr = curr, prev
    return prev[n]


def is_pangram(s: str) -> bool:
    seen = [False] * 26
    for c in s:
        if "a" <= c <= "z":
            seen[ord(c) - ord("a")] = True
        elif "A" <= c <= "Z":
            seen[ord(c) - ord("A")] = True
    return all(seen)


def max_profit(prices: list[int]) -> int:
    if not prices:
        return 0
    min_price = prices[0]
    profit = 0
    for p in prices[1:]:
        if p < min_price:
            min_price = p
        gain = p - min_price
        if gain > profit:
            profit = gain
    return profit


def minimum_coin_change(coins: list[int], amount: int) -> list[int]:
    out: list[int] = []
    rem = amount
    for c in coins:
        while rem >= c:
            out.append(c)
            rem -= c
    return out


def minimum_waiting_time(queries: list[int]) -> int:
    if len(queries) <= 1:
        return 0
    sorted_q = sorted(queries)
    total = 0
    n = len(sorted_q)
    for i, q in enumerate(sorted_q):
        total += q * (n - i - 1)
    return total


def fractional_knapsack(values: list[float], weights: list[float], capacity: float) -> float:
    if capacity <= 0 or not values:
        return 0.0
    items = list(zip(values, weights))
    items.sort(key=lambda x: x[0] / x[1], reverse=True)
    rem = capacity
    total = 0.0
    for value, weight in items:
        if rem <= 0:
            break
        if weight <= rem:
            total += value
            rem -= weight
        else:
            total += value * (rem / weight)
            rem = 0.0
    return total


def matrix_multiply(a: list[int], b: list[int], dim: int) -> list[int]:
    c = [0] * (dim * dim)
    for i in range(dim):
        for j in range(dim):
            total = 0
            for k in range(dim):
                total += a[i * dim + k] * b[k * dim + j]
            c[i * dim + j] = total
    return c


def transpose(mat: list[int], rows: int, cols: int) -> list[int]:
    out = [0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            out[c * rows + r] = mat[r * cols + c]
    return out


def rotate90(mat: list[int], n: int) -> None:
    for r in range(n):
        for c in range(r + 1, n):
            i = r * n + c
            j = c * n + r
            mat[i], mat[j] = mat[j], mat[i]
    for r in range(n):
        lo = 0
        hi = n - 1
        while lo < hi:
            i = r * n + lo
            j = r * n + hi
            mat[i], mat[j] = mat[j], mat[i]
            lo += 1
            hi -= 1


def spiral_order(mat: list[int], rows: int, cols: int) -> list[int]:
    out = [0] * (rows * cols)
    top = 0
    bottom = rows
    left = 0
    right = cols
    idx = 0
    while top < bottom and left < right:
        for c in range(left, right):
            out[idx] = mat[top * cols + c]
            idx += 1
        top += 1
        for r in range(top, bottom):
            out[idx] = mat[r * cols + (right - 1)]
            idx += 1
        right -= 1
        if top < bottom:
            c = right
            while c > left:
                c -= 1
                out[idx] = mat[(bottom - 1) * cols + c]
                idx += 1
            bottom -= 1
        if left < right:
            r = bottom
            while r > top:
                r -= 1
                out[idx] = mat[r * cols + left]
                idx += 1
            left += 1
    return out


def pascal_triangle(num_rows: int) -> list[list[int]]:
    tri: list[list[int]] = []
    for r in range(num_rows):
        row = [1] * (r + 1)
        if r >= 2:
            prev = tri[r - 1]
            for c in range(1, r):
                row[c] = prev[c - 1] + prev[c]
        tri.append(row)
    return tri


@dataclass
class BenchCase:
    name: str
    category: str
    iterations: int
    fn: Callable[[], int]


def bench_case(case: BenchCase) -> tuple[str, str, int, int, int, int]:
    _ = case.fn()  # warmup
    checksum = 0
    start = time.perf_counter_ns()
    for _ in range(case.iterations):
        checksum = (checksum + case.fn()) & MASK_64
    total_ns = time.perf_counter_ns() - start
    avg_ns = total_ns // case.iterations
    return case.name, case.category, case.iterations, total_ns, avg_ns, checksum


def extended_euclidean_checksum(pairs: list[tuple[int, int]]) -> int:
    total = 0
    for a, b in pairs:
        g, x, y = extended_euclidean(a, b)
        total = (
            total
            + signed_i64_to_u64(g)
            + signed_i64_to_u64(x)
            + signed_i64_to_u64(y)
        ) & MASK_64
    return total


def crt_many_checksum(systems: list[tuple[list[int], list[int]]]) -> int:
    total = 0
    for rem, mod in systems:
        total = (total + chinese_remainder_theorem(rem, mod)) & MASK_64
    return total


def main() -> None:
    bubble_base = generate_int_data(1_200)
    n2_base = generate_int_data(1_300)
    nlog_base = generate_int_data(28_000)
    non_neg_base = generate_non_negative_data(30_000)
    search_data = generate_sorted_data(40_000)
    search_queries = generate_search_queries(1_200, len(search_data))
    math_values = generate_u64_data(40_000)
    ext_pairs: list[tuple[int, int]] = []
    for i in range(0, 8_000, 2):
        a = int(math_values[i] % 200_000) - 100_000
        b = int(math_values[i + 1] % 200_000) - 100_000
        if a == 0 and b == 0:
            b = 1
        ext_pairs.append((a, b))

    modinv_pairs: list[tuple[int, int]] = []
    for i in range(2_000):
        m = ((i * 37) + 101) % 50_000 + 3
        if m % 2 == 0:
            m += 1
        a = ((i * 97) + 31) % m
        if a == 0:
            a = 1
        while math.gcd(a, m) != 1:
            a = (a + 1) % m
            if a == 0:
                a = 1
        modinv_pairs.append((a, m))

    totient_inputs = [int(v % 1_000_000) + 1 for v in math_values[:20_000]]

    crt_systems: list[tuple[list[int], list[int]]] = []
    crt_moduli = [3, 5, 7]
    for i in range(2_500):
        crt_systems.append(
            (
                [i % 3, (i * 2 + 1) % 5, (i * 3 + 2) % 7],
                crt_moduli,
            )
        )

    binom_pairs = [(((i * 19) + 20) % 47 + 20, ((i * 11) + 7) % 20 + 1) for i in range(3_000)]

    isqrt_inputs = [((i * 9_999_991) + 1_234_567_891) % ((1 << 63) - 1) for i in range(40_000)]
    trie_words = [encode_base26_word(i, 6) for i in range(12_000)]
    disjoint_set_n = 50_000
    avl_values = [((i * 73) + 19) % 50_000 for i in range(50_000)]
    avl_queries = [((i * 97) + 31) % 50_000 for i in range(20_000)]
    max_heap_values = generate_int_data(50_000)
    priority_queue_n = 60_000
    dp_array = generate_int_data(90_000)
    prices = [abs(x) % 1000 for x in generate_int_data(80_000)]
    waiting_queries = [x % 100 for x in generate_u64_data(50_000)]
    matrix_dim = 70
    matrix_a = generate_matrix_data(matrix_dim * matrix_dim, 31, 7, 41, 20)
    matrix_b = generate_matrix_data(matrix_dim * matrix_dim, 17, 11, 37, 18)
    transpose_rows = 180
    transpose_cols = 220
    transpose_mat = generate_matrix_data(transpose_rows * transpose_cols, 23, 5, 53, 26)
    rotate_n = 150
    rotate_mat = generate_matrix_data(rotate_n * rotate_n, 29, 3, 71, 35)
    spiral_rows = 160
    spiral_cols = 180
    spiral_mat = generate_matrix_data(spiral_rows * spiral_cols, 41, 9, 67, 33)
    text = generate_ascii_string(130_000, 7, 3)
    pattern = text[19_000:19_020]
    s1 = generate_ascii_string(420, 7, 3)
    s2 = generate_ascii_string(450, 11, 5)
    s3 = generate_ascii_string(100_000, 5, 1)
    s4 = generate_ascii_string(100_000, 9, 4)
    pangram_text = "The quick brown fox jumps over the lazy dog " * 8_000
    anagram_a = "This is a string " * 4_000
    anagram_b = "Is this a string " * 4_000
    fib_inputs = list(range(1, 91))
    fact_inputs = list(range(1, 21))
    power_pairs = [((i % 19) + 2, (i % 10) + 5) for i in range(2_000)]
    stairs_inputs = [((i % 45) + 1) for i in range(2_000)]
    unique_arr = [11, 7, 11, 3, 3, 5, 5, 9, 9, 13, 13, 17, 17, 19, 19]
    bin_samples = [decimal_to_binary(i * 12_345) for i in range(1, 200)]

    graph_n = 6_000
    graph_adj: list[list[int]] = [[] for _ in range(graph_n)]
    for i in range(graph_n):
        if i + 1 < graph_n:
            graph_adj[i].append(i + 1)
        if i + 2 < graph_n:
            graph_adj[i].append(i + 2)
        if i % 3 == 0 and i + 17 < graph_n:
            graph_adj[i].append(i + 17)

    weighted_graph_n = 2_200
    weighted_graph_adj: list[list[tuple[int, int]]] = [[] for _ in range(weighted_graph_n)]
    for i in range(weighted_graph_n):
        if i + 1 < weighted_graph_n:
            weighted_graph_adj[i].append((i + 1, ((i * 17) + 3) % 23 + 1))
        if i + 2 < weighted_graph_n:
            weighted_graph_adj[i].append((i + 2, ((i * 31) + 7) % 29 + 1))
        if i % 3 == 0 and i + 17 < weighted_graph_n:
            weighted_graph_adj[i].append((i + 17, ((i * 13) + 11) % 41 + 1))

    bellman_n = 1_800
    bellman_edges: list[tuple[int, int, int]] = []
    for i in range(bellman_n):
        if i + 1 < bellman_n:
            bellman_edges.append((i, i + 1, ((i * 17) + 3) % 23 + 1))
        if i + 2 < bellman_n:
            bellman_edges.append((i, i + 2, ((i * 31) + 7) % 29 + 1))
        if i % 3 == 0 and i + 17 < bellman_n:
            bellman_edges.append((i, i + 17, ((i * 13) + 11) % 41 + 1))

    floyd_n = 120
    floyd_inf = 1_000_000_000_000
    floyd_mat = [floyd_inf] * (floyd_n * floyd_n)
    for i in range(floyd_n):
        floyd_mat[i * floyd_n + i] = 0
        if i + 1 < floyd_n:
            floyd_mat[i * floyd_n + (i + 1)] = ((i * 17) + 3) % 23 + 1
        if i + 2 < floyd_n:
            floyd_mat[i * floyd_n + (i + 2)] = ((i * 31) + 7) % 29 + 1
        if i % 3 == 0 and i + 17 < floyd_n:
            floyd_mat[i * floyd_n + (i + 17)] = ((i * 13) + 11) % 41 + 1

    cycle_graph_n = 600
    cycle_graph_adj: list[list[int]] = [[] for _ in range(cycle_graph_n)]
    for i in range(cycle_graph_n):
        if i + 1 < cycle_graph_n:
            cycle_graph_adj[i].append(i + 1)
        if i + 2 < cycle_graph_n:
            cycle_graph_adj[i].append(i + 2)
        if i % 3 == 0 and i + 17 < cycle_graph_n:
            cycle_graph_adj[i].append(i + 17)
    cycle_graph_adj[-1].append(0)

    component_n = 6_000
    component_adj: list[list[int]] = [[] for _ in range(component_n)]
    split = (component_n // 2) - 1
    for i in range(component_n):
        if i + 1 < component_n and i != split:
            component_adj[i].append(i + 1)
            component_adj[i + 1].append(i)

    mst_n = 1_800
    mst_edges: list[tuple[int, int, int]] = []
    mst_adj: list[list[tuple[int, int]]] = [[] for _ in range(mst_n)]
    for i in range(mst_n):
        if i + 1 < mst_n:
            w = ((i * 19) + 5) % 37 + 1
            mst_edges.append((i, i + 1, w))
            mst_adj[i].append((i + 1, w))
            mst_adj[i + 1].append((i, w))
        if i + 2 < mst_n:
            w = ((i * 23) + 7) % 43 + 1
            mst_edges.append((i, i + 2, w))
            mst_adj[i].append((i + 2, w))
            mst_adj[i + 2].append((i, w))
        if i % 5 == 0 and i + 11 < mst_n:
            w = ((i * 29) + 13) % 53 + 1
            mst_edges.append((i, i + 11, w))
            mst_adj[i].append((i + 11, w))
            mst_adj[i + 11].append((i, w))

    knapsack_weights = [(((i * 73) + 19) % 40) + 1 for i in range(180)]
    knapsack_values = [(((i * 97) + 53) % 500) + 1 for i in range(180)]
    rod_prices = [(((i * 37) + 11) % 600) + 1 for i in range(220)]
    rod_length = 200
    mcm_dims = [(((i * 13) + 7) % 50) + 5 for i in range(71)]
    pal_part_text = "abacdcaba" * 80
    word_break_text = "zigisfastandsafe" * 3000
    word_break_dict = ["zig", "is", "fast", "and", "safe"]
    catalan_inputs = list(range(1, 31))
    coin_set = [1, 2, 3, 5, 7, 11, 13]
    coins_desc = [2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
    frac_values = [60.0, 100.0, 120.0, 140.0, 30.0, 20.0, 80.0, 75.0]
    frac_weights = [10.0, 20.0, 30.0, 40.0, 10.0, 5.0, 15.0, 25.0]

    missing_nums = list(range(50_001))
    missing_target = 30_123
    missing_nums.remove(missing_target)

    bin_text = "10101111100100101111000011101010"

    cases: list[BenchCase] = []

    # Sorts (12)
    cases.append(
        BenchCase(
            "bubble_sort",
            "sorts",
            2,
            lambda: (
                lambda data: (bubble_sort(data), checksum_ints(data))[1]
            )(bubble_base.copy()),
        )
    )
    cases.append(
        BenchCase(
            "insertion_sort",
            "sorts",
            2,
            lambda: (
                lambda data: (insertion_sort(data), checksum_ints(data))[1]
            )(n2_base.copy()),
        )
    )
    cases.append(
        BenchCase("merge_sort", "sorts", 4, lambda: checksum_ints(merge_sort(nlog_base.copy())))
    )
    cases.append(
        BenchCase(
            "quick_sort",
            "sorts",
            4,
            lambda: (
                lambda data: (quick_sort(data), checksum_ints(data))[1]
            )(nlog_base.copy()),
        )
    )
    cases.append(
        BenchCase(
            "heap_sort",
            "sorts",
            4,
            lambda: (
                lambda data: (heap_sort(data), checksum_ints(data))[1]
            )(nlog_base.copy()),
        )
    )
    cases.append(BenchCase("radix_sort", "sorts", 4, lambda: checksum_ints(radix_sort(nlog_base.copy()))))
    cases.append(BenchCase("bucket_sort", "sorts", 4, lambda: checksum_ints(bucket_sort(nlog_base.copy()))))
    cases.append(
        BenchCase(
            "selection_sort",
            "sorts",
            2,
            lambda: (
                lambda data: (selection_sort(data), checksum_ints(data))[1]
            )(n2_base.copy()),
        )
    )
    cases.append(
        BenchCase(
            "shell_sort",
            "sorts",
            4,
            lambda: (
                lambda data: (shell_sort(data), checksum_ints(data))[1]
            )(nlog_base.copy()),
        )
    )
    cases.append(BenchCase("counting_sort", "sorts", 4, lambda: checksum_ints(counting_sort(non_neg_base.copy()))))
    cases.append(
        BenchCase(
            "cocktail_shaker_sort",
            "sorts",
            2,
            lambda: (
                lambda data: (cocktail_shaker_sort(data), checksum_ints(data))[1]
            )(bubble_base.copy()),
        )
    )
    cases.append(
        BenchCase(
            "gnome_sort",
            "sorts",
            2,
            lambda: (
                lambda data: (gnome_sort(data), checksum_ints(data))[1]
            )(bubble_base.copy()),
        )
    )

    # Searches (6)
    cases.append(
        BenchCase(
            "linear_search",
            "searches",
            3,
            lambda: checksum_u(sum((linear_search(search_data, q) + 1) for q in search_queries)),
        )
    )
    cases.append(
        BenchCase(
            "binary_search",
            "searches",
            4,
            lambda: checksum_u(sum((binary_search(search_data, q) + 1) for q in search_queries)),
        )
    )
    cases.append(
        BenchCase(
            "exponential_search",
            "searches",
            4,
            lambda: checksum_u(sum((exponential_search(search_data, q) + 1) for q in search_queries)),
        )
    )
    cases.append(
        BenchCase(
            "interpolation_search",
            "searches",
            4,
            lambda: checksum_u(sum((interpolation_search(search_data, q) + 1) for q in search_queries)),
        )
    )
    cases.append(
        BenchCase(
            "jump_search",
            "searches",
            4,
            lambda: checksum_u(sum((jump_search(search_data, q) + 1) for q in search_queries)),
        )
    )
    cases.append(
        BenchCase(
            "ternary_search",
            "searches",
            4,
            lambda: checksum_u(sum((ternary_search(search_data, q) + 1) for q in search_queries)),
        )
    )

    # Maths (14)
    cases.append(
        BenchCase(
            "gcd",
            "maths",
            6,
            lambda: checksum_u(sum(gcd(math_values[i], math_values[i + 1]) for i in range(0, 20_000, 2))),
        )
    )
    cases.append(
        BenchCase(
            "lcm",
            "maths",
            6,
            lambda: checksum_u(sum(lcm(math_values[i] + 1, math_values[i + 1] + 1) for i in range(0, 20_000, 2))),
        )
    )
    cases.append(BenchCase("fibonacci", "maths", 200, lambda: checksum_u(sum(fibonacci(i) for i in fib_inputs))))
    cases.append(BenchCase("factorial", "maths", 200, lambda: checksum_u(sum(factorial(i) for i in fact_inputs))))
    cases.append(
        BenchCase(
            "power",
            "maths",
            300,
            lambda: checksum_u(sum(power(b, e) for (b, e) in power_pairs)),
        )
    )
    cases.append(BenchCase("prime_check", "maths", 20, lambda: checksum_u(sum(1 for i in range(2, 120_000) if is_prime(i)))))
    cases.append(BenchCase("sieve_of_eratosthenes", "maths", 6, lambda: checksum_ints(sieve(300_000))))
    cases.append(BenchCase("collatz_sequence", "maths", 8, lambda: checksum_u(sum(collatz_steps(i) for i in range(2, 60_000)))))
    cases.append(BenchCase("extended_euclidean", "maths", 20, lambda: checksum_u(extended_euclidean_checksum(ext_pairs))))
    cases.append(
        BenchCase(
            "modular_inverse",
            "maths",
            20,
            lambda: checksum_u(sum(modular_inverse(a, m) for (a, m) in modinv_pairs)),
        )
    )
    cases.append(
        BenchCase(
            "eulers_totient",
            "maths",
            12,
            lambda: checksum_u(sum(eulers_totient(v) for v in totient_inputs)),
        )
    )
    cases.append(BenchCase("chinese_remainder_theorem", "maths", 25, lambda: checksum_u(crt_many_checksum(crt_systems))))
    cases.append(
        BenchCase(
            "binomial_coefficient",
            "maths",
            20,
            lambda: checksum_u(sum(binomial_coefficient(n, k if k <= n else n) for (n, k) in binom_pairs)),
        )
    )
    cases.append(
        BenchCase(
            "integer_square_root",
            "maths",
            25,
            lambda: checksum_u(sum(integer_square_root(v) for v in isqrt_inputs)),
        )
    )

    # Data Structures (5)
    cases.append(BenchCase("trie", "data_structures", 8, lambda: checksum_u(trie_workload(trie_words))))
    cases.append(BenchCase("disjoint_set", "data_structures", 10, lambda: checksum_u(disjoint_set_workload(disjoint_set_n))))
    cases.append(BenchCase("avl_tree", "data_structures", 4, lambda: checksum_u(avl_tree_workload(avl_values, avl_queries))))
    cases.append(BenchCase("max_heap", "data_structures", 8, lambda: checksum_u(max_heap_workload(max_heap_values))))
    cases.append(BenchCase("priority_queue", "data_structures", 8, lambda: checksum_u(priority_queue_workload(priority_queue_n))))

    # Dynamic Programming (13)
    cases.append(
        BenchCase(
            "climbing_stairs",
            "dynamic_programming",
            1000,
            lambda: checksum_u(sum(climbing_stairs(n) for n in stairs_inputs)),
        )
    )
    cases.append(BenchCase("fibonacci_dp", "dynamic_programming", 400, lambda: checksum_ints(fibonacci_dp(280))))
    cases.append(BenchCase("coin_change", "dynamic_programming", 80, lambda: checksum_u(coin_change_ways(coin_set, 420))))
    cases.append(BenchCase("max_subarray_sum", "dynamic_programming", 20, lambda: checksum_u(max_subarray_sum(dp_array, False))))
    cases.append(BenchCase("longest_increasing_subsequence", "dynamic_programming", 12, lambda: checksum_u(longest_increasing_subsequence(dp_array))))
    cases.append(BenchCase("rod_cutting", "dynamic_programming", 80, lambda: checksum_u(rod_cutting(rod_prices, rod_length))))
    cases.append(BenchCase("matrix_chain_multiplication", "dynamic_programming", 30, lambda: checksum_u(matrix_chain_multiplication(mcm_dims))))
    cases.append(BenchCase("palindrome_partitioning", "dynamic_programming", 60, lambda: checksum_u(palindrome_partition_min_cuts(pal_part_text))))
    cases.append(BenchCase("word_break", "dynamic_programming", 60, lambda: checksum_bool(word_break(word_break_text, word_break_dict))))
    cases.append(BenchCase("catalan_numbers", "dynamic_programming", 120, lambda: checksum_u(sum(catalan_number(i) for i in catalan_inputs))))
    cases.append(BenchCase("longest_common_subsequence", "dynamic_programming", 30, lambda: checksum_u(lcs_length(s1, s2))))
    cases.append(BenchCase("edit_distance", "dynamic_programming", 30, lambda: checksum_u(edit_distance(s1, s2))))
    cases.append(
        BenchCase(
            "knapsack",
            "dynamic_programming",
            60,
            lambda: checksum_u(knapsack(800, knapsack_weights, knapsack_values)),
        )
    )

    # Graphs (10)
    cases.append(BenchCase("bfs", "graphs", 12, lambda: checksum_ints(bfs(graph_adj, 0))))
    cases.append(BenchCase("dfs", "graphs", 12, lambda: checksum_ints(dfs(graph_adj, 0))))
    cases.append(BenchCase("dijkstra", "graphs", 8, lambda: checksum_ints(dijkstra(weighted_graph_adj, 0))))
    cases.append(BenchCase("bellman_ford", "graphs", 4, lambda: checksum_ints(bellman_ford(bellman_n, bellman_edges, 0))))
    cases.append(BenchCase("topological_sort", "graphs", 12, lambda: checksum_ints(topological_sort(graph_adj))))
    cases.append(BenchCase("floyd_warshall", "graphs", 2, lambda: checksum_ints(floyd_warshall(floyd_mat, floyd_n, floyd_inf))))
    cases.append(BenchCase("detect_cycle", "graphs", 20, lambda: checksum_bool(detect_cycle(cycle_graph_adj))))
    cases.append(BenchCase("connected_components", "graphs", 8, lambda: checksum_u(connected_components(component_adj))))
    cases.append(BenchCase("kruskal", "graphs", 6, lambda: checksum_u(kruskal_mst_weight(mst_n, mst_edges))))
    cases.append(BenchCase("prim", "graphs", 6, lambda: checksum_u(prim_mst_weight(mst_adj, 0))))

    # Bit manipulation (6)
    cases.append(BenchCase("is_power_of_two", "bit_manipulation", 120, lambda: checksum_u(sum(1 for i in range(1, 3_000_000) if is_power_of_two(i)))))
    cases.append(BenchCase("count_set_bits", "bit_manipulation", 60, lambda: checksum_u(sum(count_set_bits(i) for i in range(1, 2_000_000)))))
    cases.append(BenchCase("find_unique_number", "bit_manipulation", 1000, lambda: checksum_u(find_unique_number(unique_arr))))
    cases.append(BenchCase("reverse_bits", "bit_manipulation", 80, lambda: checksum_u(sum(reverse_bits(i) for i in range(0, 200_000)))))
    cases.append(BenchCase("missing_number", "bit_manipulation", 220, lambda: checksum_u(missing_number(missing_nums))))
    cases.append(BenchCase("power_of_4", "bit_manipulation", 120, lambda: checksum_u(sum(1 for i in range(1, 3_000_000) if is_power_of_four(i)))))

    # Conversions (4)
    cases.append(BenchCase("decimal_to_binary", "conversions", 120, lambda: checksum_bytes(decimal_to_binary(987_654_321))))
    cases.append(
        BenchCase(
            "binary_to_decimal",
            "conversions",
            120,
            lambda: checksum_u(sum(binary_to_decimal(s) for s in bin_samples)),
        )
    )
    cases.append(BenchCase("decimal_to_hexadecimal", "conversions", 120, lambda: checksum_bytes(decimal_to_hex(987_654_321))))
    cases.append(BenchCase("binary_to_hexadecimal", "conversions", 120, lambda: checksum_bytes(binary_to_hex(bin_text))))

    # Greedy (4)
    cases.append(BenchCase("best_time_to_buy_sell_stock", "greedy_methods", 20, lambda: checksum_u(max_profit(prices))))
    cases.append(BenchCase("minimum_coin_change", "greedy_methods", 200, lambda: checksum_ints(minimum_coin_change(coins_desc, 987))))
    cases.append(BenchCase("minimum_waiting_time", "greedy_methods", 15, lambda: checksum_u(minimum_waiting_time(waiting_queries))))
    cases.append(BenchCase("fractional_knapsack", "greedy_methods", 600, lambda: checksum_u(int(fractional_knapsack(frac_values, frac_weights, 50.0) * 1_000_000))))

    # Matrix (5)
    cases.append(BenchCase("matrix_multiply", "matrix", 10, lambda: checksum_ints(matrix_multiply(matrix_a, matrix_b, matrix_dim))))
    cases.append(BenchCase("matrix_transpose", "matrix", 25, lambda: checksum_ints(transpose(transpose_mat, transpose_rows, transpose_cols))))
    cases.append(
        BenchCase(
            "rotate_matrix",
            "matrix",
            25,
            lambda: (
                lambda m: (rotate90(m, rotate_n), checksum_ints(m))[1]
            )(rotate_mat.copy()),
        )
    )
    cases.append(BenchCase("spiral_print", "matrix", 20, lambda: checksum_ints(spiral_order(spiral_mat, spiral_rows, spiral_cols))))
    cases.append(
        BenchCase(
            "pascal_triangle",
            "matrix",
            120,
            lambda: checksum_u(sum(pascal_triangle(180)[-1]) & MASK_64),
        )
    )

    # Strings (10)
    cases.append(BenchCase("palindrome", "strings", 120, lambda: checksum_bool(is_palindrome("amanaplanacanalpanama" * 5000))))
    cases.append(BenchCase("reverse_words", "strings", 80, lambda: checksum_bytes(reverse_words("I     Love          Python " * 12_000))))
    cases.append(BenchCase("anagram", "strings", 600, lambda: checksum_bool(is_anagram(anagram_a, anagram_b))))
    cases.append(BenchCase("hamming_distance", "strings", 120, lambda: checksum_u(hamming_distance(s3, s4))))
    cases.append(BenchCase("naive_string_search", "strings", 30, lambda: checksum_ints(naive_search(text, pattern))))
    cases.append(BenchCase("knuth_morris_pratt", "strings", 80, lambda: checksum_u(kmp_search(text, pattern))))
    cases.append(BenchCase("rabin_karp", "strings", 60, lambda: checksum_bool(rabin_karp(text, pattern))))
    cases.append(BenchCase("z_function", "strings", 80, lambda: checksum_ints(z_function(text))))
    cases.append(BenchCase("levenshtein_distance", "strings", 120, lambda: checksum_u(levenshtein_distance(s1, s2))))
    cases.append(BenchCase("is_pangram", "strings", 120, lambda: checksum_bool(is_pangram(pangram_text))))

    selected_algorithm = os.getenv("BENCH_ALGORITHM", "").strip()
    if selected_algorithm:
        cases = [case for case in cases if case.name == selected_algorithm]
        if not cases:
            raise SystemExit(f"unknown benchmark algorithm: {selected_algorithm}")

    print("algorithm,category,iterations,total_ns,avg_ns,checksum")
    for case in cases:
        name, category, iterations, total_ns, avg_ns, checksum = bench_case(case)
        print(f"{name},{category},{iterations},{total_ns},{avg_ns},{checksum}")


if __name__ == "__main__":
    main()
