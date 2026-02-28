#!/usr/bin/env python3
"""Python vs Zig benchmark harness (Python side, 64 alignable algorithms)."""

from __future__ import annotations

import heapq
import math
import time
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


def main() -> None:
    bubble_base = generate_int_data(1_200)
    n2_base = generate_int_data(1_300)
    nlog_base = generate_int_data(28_000)
    non_neg_base = generate_non_negative_data(30_000)
    search_data = generate_sorted_data(40_000)
    search_queries = generate_search_queries(1_200, len(search_data))
    math_values = generate_u64_data(40_000)
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

    knapsack_weights = [(((i * 73) + 19) % 40) + 1 for i in range(180)]
    knapsack_values = [(((i * 97) + 53) % 500) + 1 for i in range(180)]
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

    # Maths (8)
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

    # Dynamic Programming (7)
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

    # Graphs (2)
    cases.append(BenchCase("bfs", "graphs", 12, lambda: checksum_ints(bfs(graph_adj, 0))))
    cases.append(BenchCase("dfs", "graphs", 12, lambda: checksum_ints(dfs(graph_adj, 0))))

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

    print("algorithm,category,iterations,total_ns,avg_ns,checksum")
    for case in cases:
        name, category, iterations, total_ns, avg_ns, checksum = bench_case(case)
        print(f"{name},{category},{iterations},{total_ns},{avg_ns},{checksum}")


if __name__ == "__main__":
    main()
