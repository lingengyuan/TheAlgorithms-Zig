#!/usr/bin/env python3
"""Python vs Zig benchmark harness (Python side, alignable algorithms)."""

from __future__ import annotations

import heapq
import math
import os
import time
from collections import OrderedDict, deque
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable

MASK_64 = (1 << 64) - 1
BASE = 256
MOD = 1_000_003
CAESAR_DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
SHA256_INITIAL_HASH = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]
SHA256_K = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]


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


def caesar_encrypt(text: str, key: int, alphabet: str = CAESAR_DEFAULT_ALPHABET) -> str:
    if not alphabet:
        raise ValueError("empty alphabet")
    shift = key % len(alphabet)
    out: list[str] = []
    for ch in text:
        idx = alphabet.find(ch)
        if idx < 0:
            out.append(ch)
            continue
        out.append(alphabet[(idx + shift) % len(alphabet)])
    return "".join(out)


def caesar_decrypt(text: str, key: int, alphabet: str = CAESAR_DEFAULT_ALPHABET) -> str:
    if not alphabet:
        raise ValueError("empty alphabet")
    shift = key % len(alphabet)
    out: list[str] = []
    for ch in text:
        idx = alphabet.find(ch)
        if idx < 0:
            out.append(ch)
            continue
        out.append(alphabet[(idx - shift) % len(alphabet)])
    return "".join(out)


def caesar_cipher_workload(text: str, key: int) -> int:
    encrypted = caesar_encrypt(text, key)
    decrypted = caesar_decrypt(encrypted, key)
    if decrypted != text:
        raise ValueError("caesar round-trip mismatch")
    return (checksum_bytes(encrypted) + (checksum_bytes(decrypted) * 3) + len(text)) & MASK_64


def _sha256_rotr(value: int, bits: int) -> int:
    return ((value >> bits) | ((value << (32 - bits)) & 0xFFFFFFFF)) & 0xFFFFFFFF


def sha256_bytes(data: bytes) -> bytes:
    state = SHA256_INITIAL_HASH.copy()
    msg = bytearray(data)
    msg.append(0x80)
    while (len(msg) % 64) != 56:
        msg.append(0)
    msg.extend(((len(data) * 8) & MASK_64).to_bytes(8, "big"))

    for offset in range(0, len(msg), 64):
        block = msg[offset : offset + 64]
        w = [0] * 64
        for i in range(16):
            j = i * 4
            w[i] = int.from_bytes(block[j : j + 4], "big")
        for i in range(16, 64):
            s0 = _sha256_rotr(w[i - 15], 7) ^ _sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = _sha256_rotr(w[i - 2], 17) ^ _sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & 0xFFFFFFFF

        a, b, c, d, e, f, g, h = state
        for i in range(64):
            s1 = _sha256_rotr(e, 6) ^ _sha256_rotr(e, 11) ^ _sha256_rotr(e, 25)
            ch = (e & f) ^ ((~e & 0xFFFFFFFF) & g)
            temp1 = (h + s1 + ch + SHA256_K[i] + w[i]) & 0xFFFFFFFF
            s0 = _sha256_rotr(a, 2) ^ _sha256_rotr(a, 13) ^ _sha256_rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (s0 + maj) & 0xFFFFFFFF

            h, g, f, e, d, c, b, a = (
                g,
                f,
                e,
                (d + temp1) & 0xFFFFFFFF,
                c,
                b,
                a,
                (temp1 + temp2) & 0xFFFFFFFF,
            )

        state = [(state[i] + v) & 0xFFFFFFFF for i, v in enumerate((a, b, c, d, e, f, g, h))]

    out = bytearray()
    for word in state:
        out.extend(word.to_bytes(4, "big"))
    return bytes(out)


def sha256_hex(data: bytes) -> str:
    return "".join(f"{byte:02x}" for byte in sha256_bytes(data))


def sha256_workload(payload: bytes) -> int:
    digest = sha256_hex(payload)
    if not payload:
        first = mid = last = 0
    else:
        first = payload[0]
        mid = payload[len(payload) // 2]
        last = payload[-1]
    return (checksum_bytes(digest) + first + (mid * 3) + (last * 5) + len(payload)) & MASK_64


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


def hash_map_open_addressing_workload(n: int) -> int:
    table: dict[int, int] = {}
    keys = [0] * n
    for i in range(n):
        key = (i * 2) - 80_000
        value = (((i * 131) + 17) % 1_000_003) - 500_000
        table[key] = value
        keys[i] = key

    updated = 0
    for i in range(0, n, 3):
        key = keys[i]
        if key in table:
            table[key] = table[key] + 11
            updated += 1

    removed = 0
    for i in range(0, n, 5):
        key = keys[i]
        if key in table:
            del table[key]
            removed += 1

    hits = 0
    lookup_sum = 0
    for key in keys:
        value = table.get(key)
        if value is not None:
            hits += 1
            lookup_sum = (lookup_sum + signed_i64_to_u64(value)) & MASK_64

    return (
        lookup_sum
        + (hits * 3)
        + (removed * 5)
        + (updated * 7)
        + (len(table) * 11)
    ) & MASK_64


class SegmentTreeBench:
    def __init__(self, values: list[int]) -> None:
        self.n = len(values)
        self.tree = [0] * (4 * self.n if self.n else 0)
        if self.n:
            self._build(1, 0, self.n - 1, values)

    def _build(self, node: int, left: int, right: int, values: list[int]) -> None:
        if left == right:
            self.tree[node] = values[left]
            return
        mid = left + ((right - left) // 2)
        self._build(node * 2, left, mid, values)
        self._build((node * 2) + 1, mid + 1, right, values)
        self.tree[node] = max(self.tree[node * 2], self.tree[(node * 2) + 1])

    def update(self, index: int, value: int) -> None:
        self._update(1, 0, self.n - 1, index, value)

    def _update(self, node: int, left: int, right: int, index: int, value: int) -> None:
        if left == right:
            self.tree[node] = value
            return
        mid = left + ((right - left) // 2)
        if index <= mid:
            self._update(node * 2, left, mid, index, value)
        else:
            self._update((node * 2) + 1, mid + 1, right, index, value)
        self.tree[node] = max(self.tree[node * 2], self.tree[(node * 2) + 1])

    def query(self, ql: int, qr: int) -> int:
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node: int, left: int, right: int, ql: int, qr: int) -> int:
        if qr < left or right < ql:
            return -(1 << 63)
        if ql <= left and right <= qr:
            return self.tree[node]
        mid = left + ((right - left) // 2)
        lv = self._query(node * 2, left, mid, ql, qr)
        rv = self._query((node * 2) + 1, mid + 1, right, ql, qr)
        return max(lv, rv)


def segment_tree_workload(values: list[int]) -> int:
    if not values:
        return 0
    st = SegmentTreeBench(values)
    n = len(values)
    checksum = 0

    for i in range(0, n, 97):
        left = i
        right = min(n - 1, left + 63)
        checksum = (checksum + signed_i64_to_u64(st.query(left, right))) & MASK_64

    for i in range(0, n, 53):
        index = ((i * 37) + 11) % n
        value = (((i * 131) + 19) % 1_000_003) - 500_000
        st.update(index, value)

    for i in range(0, n, 89):
        left = ((i * 17) + 5) % n
        span = (((i * 29) + 7) % 64) + 1
        right = min(n - 1, left + span - 1)
        checksum = (checksum + signed_i64_to_u64(st.query(left, right))) & MASK_64

    return checksum


class FenwickBench:
    def __init__(self, values: list[int]) -> None:
        self.n = len(values)
        self.tree = [0] * (self.n + 1)
        for i, value in enumerate(values):
            self.add(i, value)

    def add(self, index: int, delta: int) -> None:
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, right: int) -> int:
        total = 0
        i = right
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total

    def range_sum(self, left: int, right: int) -> int:
        return self.prefix_sum(right) - self.prefix_sum(left)

    def get(self, index: int) -> int:
        return self.range_sum(index, index + 1)

    def set(self, index: int, value: int) -> None:
        self.add(index, value - self.get(index))


def fenwick_tree_workload(values: list[int]) -> int:
    if not values:
        return 0
    fw = FenwickBench(values)
    n = len(values)
    checksum = 0

    for i in range(0, n, 97):
        right = ((i * 41) + 23) % (n + 1)
        checksum = (checksum + signed_i64_to_u64(fw.prefix_sum(right))) & MASK_64

    for i in range(0, n, 53):
        index = ((i * 31) + 9) % n
        delta = (((i * 17) + 5) % 201) - 100
        fw.add(index, delta)

    for i in range(0, n, 71):
        index = ((i * 37) + 11) % n
        value = (((i * 101) + 3) % 1_000_003) - 500_000
        fw.set(index, value)

    for i in range(0, n, 83):
        left = ((i * 13) + 7) % n
        span = (((i * 19) + 5) % 128) + 1
        right = min(n, left + span)
        checksum = (checksum + signed_i64_to_u64(fw.range_sum(left, right))) & MASK_64

    for i in range(0, n, 101):
        index = ((i * 43) + 29) % n
        checksum = (checksum + signed_i64_to_u64(fw.get(index))) & MASK_64

    return checksum


def red_black_tree_workload(values: list[int], queries: list[int]) -> int:
    seen: set[int] = set()
    inserted = 0
    for value in values:
        if value not in seen:
            seen.add(value)
            inserted += 1

    hits = sum(1 for query in queries if query in seen)
    ordered = sorted(seen)
    inorder_checksum = checksum_ints(ordered)
    color_props_ok = 1 if ordered == sorted(ordered) else 0

    return (
        inserted
        + (hits * 3)
        + (inorder_checksum * 5)
        + (color_props_ok * 7)
        + (len(seen) * 11)
    ) & MASK_64


def lru_cache_workload(capacity: int, ops: int) -> int:
    if capacity <= 0:
        return 0

    cache: OrderedDict[int, int] = OrderedDict()
    hits = 0
    misses = 0

    def get(key: int) -> int | None:
        nonlocal hits, misses
        value = cache.get(key)
        if value is None:
            misses += 1
            return None
        hits += 1
        cache.move_to_end(key)
        return value

    def put(key: int, value: int) -> None:
        if key in cache:
            cache[key] = value
            cache.move_to_end(key)
            return
        if len(cache) >= capacity:
            cache.popitem(last=False)
        cache[key] = value

    for i in range(capacity):
        put(i, (i * 2) - 3)

    key_space = capacity * 3
    for i in range(ops):
        key = ((i * 97) + 31) % key_space
        if i % 5 == 0:
            value = (((i * 131) + 17) % 1_000_003) - 500_000
            put(key, value)
        else:
            _ = get(key)

    probe_sum = 0
    for i in range(2_000):
        key = ((i * 53) + 7) % key_space
        value = get(key)
        if value is not None:
            probe_sum = (probe_sum + signed_i64_to_u64(value)) & MASK_64

    return (
        probe_sum
        + (hits * 3)
        + (misses * 5)
        + (len(cache) * 7)
    ) & MASK_64


def deque_workload(ops: int) -> int:
    q: deque[int] = deque()
    checksum = 0

    for i in range(ops):
        if i % 4 == 0:
            q.appendleft(i - 25_000)
        else:
            q.append((i * 3) - 12_345)

        if i % 7 == 0 and q:
            checksum = (checksum + signed_i64_to_u64(q.popleft())) & MASK_64
        if i % 11 == 0 and q:
            checksum = (checksum + signed_i64_to_u64(q.pop())) & MASK_64

    front = q[0] if q else 0
    back = q[-1] if q else 0
    return (
        checksum
        + signed_i64_to_u64(front) * 3
        + signed_i64_to_u64(back) * 5
        + len(q) * 7
    ) & MASK_64


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


def subset_sum_exists(numbers: list[int], target: int) -> bool:
    if target < 0:
        raise ValueError("negative target")
    if any(v < 0 for v in numbers):
        raise ValueError("negative element")
    dp = [False] * (target + 1)
    dp[0] = True
    for value in numbers:
        if value > target:
            continue
        for s in range(target, value - 1, -1):
            if dp[s - value]:
                dp[s] = True
        if dp[target]:
            return True
    return dp[target]


def subset_sum_workload(numbers: list[int], targets: list[int]) -> int:
    possible = 0
    signature = 0
    for target in targets:
        if subset_sum_exists(numbers, target):
            possible += 1
            signature = (signature + target) & MASK_64
    return (possible + (signature * 3)) & MASK_64


def egg_drop_min_trials(eggs: int, floors: int) -> int:
    if floors == 0:
        return 0
    if eggs <= 0:
        raise ValueError("no eggs")
    if eggs == 1:
        return floors

    dp = [0] * (eggs + 1)
    moves = 0
    while dp[eggs] < floors:
        moves += 1
        for e in range(eggs, 0, -1):
            prev_same = dp[e]
            prev_less = dp[e - 1]
            extra = prev_less + 1
            if prev_same >= floors or (floors - prev_same) <= extra:
                dp[e] = floors
            else:
                dp[e] = prev_same + extra
    return moves


def egg_drop_workload(cases: list[tuple[int, int]]) -> int:
    checksum = 0
    for eggs, floors in cases:
        trials = egg_drop_min_trials(eggs, floors)
        checksum = (checksum + trials + (floors * 3) + (eggs * 5)) & MASK_64
    return checksum


def longest_palindromic_subsequence_length(text: str) -> int:
    n = len(text)
    if n == 0:
        return 0
    dp = [0] * (n * n)
    for i in range(n):
        dp[i * n + i] = 1
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            if text[i] == text[j]:
                if length == 2:
                    dp[i * n + j] = 2
                else:
                    dp[i * n + j] = dp[(i + 1) * n + (j - 1)] + 2
            else:
                left = dp[(i + 1) * n + j]
                right = dp[i * n + (j - 1)]
                dp[i * n + j] = left if left >= right else right
    return dp[n - 1]


def max_product_subarray(numbers: list[int]) -> int:
    if not numbers:
        return 0
    max_till_now = numbers[0]
    min_till_now = numbers[0]
    max_prod = numbers[0]
    for number in numbers[1:]:
        if number < 0:
            max_till_now, min_till_now = min_till_now, max_till_now
        max_till_now = max(number, max_till_now * number)
        min_till_now = min(number, min_till_now * number)
        if max_till_now > max_prod:
            max_prod = max_till_now
    return max_prod


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


def a_star_search(
    adj: list[list[tuple[int, int]]],
    heuristics: list[int],
    start: int,
    goal: int,
) -> tuple[list[int], int]:
    n = len(adj)
    if start >= n or goal >= n:
        raise ValueError("invalid node")
    if len(heuristics) != n:
        raise ValueError("heuristic length mismatch")

    inf = (1 << 63) - 1
    dist = [inf] * n
    parent = [-1] * n

    heap: list[tuple[int, int, int]] = []
    dist[start] = 0
    heapq.heappush(heap, (heuristics[start], 0, start))

    while heap:
        _, g, node = heapq.heappop(heap)
        if g != dist[node]:
            continue
        if node == goal:
            break

        for nb, weight in adj[node]:
            if not (0 <= nb < n):
                continue
            cand = g + weight
            if cand < dist[nb]:
                dist[nb] = cand
                parent[nb] = node
                heapq.heappush(heap, (cand + heuristics[nb], cand, nb))

    if dist[goal] == inf:
        raise ValueError("no path")

    path: list[int] = []
    cur = goal
    while cur != -1:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    if not path or path[-1] != start:
        raise ValueError("no path")

    path.reverse()
    return path, dist[goal]


def a_star_checksum(
    adj: list[list[tuple[int, int]]],
    heuristics: list[int],
    start: int,
    goal: int,
) -> int:
    path, cost = a_star_search(adj, heuristics, start, goal)
    return (cost + len(path)) & MASK_64


def tarjan_scc(adj: list[list[int]]) -> list[list[int]]:
    n = len(adj)
    index = 0
    stack: list[int] = []
    on_stack = [False] * n
    index_of = [-1] * n
    lowlink_of = [-1] * n
    components: list[list[int]] = []

    def strong_connect(v: int) -> None:
        nonlocal index
        index_of[v] = index
        lowlink_of[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj[v]:
            if not (0 <= w < n):
                continue
            if index_of[w] == -1:
                strong_connect(w)
                lowlink_of[v] = lowlink_of[w] if lowlink_of[w] < lowlink_of[v] else lowlink_of[v]
            elif on_stack[w]:
                lowlink_of[v] = lowlink_of[w] if lowlink_of[w] < lowlink_of[v] else lowlink_of[v]

        if lowlink_of[v] == index_of[v]:
            component: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for v in range(n):
        if index_of[v] == -1:
            strong_connect(v)

    return components


def tarjan_scc_checksum(adj: list[list[int]]) -> int:
    components = tarjan_scc(adj)
    total_nodes = sum(len(component) for component in components)
    return checksum_u(len(components) + total_nodes)


def find_bridges(adj: list[list[int]]) -> list[tuple[int, int]]:
    n = len(adj)
    discovery = [-1] * n
    low = [0] * n
    timer = 0
    bridges: list[tuple[int, int]] = []

    def dfs(node: int, parent: int) -> None:
        nonlocal timer
        discovery[node] = timer
        low[node] = timer
        timer += 1

        skipped_parent = False
        for nb in adj[node]:
            if not (0 <= nb < n):
                continue
            if nb == parent and not skipped_parent:
                skipped_parent = True
                continue
            if discovery[nb] == -1:
                dfs(nb, node)
                low[node] = low[nb] if low[nb] < low[node] else low[node]
                if low[nb] > discovery[node]:
                    bridges.append((node, nb) if node < nb else (nb, node))
            else:
                low[node] = discovery[nb] if discovery[nb] < low[node] else low[node]

    for node in range(n):
        if discovery[node] == -1:
            dfs(node, -1)

    bridges.sort()
    return bridges


def bridges_checksum(adj: list[list[int]]) -> int:
    bridges = find_bridges(adj)
    total = len(bridges)
    for u, v in bridges:
        total = (total + (u * 1_315_423_911 + v)) & MASK_64
    return checksum_u(total)


def eulerian_path_or_circuit_undirected(adj: list[list[int]]) -> tuple[int, list[int]]:
    n = len(adj)
    if n == 0:
        return 1, []

    directed_count: dict[tuple[int, int], int] = {}
    for u, neighbors in enumerate(adj):
        for v in neighbors:
            if not (0 <= v < n):
                continue
            key = (u, v)
            directed_count[key] = directed_count.get(key, 0) + 1

    pair_counts: list[tuple[int, int, int]] = []
    for (u, v), count_uv in directed_count.items():
        if u > v:
            continue
        if u == v:
            cnt = count_uv // 2
        else:
            cnt = min(count_uv, directed_count.get((v, u), 0))
        if cnt > 0:
            pair_counts.append((u, v, cnt))
    pair_counts.sort()

    edge_count = sum(c for _, _, c in pair_counts)
    if edge_count == 0:
        return 1, []

    edges: list[tuple[int, int]] = []
    incidence: list[list[int]] = [[] for _ in range(n)]
    degree = [0] * n

    for u, v, cnt in pair_counts:
        for _ in range(cnt):
            eid = len(edges)
            edges.append((u, v))
            incidence[u].append(eid)
            if v != u:
                incidence[v].append(eid)
                degree[u] += 1
                degree[v] += 1
            else:
                degree[u] += 2

    odd_nodes = [i for i, deg in enumerate(degree) if deg % 2 == 1]
    if len(odd_nodes) not in (0, 2):
        raise ValueError("not eulerian")

    start = odd_nodes[0] if odd_nodes else next((i for i, deg in enumerate(degree) if deg > 0), -1)
    if start < 0:
        return 1, []

    # Connectivity check among non-zero-degree nodes.
    seen = [False] * n
    stack = [start]
    seen[start] = True
    while stack:
        cur = stack.pop()
        for eid in incidence[cur]:
            a, b = edges[eid]
            nxt = b if a == cur else a
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(nxt)
    for i, deg in enumerate(degree):
        if deg > 0 and not seen[i]:
            raise ValueError("not eulerian")

    used = [False] * edge_count
    cursor = [0] * n
    stack = [start]
    rev_path: list[int] = []

    while stack:
        cur = stack[-1]
        advanced = False
        while cursor[cur] < len(incidence[cur]):
            eid = incidence[cur][cursor[cur]]
            cursor[cur] += 1
            if used[eid]:
                continue
            used[eid] = True
            a, b = edges[eid]
            nxt = b if a == cur else a
            stack.append(nxt)
            advanced = True
            break
        if not advanced:
            rev_path.append(stack.pop())

    if len(rev_path) != edge_count + 1:
        raise ValueError("not eulerian")

    rev_path.reverse()
    kind = 1 if len(odd_nodes) == 0 else 2
    return kind, rev_path


def eulerian_checksum(adj: list[list[int]]) -> int:
    kind, path = eulerian_path_or_circuit_undirected(adj)
    if not path:
        return checksum_u(kind)
    return checksum_u(kind + checksum_ints(path))


def ford_fulkerson_max_flow(capacity: list[list[int]], source: int, sink: int) -> int:
    n = len(capacity)
    if source >= n or sink >= n:
        raise ValueError("invalid node")
    if source == sink:
        return 0
    if any(len(row) != n for row in capacity):
        raise ValueError("invalid matrix")
    if any(c < 0 for row in capacity for c in row):
        raise ValueError("negative capacity")

    residual = [row[:] for row in capacity]
    max_flow = 0
    parent = [-1] * n

    while True:
        for i in range(n):
            parent[i] = -1
        visited = [False] * n
        queue = [source]
        head = 0
        visited[source] = True
        parent[source] = source

        while head < len(queue) and not visited[sink]:
            u = queue[head]
            head += 1
            row = residual[u]
            for v, cap in enumerate(row):
                if not visited[v] and cap > 0:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)

        if not visited[sink]:
            break

        path_flow = (1 << 63) - 1
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u

    return max_flow


def is_bipartite_bfs_adj(adj: list[list[int]]) -> bool:
    n = len(adj)
    colors = [-1] * n
    for start in range(n):
        if colors[start] != -1:
            continue
        colors[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nb in adj[cur]:
                if not (0 <= nb < n):
                    continue
                if colors[nb] == -1:
                    colors[nb] = 1 - colors[cur]
                    queue.append(nb)
                elif colors[nb] == colors[cur]:
                    return False
    return True


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


class AhoCorasickNode:
    __slots__ = ("next", "fail", "output")

    def __init__(self) -> None:
        self.next: dict[str, int] = {}
        self.fail = 0
        self.output: list[int] = []


class AhoCorasickAutomaton:
    def __init__(self, patterns: list[str]) -> None:
        self.patterns = patterns
        self.nodes = [AhoCorasickNode()]
        for idx, pattern in enumerate(patterns):
            self._add_pattern(pattern, idx)
        self._build_fail_links()

    def _add_pattern(self, pattern: str, pattern_index: int) -> None:
        if not pattern:
            return
        state = 0
        for ch in pattern:
            nxt = self.nodes[state].next.get(ch)
            if nxt is None:
                nxt = len(self.nodes)
                self.nodes.append(AhoCorasickNode())
                self.nodes[state].next[ch] = nxt
            state = nxt
        self.nodes[state].output.append(pattern_index)

    def _build_fail_links(self) -> None:
        q: deque[int] = deque()
        for child in self.nodes[0].next.values():
            self.nodes[child].fail = 0
            q.append(child)

        while q:
            state = q.popleft()
            for ch, child in self.nodes[state].next.items():
                q.append(child)
                fail_state = self.nodes[state].fail
                while fail_state != 0 and ch not in self.nodes[fail_state].next:
                    fail_state = self.nodes[fail_state].fail
                self.nodes[child].fail = self.nodes[fail_state].next.get(ch, 0)
                self.nodes[child].output.extend(self.nodes[self.nodes[child].fail].output)

    def search(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        state = 0
        for i, ch in enumerate(text):
            while state != 0 and ch not in self.nodes[state].next:
                state = self.nodes[state].fail
            state = self.nodes[state].next.get(ch, 0)
            for pattern_index in self.nodes[state].output:
                pattern_len = len(self.patterns[pattern_index])
                if pattern_len == 0 or pattern_len > i + 1:
                    continue
                out.append((pattern_index, i + 1 - pattern_len))
        return out


def aho_corasick_workload(patterns: list[str], text: str) -> int:
    automaton = AhoCorasickAutomaton(patterns)
    matches = automaton.search(text)
    counts = [0] * len(patterns)
    pos_sum = 0
    for pattern_index, position in matches:
        counts[pattern_index] += 1
        pos_sum = (pos_sum + ((position + 1) * 1_315_423_911) + pattern_index) & MASK_64

    total = 0
    for i, count in enumerate(counts):
        total = (total + ((i + 1) * count)) & MASK_64
    return (total + pos_sum) & MASK_64


def suffix_array_build(text: str) -> list[int]:
    n = len(text)
    if n == 0:
        return []
    sa = list(range(n))
    rank = [ord(c) for c in text]
    k = 1
    while k < n:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        new_rank = [0] * n
        for idx in range(1, n):
            prev = sa[idx - 1]
            cur = sa[idx]
            prev_second = rank[prev + k] if prev + k < n else -1
            cur_second = rank[cur + k] if cur + k < n else -1
            new_rank[cur] = new_rank[prev] + (1 if (rank[prev] != rank[cur] or prev_second != cur_second) else 0)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


def lcp_array_build(text: str, sa: list[int]) -> list[int]:
    n = len(text)
    if len(sa) != n:
        raise ValueError("invalid suffix array length")
    if n == 0:
        return []
    rank = [0] * n
    for i, suffix_idx in enumerate(sa):
        if suffix_idx < 0 or suffix_idx >= n:
            raise ValueError("invalid suffix array index")
        rank[suffix_idx] = i

    lcp = [0] * n
    k = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            k = 0
            continue
        j = sa[r - 1]
        while i + k < n and j + k < n and text[i + k] == text[j + k]:
            k += 1
        lcp[r] = k
        if k > 0:
            k -= 1
    return lcp


def suffix_array_workload(text: str) -> int:
    sa = suffix_array_build(text)
    lcp = lcp_array_build(text, sa)
    return (checksum_ints(sa) + (checksum_ints(lcp) * 3) + (len(text) * 7)) & MASK_64


def run_length_encode_text(text: str) -> list[tuple[str, int]]:
    if not text:
        return []
    out: list[tuple[str, int]] = []
    current = text[0]
    count = 1
    for ch in text[1:]:
        if ch == current:
            count += 1
            continue
        out.append((current, count))
        current = ch
        count = 1
    out.append((current, count))
    return out


def run_length_decode_runs(runs: list[tuple[str, int]]) -> str:
    out: list[str] = []
    for ch, count in runs:
        if count <= 0:
            raise ValueError("invalid run length")
        out.append(ch * count)
    return "".join(out)


def run_length_encoding_workload(text: str) -> int:
    encoded = run_length_encode_text(text)
    decoded = run_length_decode_runs(encoded)
    if decoded != text:
        raise ValueError("rle round-trip mismatch")

    first_count = encoded[0][1] if encoded else 0
    last_count = encoded[-1][1] if encoded else 0
    return (
        checksum_bytes(decoded)
        + (len(encoded) * 3)
        + (first_count * 5)
        + (last_count * 7)
    ) & MASK_64


def permutations_workload(items: list[int]) -> int:
    arr = items.copy()
    total = 0
    count = 0

    def dfs(start: int) -> None:
        nonlocal total, count
        if start == len(arr):
            sig = 0
            for i, v in enumerate(arr):
                sig = (sig + ((i + 1) * signed_i64_to_u64(v))) & MASK_64
            total = (total + sig) & MASK_64
            count += 1
            return

        for i in range(start, len(arr)):
            arr[start], arr[i] = arr[i], arr[start]
            dfs(start + 1)
            arr[start], arr[i] = arr[i], arr[start]

    dfs(0)
    return (total + (count * 17) + len(items)) & MASK_64


def combinations_workload(n: int, k: int) -> int:
    total = 0
    count = 0
    current: list[int] = []

    def dfs(start: int) -> None:
        nonlocal total, count
        if len(current) == k:
            sig = (len(current) * 11) & MASK_64
            for i, v in enumerate(current):
                sig = (sig + ((i + 1) * v)) & MASK_64
            total = (total + sig) & MASK_64
            count += 1
            return

        remaining = k - len(current)
        limit = n - remaining + 1
        for x in range(start, limit + 1):
            current.append(x)
            dfs(x + 1)
            current.pop()

    dfs(1)
    return (total + (count * 13) + (n * 5) + k) & MASK_64


def subsets_workload(items: list[int]) -> int:
    total = 0
    count = 0
    current: list[int] = []

    def dfs(index: int) -> None:
        nonlocal total, count
        sig = (len(current) * 19) & MASK_64
        for i, v in enumerate(current):
            sig = (sig + ((i + 1) * signed_i64_to_u64(v + 37))) & MASK_64
        total = (total + sig) & MASK_64
        count += 1

        for i in range(index, len(items)):
            current.append(items[i])
            dfs(i + 1)
            current.pop()

    dfs(0)
    return (total + (count * 7) + (len(items) * 3)) & MASK_64


def generate_parentheses_workload(n: int) -> int:
    total = 0
    count = 0

    def dfs(partial: str, opened: int, closed: int) -> None:
        nonlocal total, count
        if len(partial) == 2 * n:
            total = (total + checksum_bytes(partial)) & MASK_64
            count += 1
            return
        if opened < n:
            dfs(partial + "(", opened + 1, closed)
        if closed < opened:
            dfs(partial + ")", opened, closed + 1)

    dfs("", 0, 0)
    return (total + (count * 23) + n) & MASK_64


def n_queens_count(n: int) -> int:
    if n <= 0:
        return 0
    full = (1 << n) - 1

    def dfs(cols: int, diag1: int, diag2: int) -> int:
        if cols == full:
            return 1
        available = full & ~(cols | diag1 | diag2)
        count = 0
        while available:
            bit = available & -available
            available -= bit
            count += dfs(cols | bit, ((diag1 | bit) << 1) & full, (diag2 | bit) >> 1)
        return count

    return dfs(0, 0, 0)


def n_queens_workload(n: int) -> int:
    count = n_queens_count(n)
    return checksum_u((count * 97) + n)


def sudoku_solve(grid: list[list[int]]) -> bool:
    def is_safe(row: int, col: int, value: int) -> bool:
        for i in range(9):
            if grid[row][i] == value or grid[i][col] == value:
                return False
        box_r = (row // 3) * 3
        box_c = (col // 3) * 3
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if grid[r][c] == value:
                    return False
        return True

    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                continue
            for value in range(1, 10):
                if not is_safe(r, c, value):
                    continue
                grid[r][c] = value
                if sudoku_solve(grid):
                    return True
                grid[r][c] = 0
            return False
    return True


def sudoku_workload(solvable: list[list[int]], unsolvable: list[list[int]]) -> int:
    grid = [row[:] for row in solvable]
    if not sudoku_solve(grid):
        raise ValueError("sudoku expected solvable")

    flat = [cell for row in grid for cell in row]
    weighted = 0
    for i, cell in enumerate(flat):
        weighted = (weighted + ((i + 1) * cell)) & MASK_64

    impossible = [row[:] for row in unsolvable]
    unsolved_ok = not sudoku_solve(impossible)
    return (checksum_ints(flat) + weighted + (97 if unsolved_ok else 0)) & MASK_64


def stack_workload(values: list[int]) -> int:
    stack: list[int] = []
    total = 0
    for i, v in enumerate(values):
        stack.append(v)
        if i % 3 == 0:
            total = (total + signed_i64_to_u64(stack[-1])) & MASK_64
        if i % 5 == 0:
            total = (total + ((signed_i64_to_u64(stack.pop()) * 3) & MASK_64)) & MASK_64

    while stack:
        total = (total + signed_i64_to_u64(stack.pop())) & MASK_64
    return (total + (len(values) * 11)) & MASK_64


def queue_workload(values: list[int]) -> int:
    q: deque[int] = deque()
    total = 0
    for i, v in enumerate(values):
        q.append(v)
        if i % 4 == 0:
            total = (total + signed_i64_to_u64(q[0])) & MASK_64
        if i % 6 == 0:
            total = (total + ((signed_i64_to_u64(q.popleft()) * 3) & MASK_64)) & MASK_64
        if i % 25 == 0 and q:
            q.append(q.popleft())

    while q:
        total = (total + signed_i64_to_u64(q.popleft())) & MASK_64
    return (total + (len(values) * 13)) & MASK_64


class SNode:
    __slots__ = ("data", "next")

    def __init__(self, data: int) -> None:
        self.data = data
        self.next: SNode | None = None


class SinglyList:
    __slots__ = ("head", "length")

    def __init__(self) -> None:
        self.head: SNode | None = None
        self.length = 0

    def is_empty(self) -> bool:
        return self.head is None

    def insert_head(self, value: int) -> None:
        node = SNode(value)
        node.next = self.head
        self.head = node
        self.length += 1

    def insert_tail(self, value: int) -> None:
        node = SNode(value)
        if self.head is None:
            self.head = node
            self.length += 1
            return
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = node
        self.length += 1

    def delete_head(self) -> int | None:
        if self.head is None:
            return None
        node = self.head
        self.head = node.next
        self.length -= 1
        return node.data

    def delete_tail(self) -> int | None:
        if self.head is None:
            return None
        if self.head.next is None:
            value = self.head.data
            self.head = None
            self.length -= 1
            return value
        cur = self.head
        while cur.next is not None and cur.next.next is not None:
            cur = cur.next
        assert cur.next is not None
        value = cur.next.data
        cur.next = None
        self.length -= 1
        return value

    def get(self, index: int) -> int | None:
        if index < 0 or index >= self.length:
            return None
        cur = self.head
        for _ in range(index):
            assert cur is not None
            cur = cur.next
        return None if cur is None else cur.data

    def reverse(self) -> None:
        prev: SNode | None = None
        cur = self.head
        while cur is not None:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        self.head = prev


def singly_linked_list_workload(values: list[int]) -> int:
    lst = SinglyList()
    total = 0
    for i, v in enumerate(values):
        if i % 2 == 0:
            lst.insert_tail(v)
        else:
            lst.insert_head(v)

        if i % 7 == 0:
            head = lst.get(0)
            if head is not None:
                total = (total + signed_i64_to_u64(head)) & MASK_64
        if i % 11 == 0 and not lst.is_empty():
            removed = lst.delete_tail()
            if removed is not None:
                total = (total + ((signed_i64_to_u64(removed) * 5) & MASK_64)) & MASK_64

    probe = min(lst.length, 1024)
    for idx in range(probe):
        value = lst.get(idx)
        if value is not None:
            total = (total + signed_i64_to_u64(value)) & MASK_64

    lst.reverse()
    while not lst.is_empty():
        value = lst.delete_head()
        if value is not None:
            total = (total + ((signed_i64_to_u64(value) * 3) & MASK_64)) & MASK_64
    return (total + (len(values) * 17)) & MASK_64


class DNode:
    __slots__ = ("data", "prev", "next")

    def __init__(self, data: int) -> None:
        self.data = data
        self.prev: DNode | None = None
        self.next: DNode | None = None


class DoublyList:
    __slots__ = ("head", "tail", "length")

    def __init__(self) -> None:
        self.head: DNode | None = None
        self.tail: DNode | None = None
        self.length = 0

    def is_empty(self) -> bool:
        return self.head is None

    def insert_head(self, value: int) -> None:
        node = DNode(value)
        node.next = self.head
        if self.head is not None:
            self.head.prev = node
        else:
            self.tail = node
        self.head = node
        self.length += 1

    def insert_tail(self, value: int) -> None:
        node = DNode(value)
        node.prev = self.tail
        if self.tail is not None:
            self.tail.next = node
        else:
            self.head = node
        self.tail = node
        self.length += 1

    def delete_head(self) -> int | None:
        if self.head is None:
            return None
        node = self.head
        self.head = node.next
        if self.head is not None:
            self.head.prev = None
        else:
            self.tail = None
        self.length -= 1
        return node.data

    def delete_tail(self) -> int | None:
        if self.tail is None:
            return None
        node = self.tail
        self.tail = node.prev
        if self.tail is not None:
            self.tail.next = None
        else:
            self.head = None
        self.length -= 1
        return node.data

    def get(self, index: int) -> int | None:
        if index < 0 or index >= self.length:
            return None
        cur = self.head
        for _ in range(index):
            assert cur is not None
            cur = cur.next
        return None if cur is None else cur.data

    def reverse(self) -> None:
        cur = self.head
        while cur is not None:
            cur.prev, cur.next = cur.next, cur.prev
            cur = cur.prev
        self.head, self.tail = self.tail, self.head


def doubly_linked_list_workload(values: list[int]) -> int:
    lst = DoublyList()
    total = 0
    for i, v in enumerate(values):
        if i % 3 == 0:
            lst.insert_head(v)
        else:
            lst.insert_tail(v)

        if i % 9 == 0 and not lst.is_empty():
            removed = lst.delete_head()
            if removed is not None:
                total = (total + ((signed_i64_to_u64(removed) * 3) & MASK_64)) & MASK_64
        if i % 13 == 0 and not lst.is_empty():
            removed = lst.delete_tail()
            if removed is not None:
                total = (total + ((signed_i64_to_u64(removed) * 5) & MASK_64)) & MASK_64

    probe = min(lst.length, 1024)
    for idx in range(probe):
        value = lst.get(idx)
        if value is not None:
            total = (total + signed_i64_to_u64(value)) & MASK_64

    lst.reverse()
    while not lst.is_empty():
        value = lst.delete_tail()
        if value is not None:
            total = (total + signed_i64_to_u64(value)) & MASK_64
    return (total + (len(values) * 19)) & MASK_64


class BstNode:
    __slots__ = ("value", "left", "right")

    def __init__(self, value: int) -> None:
        self.value = value
        self.left: BstNode | None = None
        self.right: BstNode | None = None


class Bst:
    __slots__ = ("root", "length")

    def __init__(self) -> None:
        self.root: BstNode | None = None
        self.length = 0

    def insert(self, value: int) -> None:
        if self.root is None:
            self.root = BstNode(value)
            self.length += 1
            return
        cur = self.root
        while True:
            if value < cur.value:
                if cur.left is None:
                    cur.left = BstNode(value)
                    break
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = BstNode(value)
                    break
                cur = cur.right
        self.length += 1

    def search(self, value: int) -> bool:
        cur = self.root
        while cur is not None:
            if value == cur.value:
                return True
            cur = cur.left if value < cur.value else cur.right
        return False

    def inorder(self) -> list[int]:
        out: list[int] = []
        stack: list[BstNode] = []
        cur = self.root
        while cur is not None or stack:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            out.append(cur.value)
            cur = cur.right
        return out

    def get_min(self) -> int | None:
        cur = self.root
        if cur is None:
            return None
        while cur.left is not None:
            cur = cur.left
        return cur.value

    def get_max(self) -> int | None:
        cur = self.root
        if cur is None:
            return None
        while cur.right is not None:
            cur = cur.right
        return cur.value

    def remove(self, value: int) -> bool:
        parent: BstNode | None = None
        cur = self.root

        while cur is not None and cur.value != value:
            parent = cur
            cur = cur.left if value < cur.value else cur.right

        if cur is None:
            return False

        if cur.left is not None and cur.right is not None:
            succ_parent = cur
            succ = cur.right
            assert succ is not None
            while succ.left is not None:
                succ_parent = succ
                succ = succ.left
            cur.value = succ.value
            parent = succ_parent
            cur = succ

        child = cur.left if cur.left is not None else cur.right

        if parent is None:
            self.root = child
        elif parent.left is cur:
            parent.left = child
        else:
            parent.right = child

        self.length -= 1
        return True


def binary_search_tree_workload(values: list[int], queries: list[int], removals: list[int]) -> int:
    tree = Bst()
    for v in values:
        tree.insert(v)

    hits = 0
    for q in queries:
        if tree.search(q):
            hits += 1

    ordered = tree.inorder()
    inorder_checksum = checksum_ints(ordered)
    min_v = tree.get_min() or 0
    max_v = tree.get_max() or 0

    removed = 0
    for v in removals:
        if tree.remove(v):
            removed += 1

    return (
        inorder_checksum
        + (hits * 31)
        + (removed * 17)
        + (signed_i64_to_u64(min_v) * 3)
        + (signed_i64_to_u64(max_v) * 5)
        + (tree.length * 7)
    ) & MASK_64


def min_heap_workload(values: list[int], push_count: int) -> int:
    heap = values.copy()
    heapq.heapify(heap)
    total = 0
    for i in range(push_count):
        if heap:
            total = (total + signed_i64_to_u64(heapq.heappop(heap))) & MASK_64
        new_value = ((i * 97) + 31) % 100_000 - 50_000
        heapq.heappush(heap, new_value)
        if i % 4 == 0 and heap:
            total = (total + ((signed_i64_to_u64(heap[0]) * 3) & MASK_64)) & MASK_64

    while heap:
        total = (total + signed_i64_to_u64(heapq.heappop(heap))) & MASK_64

    return (total + (len(values) * 23) + (push_count * 5)) & MASK_64


def activity_selection(start: list[int], finish: list[int]) -> list[int]:
    if len(start) != len(finish):
        raise ValueError("length mismatch")
    if not start:
        return []
    if any(finish[i] < finish[i - 1] for i in range(1, len(finish))):
        raise ValueError("finish times must be sorted")

    selected = [0]
    last_finish = finish[0]
    for i in range(1, len(start)):
        if start[i] >= last_finish:
            selected.append(i)
            last_finish = finish[i]
    return selected


def activity_selection_workload(start: list[int], finish: list[int]) -> int:
    selected = activity_selection(start, finish)
    return checksum_ints(selected)


def build_huffman_codes(text: str) -> dict[str, str]:
    if not text:
        return {}

    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    heap: list[tuple[int, int, int, object]] = []
    serial = 0
    for ch, count in freq.items():
        heap.append((count, ord(ch), serial, ch))
        serial += 1
    heapq.heapify(heap)

    if len(heap) == 1:
        only = heap[0][3]
        assert isinstance(only, str)
        return {only: "0"}

    while len(heap) > 1:
        f1, m1, _, left = heapq.heappop(heap)
        f2, m2, _, right = heapq.heappop(heap)
        merged = (left, right)
        heapq.heappush(heap, (f1 + f2, min(m1, m2), serial, merged))
        serial += 1

    root = heap[0][3]
    codes: dict[str, str] = {}

    def dfs(node: object, prefix: str) -> None:
        if isinstance(node, str):
            codes[node] = prefix or "0"
            return
        left, right = node
        dfs(left, prefix + "0")
        dfs(right, prefix + "1")

    dfs(root, "")
    return codes


def encode_huffman_text(text: str, codes: dict[str, str]) -> str:
    return "".join(codes[ch] for ch in text)


def decode_huffman_bits(bits: str, codes: dict[str, str]) -> str:
    if not bits:
        return ""
    tree: dict[str, object] = {}
    for symbol, code in codes.items():
        if not code:
            raise ValueError("invalid huffman code")
        cur = tree
        for bit in code:
            if bit not in ("0", "1"):
                raise ValueError("invalid huffman bit")
            nxt = cur.get(bit)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[bit] = nxt
            cur = nxt
        if "$" in cur:
            raise ValueError("duplicate huffman code")
        cur["$"] = symbol

    out: list[str] = []
    cur = tree
    for bit in bits:
        nxt = cur.get(bit)
        if not isinstance(nxt, dict):
            raise ValueError("invalid huffman encoding")
        cur = nxt
        symbol = cur.get("$")
        if isinstance(symbol, str):
            out.append(symbol)
            cur = tree
    if cur is not tree:
        raise ValueError("incomplete huffman bitstream")
    return "".join(out)


def huffman_coding_workload(text: str) -> int:
    codes = build_huffman_codes(text)
    encoded = encode_huffman_text(text, codes)
    decoded = decode_huffman_bits(encoded, codes)
    if decoded != text:
        raise ValueError("huffman round-trip mismatch")

    first_bit = 1 if encoded and encoded[0] == "1" else 0
    last_bit = 1 if encoded and encoded[-1] == "1" else 0
    return (
        checksum_bytes(decoded)
        + (len(encoded) * 3)
        + (len(codes) * 5)
        + (first_bit * 7)
        + (last_bit * 11)
    ) & MASK_64


def job_sequencing_with_deadline_workload(jobs: list[tuple[int, int, int]]) -> int:
    if not jobs:
        return 0

    jobs_sorted = sorted(jobs, key=lambda j: (-j[2], j[1]))
    max_deadline = max(job[1] for job in jobs_sorted)
    if max_deadline <= 0:
        return 0

    slots = [-1] * max_deadline
    count = 0
    profit = 0

    for job_id, deadline, job_profit in jobs_sorted:
        if deadline <= 0 or job_profit <= 0:
            continue
        for i in range(min(deadline, max_deadline) - 1, -1, -1):
            if slots[i] == -1:
                slots[i] = job_id
                count += 1
                profit += job_profit
                break

    return checksum_u(count + (profit * 3))


ROMAN_TABLE = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ROMAN_SUBTRACTIVE = {"IV", "IX", "XL", "XC", "CD", "CM"}


def integer_to_roman(number: int) -> str:
    if number <= 0 or number > 3999:
        raise ValueError("out of range")
    result: list[str] = []
    n = number
    for arabic, roman in ROMAN_TABLE:
        factor, n = divmod(n, arabic)
        if factor:
            result.append(roman * factor)
        if n == 0:
            break
    return "".join(result)


def roman_to_integer(roman: str) -> int:
    if not roman:
        raise ValueError("empty roman numeral")
    total = 0
    i = 0
    while i < len(roman):
        cur = ROMAN_VALUES.get(roman[i])
        if cur is None:
            raise ValueError("invalid roman numeral character")
        if i + 1 < len(roman):
            nxt = ROMAN_VALUES.get(roman[i + 1])
            if nxt is None:
                raise ValueError("invalid roman numeral character")
            if cur < nxt:
                if roman[i : i + 2] not in ROMAN_SUBTRACTIVE:
                    raise ValueError("invalid subtractive pair")
                total += nxt - cur
                i += 2
                continue
        total += cur
        i += 1

    if total <= 0 or total > 3999 or integer_to_roman(total) != roman:
        raise ValueError("invalid canonical roman numeral")
    return total


def integer_to_roman_workload(numbers: list[int]) -> int:
    checksum = 0
    for n in numbers:
        s = integer_to_roman(n)
        checksum = (checksum + checksum_bytes(s) + n) & MASK_64
    return checksum


def roman_to_integer_workload(romans: list[str]) -> int:
    checksum = 0
    for roman in romans:
        value = roman_to_integer(roman)
        checksum = (checksum + (value * 3) + len(roman)) & MASK_64
    return checksum


def convert_temperature(value: float, from_scale: str, to_scale: str) -> float:
    if from_scale == "C":
        kelvin = value + 273.15
    elif from_scale == "F":
        kelvin = ((value - 32.0) * 5.0 / 9.0) + 273.15
    elif from_scale == "K":
        kelvin = value
    elif from_scale == "R":
        kelvin = value * 5.0 / 9.0
    else:
        raise ValueError("unknown temperature scale")

    if kelvin < 0:
        raise ValueError("below absolute zero")

    if to_scale == "C":
        return kelvin - 273.15
    if to_scale == "F":
        return ((kelvin - 273.15) * 9.0 / 5.0) + 32.0
    if to_scale == "K":
        return kelvin
    if to_scale == "R":
        return kelvin * 9.0 / 5.0
    raise ValueError("unknown temperature scale")


def temperature_conversion_workload(cases: list[tuple[float, str, str]]) -> int:
    checksum = 0
    for value, from_scale, to_scale in cases:
        converted = convert_temperature(value, from_scale, to_scale)
        quantized = int(round(converted * 1_000_000))
        checksum = (checksum + signed_i64_to_u64(quantized)) & MASK_64
    return checksum


MILLER_RABIN_BASES = [2, 325, 9375, 28178, 450775, 9_780_504, 1_795_265_022]


def is_prime_miller_rabin(n: int) -> bool:
    if n < 2:
        return False

    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for base in MILLER_RABIN_BASES:
        a = base % n
        if a <= 1:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(1, s):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def miller_rabin_workload(values: list[int]) -> int:
    prime_count = 0
    signature = 0
    for v in values:
        if is_prime_miller_rabin(v):
            prime_count += 1
            signature = (signature + (v & MASK_64)) & MASK_64
    return (prime_count + (signature * 3)) & MASK_64


def matrix_multiply_square(a: list[int], b: list[int], n: int) -> list[int]:
    out = [0] * (n * n)
    for i in range(n):
        for k in range(n):
            aik = a[i * n + k]
            for j in range(n):
                out[i * n + j] += aik * b[k * n + j]
    return out


def matrix_power_square(matrix: list[int], n: int, exponent: int) -> list[int]:
    result = [0] * (n * n)
    for i in range(n):
        result[i * n + i] = 1
    if exponent == 0:
        return result

    base = matrix.copy()
    exp = exponent
    while exp > 0:
        if exp & 1:
            result = matrix_multiply_square(result, base, n)
        exp >>= 1
        if exp:
            base = matrix_multiply_square(base, base, n)
    return result


def matrix_exponentiation_workload(matrix: list[int], n: int, exponent: int) -> int:
    out = matrix_power_square(matrix, n, exponent)
    return checksum_ints(out)


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
    stack_values = generate_int_data(70_000)
    queue_values = generate_int_data(70_000)
    singly_list_values = generate_int_data(24_000)
    doubly_list_values = generate_int_data(24_000)
    bst_values = [(i * 73) + 19 for i in range(20_000)]
    bst_queries = [bst_values[i % len(bst_values)] if i % 2 == 0 else bst_values[i % len(bst_values)] + 1 for i in range(12_000)]
    bst_removals = [bst_values[(i * 5) % len(bst_values)] for i in range(4_000)]
    min_heap_bench_values = generate_int_data(20_000)
    min_heap_push_count = 8_000
    priority_queue_n = 60_000
    hash_map_n = 60_000
    segment_values = generate_int_data(60_000)
    fenwick_values = generate_int_data(60_000)
    rb_values = [(((i * 73) + 19) % 50_000) - 25_000 for i in range(70_000)]
    rb_queries = [(((i * 97) + 31) % 80_000) - 40_000 for i in range(30_000)]
    lru_capacity = 4_096
    lru_ops = 80_000
    deque_ops = 90_000
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
    aho_patterns = [encode_base26_word(i, 4) for i in range(600)]
    aho_text = "".join(aho_patterns[((i * 37) + 11) % len(aho_patterns)] + "x" for i in range(18_000))
    suffix_text = generate_ascii_string(12_000, 7, 3)
    rle_text = "".join(
        chr(ord("a") + (i % 26)) * ((((i * 7) + 3) % 9) + 1)
        for i in range(20_000)
    )
    caesar_text = "The quick brown fox jumps over the lazy dog 0123456789! " * 6_000
    caesar_key = 8_000
    sha_payload = generate_ascii_string(220_000, 17, 5).encode("ascii")
    permutations_items = [1, 2, 3, 4, 5, 6, 7, 8]
    combinations_n = 16
    combinations_k = 8
    subset_items = [3, 8, 13, 18, 23, 28, 2, 7, 12, 17, 22, 27, 1, 6]
    parentheses_n = 9
    n_queens_n = 10
    sudoku_solvable = [
        [3, 0, 6, 5, 0, 8, 4, 0, 0],
        [5, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 7, 0, 0, 0, 0, 3, 1],
        [0, 0, 3, 0, 1, 0, 0, 8, 0],
        [9, 0, 0, 8, 6, 3, 0, 0, 5],
        [0, 5, 0, 0, 9, 0, 6, 0, 0],
        [1, 3, 0, 0, 0, 0, 2, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 4],
        [0, 0, 5, 2, 0, 6, 3, 0, 0],
    ]
    sudoku_unsolvable = [
        [5, 0, 6, 5, 0, 8, 4, 0, 3],
        [5, 2, 0, 0, 0, 0, 0, 0, 2],
        [1, 8, 7, 0, 0, 0, 0, 3, 1],
        [0, 0, 3, 0, 1, 0, 0, 8, 0],
        [9, 0, 0, 8, 6, 3, 0, 0, 5],
        [0, 5, 0, 0, 9, 0, 6, 0, 0],
        [1, 3, 0, 0, 0, 0, 2, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 4],
        [0, 0, 5, 2, 0, 6, 3, 0, 0],
    ]
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
    weighted_graph_heuristics = [0] * weighted_graph_n
    weighted_graph_goal = weighted_graph_n - 1

    tarjan_n = 700
    tarjan_block = 5
    tarjan_adj: list[list[int]] = [[] for _ in range(tarjan_n)]
    for block_start in range(0, tarjan_n, tarjan_block):
        for offset in range(tarjan_block):
            node = block_start + offset
            nxt = block_start + ((offset + 1) % tarjan_block)
            tarjan_adj[node].append(nxt)
        next_block = block_start + tarjan_block
        if next_block < tarjan_n:
            tarjan_adj[block_start].append(next_block)

    bridges_n = 600
    bridges_adj: list[list[int]] = [[] for _ in range(bridges_n)]
    for block_start in range(0, bridges_n, 3):
        if block_start + 2 >= bridges_n:
            break
        a, b, c = block_start, block_start + 1, block_start + 2
        bridges_adj[a].append(b)
        bridges_adj[b].append(a)
        bridges_adj[b].append(c)
        bridges_adj[c].append(b)
        bridges_adj[c].append(a)
        bridges_adj[a].append(c)
        if block_start + 3 < bridges_n:
            nxt = block_start + 3
            bridges_adj[c].append(nxt)
            bridges_adj[nxt].append(c)

    euler_n = 4_000
    euler_adj: list[list[int]] = [[] for _ in range(euler_n)]
    for i in range(euler_n - 1):
        euler_adj[i].append(i + 1)
        euler_adj[i + 1].append(i)

    flow_n = 120
    flow_capacity = [[0 for _ in range(flow_n)] for _ in range(flow_n)]
    for i in range(flow_n):
        if i + 1 < flow_n:
            flow_capacity[i][i + 1] = ((i * 17) + 3) % 23 + 1
        if i + 2 < flow_n:
            flow_capacity[i][i + 2] = ((i * 31) + 7) % 19 + 1
        if i % 5 == 0 and i + 7 < flow_n:
            flow_capacity[i][i + 7] = ((i * 13) + 11) % 29 + 1

    bipartite_n = 6_000
    bipartite_adj: list[list[int]] = [[] for _ in range(bipartite_n)]
    for i in range(0, bipartite_n, 2):
        if i + 1 < bipartite_n:
            bipartite_adj[i].append(i + 1)
            bipartite_adj[i + 1].append(i)
        if i + 3 < bipartite_n:
            bipartite_adj[i].append(i + 3)
            bipartite_adj[i + 3].append(i)

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
    subset_numbers = [(((i * 17) + 5) % 50) + 1 for i in range(72)]
    subset_targets = [((i * 97) + 31) % 1200 for i in range(64)]
    egg_drop_cases = [((i % 10) + 2, ((i * 131) + 17) % 5000 + 1) for i in range(220)]
    lps_text = generate_ascii_string(700, 7, 3)
    max_product_array = [
        0 if i % 8 == 0 else ((((i * 73) + 19) % 7) - 3)
        for i in range(90_000)
    ]
    coin_set = [1, 2, 3, 5, 7, 11, 13]
    coins_desc = [2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
    frac_values = [60.0, 100.0, 120.0, 140.0, 30.0, 20.0, 80.0, 75.0]
    frac_weights = [10.0, 20.0, 30.0, 40.0, 10.0, 5.0, 15.0, 25.0]

    missing_nums = list(range(50_001))
    missing_target = 30_123
    missing_nums.remove(missing_target)

    bin_text = "10101111100100101111000011101010"
    activity_n = 60_000
    activity_start = [i if (i % 4 != 0) else i - 1 for i in range(activity_n)]
    activity_finish = [i + 1 for i in range(activity_n)]
    if activity_start:
        activity_start[0] = 0

    huffman_text = "".join(chr(ord("a") + i) * (4_000 - i * 250) for i in range(12))

    job_data = [
        (i + 1, ((i * 7) % 600) + 1, ((i * 97) % 900) + 50)
        for i in range(30_000)
    ]

    roman_numbers = [((i * 37) % 3999) + 1 for i in range(10_000)]
    roman_samples = [integer_to_roman(n) for n in roman_numbers]

    scales = ["C", "F", "K", "R"]
    temperature_cases: list[tuple[float, str, str]] = []
    for i in range(24_000):
        from_scale = scales[i % 4]
        to_scale = scales[(i * 3 + 1) % 4]
        if from_scale == "C":
            value = ((i * 17) % 7000) / 10.0 - 273.0
        elif from_scale == "F":
            value = ((i * 13) % 8000) / 10.0 - 459.0
        elif from_scale == "K":
            value = ((i * 11) % 9000) / 10.0
        else:
            value = ((i * 19) % 9000) / 10.0
        temperature_cases.append((value, from_scale, to_scale))

    miller_rabin_values = [((i * 48_271 + 12_345) % 2_000_000) + 2 for i in range(45_000)]
    miller_rabin_values.extend(
        [
            561,
            563,
            838_201,
            838_207,
            3_078_386_641,
            3_078_386_653,
            18_446_744_073_709_551_556,
            18_446_744_073_709_551_557,
        ]
    )

    matrix_exp_dim = 12
    matrix_exp_base = generate_matrix_data(matrix_exp_dim * matrix_exp_dim, 43, 17, 31, 15)
    matrix_exp_exponent = 17

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

    # Maths (16)
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
    cases.append(BenchCase("miller_rabin", "maths", 12, lambda: checksum_u(miller_rabin_workload(miller_rabin_values))))
    cases.append(
        BenchCase(
            "matrix_exponentiation",
            "maths",
            80,
            lambda: checksum_u(matrix_exponentiation_workload(matrix_exp_base, matrix_exp_dim, matrix_exp_exponent)),
        )
    )

    # Data Structures (17)
    cases.append(BenchCase("stack", "data_structures", 10, lambda: checksum_u(stack_workload(stack_values))))
    cases.append(BenchCase("queue", "data_structures", 10, lambda: checksum_u(queue_workload(queue_values))))
    cases.append(BenchCase("singly_linked_list", "data_structures", 8, lambda: checksum_u(singly_linked_list_workload(singly_list_values))))
    cases.append(BenchCase("doubly_linked_list", "data_structures", 8, lambda: checksum_u(doubly_linked_list_workload(doubly_list_values))))
    cases.append(BenchCase("binary_search_tree", "data_structures", 6, lambda: checksum_u(binary_search_tree_workload(bst_values, bst_queries, bst_removals))))
    cases.append(BenchCase("min_heap", "data_structures", 8, lambda: checksum_u(min_heap_workload(min_heap_bench_values, min_heap_push_count))))
    cases.append(BenchCase("trie", "data_structures", 8, lambda: checksum_u(trie_workload(trie_words))))
    cases.append(BenchCase("disjoint_set", "data_structures", 10, lambda: checksum_u(disjoint_set_workload(disjoint_set_n))))
    cases.append(BenchCase("avl_tree", "data_structures", 4, lambda: checksum_u(avl_tree_workload(avl_values, avl_queries))))
    cases.append(BenchCase("max_heap", "data_structures", 8, lambda: checksum_u(max_heap_workload(max_heap_values))))
    cases.append(BenchCase("priority_queue", "data_structures", 8, lambda: checksum_u(priority_queue_workload(priority_queue_n))))
    cases.append(BenchCase("hash_map_open_addressing", "data_structures", 6, lambda: checksum_u(hash_map_open_addressing_workload(hash_map_n))))
    cases.append(BenchCase("segment_tree", "data_structures", 8, lambda: checksum_u(segment_tree_workload(segment_values))))
    cases.append(BenchCase("fenwick_tree", "data_structures", 10, lambda: checksum_u(fenwick_tree_workload(fenwick_values))))
    cases.append(BenchCase("red_black_tree", "data_structures", 6, lambda: checksum_u(red_black_tree_workload(rb_values, rb_queries))))
    cases.append(BenchCase("lru_cache", "data_structures", 8, lambda: checksum_u(lru_cache_workload(lru_capacity, lru_ops))))
    cases.append(BenchCase("deque", "data_structures", 20, lambda: checksum_u(deque_workload(deque_ops))))

    # Dynamic Programming (17)
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
    cases.append(BenchCase("subset_sum", "dynamic_programming", 8, lambda: checksum_u(subset_sum_workload(subset_numbers, subset_targets))))
    cases.append(BenchCase("egg_drop_problem", "dynamic_programming", 120, lambda: checksum_u(egg_drop_workload(egg_drop_cases))))
    cases.append(BenchCase("longest_palindromic_subsequence", "dynamic_programming", 25, lambda: checksum_u(longest_palindromic_subsequence_length(lps_text))))
    cases.append(BenchCase("max_product_subarray", "dynamic_programming", 20, lambda: checksum_u(signed_i64_to_u64(max_product_subarray(max_product_array)))))

    # Graphs (16)
    cases.append(BenchCase("bfs", "graphs", 12, lambda: checksum_ints(bfs(graph_adj, 0))))
    cases.append(BenchCase("dfs", "graphs", 12, lambda: checksum_ints(dfs(graph_adj, 0))))
    cases.append(BenchCase("dijkstra", "graphs", 8, lambda: checksum_ints(dijkstra(weighted_graph_adj, 0))))
    cases.append(
        BenchCase(
            "a_star_search",
            "graphs",
            8,
            lambda: checksum_u(a_star_checksum(weighted_graph_adj, weighted_graph_heuristics, 0, weighted_graph_goal)),
        )
    )
    cases.append(BenchCase("tarjan_scc", "graphs", 10, lambda: checksum_u(tarjan_scc_checksum(tarjan_adj))))
    cases.append(BenchCase("bridges", "graphs", 8, lambda: checksum_u(bridges_checksum(bridges_adj))))
    cases.append(BenchCase("eulerian_path_circuit_undirected", "graphs", 20, lambda: checksum_u(eulerian_checksum(euler_adj))))
    cases.append(BenchCase("ford_fulkerson", "graphs", 4, lambda: checksum_u(ford_fulkerson_max_flow(flow_capacity, 0, flow_n - 1))))
    cases.append(BenchCase("bipartite_check_bfs", "graphs", 16, lambda: checksum_bool(is_bipartite_bfs_adj(bipartite_adj))))
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

    # Conversions (7)
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
    cases.append(BenchCase("roman_to_integer", "conversions", 120, lambda: checksum_u(roman_to_integer_workload(roman_samples))))
    cases.append(BenchCase("integer_to_roman", "conversions", 120, lambda: checksum_u(integer_to_roman_workload(roman_numbers))))
    cases.append(BenchCase("temperature_conversion", "conversions", 120, lambda: checksum_u(temperature_conversion_workload(temperature_cases))))

    # Ciphers (1)
    cases.append(BenchCase("caesar_cipher", "ciphers", 80, lambda: checksum_u(caesar_cipher_workload(caesar_text, caesar_key))))

    # Hashing (1)
    cases.append(BenchCase("sha256", "hashing", 10, lambda: checksum_u(sha256_workload(sha_payload))))

    # Greedy (7)
    cases.append(BenchCase("best_time_to_buy_sell_stock", "greedy_methods", 20, lambda: checksum_u(max_profit(prices))))
    cases.append(BenchCase("minimum_coin_change", "greedy_methods", 200, lambda: checksum_ints(minimum_coin_change(coins_desc, 987))))
    cases.append(BenchCase("minimum_waiting_time", "greedy_methods", 15, lambda: checksum_u(minimum_waiting_time(waiting_queries))))
    cases.append(BenchCase("fractional_knapsack", "greedy_methods", 600, lambda: checksum_u(int(fractional_knapsack(frac_values, frac_weights, 50.0) * 1_000_000))))
    cases.append(BenchCase("activity_selection", "greedy_methods", 120, lambda: checksum_u(activity_selection_workload(activity_start, activity_finish))))
    cases.append(BenchCase("huffman_coding", "greedy_methods", 60, lambda: checksum_u(huffman_coding_workload(huffman_text))))
    cases.append(BenchCase("job_sequencing_with_deadline", "greedy_methods", 15, lambda: checksum_u(job_sequencing_with_deadline_workload(job_data))))

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

    # Backtracking (6)
    cases.append(BenchCase("permutations", "backtracking", 3, lambda: checksum_u(permutations_workload(permutations_items))))
    cases.append(BenchCase("combinations", "backtracking", 6, lambda: checksum_u(combinations_workload(combinations_n, combinations_k))))
    cases.append(BenchCase("subsets", "backtracking", 6, lambda: checksum_u(subsets_workload(subset_items))))
    cases.append(BenchCase("generate_parentheses", "backtracking", 12, lambda: checksum_u(generate_parentheses_workload(parentheses_n))))
    cases.append(BenchCase("n_queens", "backtracking", 60, lambda: checksum_u(n_queens_workload(n_queens_n))))
    cases.append(BenchCase("sudoku_solver", "backtracking", 40, lambda: checksum_u(sudoku_workload(sudoku_solvable, sudoku_unsolvable))))

    # Strings (13)
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
    cases.append(BenchCase("aho_corasick", "strings", 20, lambda: checksum_u(aho_corasick_workload(aho_patterns, aho_text))))
    cases.append(BenchCase("suffix_array", "strings", 6, lambda: checksum_u(suffix_array_workload(suffix_text))))
    cases.append(BenchCase("run_length_encoding", "strings", 40, lambda: checksum_u(run_length_encoding_workload(rle_text))))

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
