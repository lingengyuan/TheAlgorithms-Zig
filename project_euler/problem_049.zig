//! Project Euler Problem 49: Prime Permutations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_049/sol1.py

const std = @import("std");
const testing = std.testing;

fn digitSignature(number: u32) [10]u8 {
    var signature = [_]u8{0} ** 10;
    var current = number;
    if (current == 0) {
        signature[0] = 1;
        return signature;
    }
    while (current > 0) : (current /= 10) {
        signature[current % 10] += 1;
    }
    return signature;
}

fn sameDigits(a: u32, b: u32) bool {
    return std.meta.eql(digitSignature(a), digitSignature(b));
}

/// Checks primality in O(sqrt(n)).
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn isPrime(number: u32) bool {
    if (number > 1 and number < 4) return true;
    if (number < 2 or number % 2 == 0 or number % 3 == 0) return false;

    var i: u32 = 5;
    while (@as(u64, i) * @as(u64, i) <= number) : (i += 6) {
        if (number % i == 0 or number % (i + 2) == 0) return false;
    }
    return true;
}

/// Binary-searches a prime inside a sorted prime list.
///
/// Time complexity: O(log n)
/// Space complexity: O(1)
pub fn search(target: u32, prime_list: []const u32) bool {
    var left: usize = 0;
    var right: usize = prime_list.len;
    while (left < right) {
        const middle = left + (right - left) / 2;
        const value = prime_list[middle];
        if (value == target) return true;
        if (value < target) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return false;
}

fn fourDigitPrimes(allocator: std.mem.Allocator) ![]u32 {
    var out = std.ArrayListUnmanaged(u32){};
    defer out.deinit(allocator);

    var n: u32 = 1001;
    while (n < 10_000) : (n += 2) {
        if (isPrime(n)) try out.append(allocator, n);
    }
    return out.toOwnedSlice(allocator);
}

/// Returns the 12-digit concatenation of the non-trivial 4-digit prime permutation sequence.
///
/// Time complexity: O(p^2 log p)
/// Space complexity: O(p)
pub fn solution(allocator: std.mem.Allocator) !u64 {
    const primes = try fourDigitPrimes(allocator);
    defer allocator.free(primes);

    var best: u64 = 0;
    for (0..primes.len) |i| {
        const a = primes[i];
        var j = i + 1;
        while (j < primes.len) : (j += 1) {
            const b = primes[j];
            const diff = b - a;
            const c = b + diff;
            if (c >= 10_000) break;
            if (!sameDigits(a, b) or !sameDigits(a, c)) continue;
            if (!search(c, primes)) continue;

            const concat = @as(u64, a) * 100_000_000 + @as(u64, b) * 10_000 + c;
            if (concat > best) best = concat;
        }
    }
    return best;
}

test "problem 049: python reference" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(u64, 296_962_999_629), try solution(allocator));
}

test "problem 049: helper semantics and extremes" {
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(27));
    try testing.expect(!isPrime(87));
    try testing.expect(isPrime(563));
    try testing.expect(isPrime(2999));
    try testing.expect(!isPrime(67_483));

    try testing.expect(search(3, &[_]u32{ 1, 2, 3 }));
    try testing.expect(!search(4, &[_]u32{ 1, 2, 3 }));
    try testing.expect(!search(101, &[_]u32{ 0, 20, 40, 60, 80, 100 }));

    try testing.expect(sameDigits(1487, 4817));
    try testing.expect(sameDigits(1487, 8147));
    try testing.expect(!sameDigits(1487, 4818));
}
