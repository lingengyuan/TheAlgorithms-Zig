//! Project Euler Problem 37: Truncatable Primes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_037/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem037Error = error{
    OutOfMemory,
};

fn pow10(exp: u32) u64 {
    var result: u64 = 1;
    var i: u32 = 0;
    while (i < exp) : (i += 1) result *= 10;
    return result;
}

fn digitCount(n: u64) u32 {
    var count: u32 = 1;
    var value = n;
    while (value >= 10) : (count += 1) value /= 10;
    return count;
}

/// Checks primality in O(sqrt(n)), matching the Python helper semantics.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn isPrime(number: u64) bool {
    if (number > 1 and number < 4) return true;
    if (number < 2 or number % 2 == 0 or number % 3 == 0) return false;

    var i: u64 = 5;
    while (i * i <= number) : (i += 6) {
        if (number % i == 0 or number % (i + 2) == 0) return false;
    }
    return true;
}

/// Returns all left/right truncations of `n` in the same order as the Python module.
///
/// Time complexity: O(digits)
/// Space complexity: O(digits)
pub fn listTruncatedNums(n: u64, allocator: std.mem.Allocator) Problem037Error![]u64 {
    const digits = digitCount(n);
    const out = try allocator.alloc(u64, 1 + (digits - 1) * 2);
    out[0] = n;

    var idx: usize = 1;
    var i: u32 = 1;
    while (i < digits) : (i += 1) {
        out[idx] = n % pow10(digits - i);
        idx += 1;
        out[idx] = n / pow10(i);
        idx += 1;
    }
    return out;
}

/// Python's three-digit prefix/suffix filter used before full truncation checks.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn validate(n: u64) bool {
    const digits = digitCount(n);
    if (digits <= 3) return true;

    const suffix = n % 1000;
    const prefix = n / pow10(digits - 3);
    return isPrime(suffix) and isPrime(prefix);
}

/// Computes the first `count` truncatable primes.
///
/// Time complexity: depends on search horizon, roughly O(k * sqrt(n))
/// Space complexity: O(count)
pub fn computeTruncatedPrimes(count: u32, allocator: std.mem.Allocator) Problem037Error![]u64 {
    if (count == 0) return try allocator.alloc(u64, 0);

    var found = std.ArrayListUnmanaged(u64){};
    defer found.deinit(allocator);

    var num: u64 = 13;
    while (found.items.len != count) : (num += 2) {
        if (!validate(num)) continue;

        const truncated = try listTruncatedNums(num, allocator);
        defer allocator.free(truncated);

        var all_prime = true;
        for (truncated) |candidate| {
            if (!isPrime(candidate)) {
                all_prime = false;
                break;
            }
        }

        if (all_prime) try found.append(allocator, num);
    }

    return try allocator.dupe(u64, found.items);
}

/// Returns the sum of the eleven truncatable primes.
pub fn solution(allocator: std.mem.Allocator) Problem037Error!u64 {
    const truncated = try computeTruncatedPrimes(11, allocator);
    defer allocator.free(truncated);

    var total: u64 = 0;
    for (truncated) |value| total += value;
    return total;
}

test "problem 037: python reference" {
    const allocator = testing.allocator;
    const truncated = try computeTruncatedPrimes(11, allocator);
    defer allocator.free(truncated);

    try testing.expectEqualSlices(
        u64,
        &[_]u64{ 23, 37, 53, 73, 313, 317, 373, 797, 3137, 3797, 739397 },
        truncated,
    );
    try testing.expectEqual(@as(u64, 748_317), try solution(allocator));
}

test "problem 037: helpers and extremes" {
    const allocator = testing.allocator;
    const example = try listTruncatedNums(927_628, allocator);
    defer allocator.free(example);
    const none = try computeTruncatedPrimes(0, allocator);
    defer allocator.free(none);

    try testing.expectEqualSlices(u64, &[_]u64{ 927_628, 27_628, 92_762, 7_628, 9_276, 628, 927, 28, 92, 8, 9 }, example);
    try testing.expect(!validate(74_679));
    try testing.expect(!validate(235_693));
    try testing.expect(validate(3_797));
    try testing.expectEqual(@as(usize, 0), none.len);
}
