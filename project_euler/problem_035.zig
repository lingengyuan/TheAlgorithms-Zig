//! Project Euler Problem 35: Circular Primes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_035/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem035Error = error{
    OutOfMemory,
};

fn buildSieve(limit: usize, allocator: std.mem.Allocator) Problem035Error![]bool {
    var sieve = try allocator.alloc(bool, limit + 1);
    @memset(sieve, true);

    if (limit >= 0) sieve[0] = false;
    if (limit >= 1) sieve[1] = false;

    var i: usize = 2;
    while (i * i <= limit) : (i += 1) {
        if (!sieve[i]) continue;

        var j = i * i;
        while (j <= limit) : (j += i) {
            sieve[j] = false;
        }
    }

    return sieve;
}

/// Returns true if `n` contains any even decimal digit. This matches the Python
/// helper semantics, including `0 -> true` and checking digits of the absolute
/// value for negatives.
pub fn containsAnEvenDigit(n: i64) bool {
    if (n == 0) return true;

    var value: u64 = @intCast(if (n < 0) -n else n);
    while (value > 0) {
        if ((value % 10) % 2 == 0) return true;
        value /= 10;
    }

    return false;
}

fn rotateLeftDecimal(value: u32) u32 {
    if (value < 10) return value;

    var pow10: u32 = 1;
    var tmp = value;
    while (tmp >= 10) {
        pow10 *= 10;
        tmp /= 10;
    }

    const leading = value / pow10;
    const tail = value % pow10;
    return tail * 10 + leading;
}

/// Returns circular primes up to and including `limit`, matching the Python
/// reference behavior. In particular, the output always starts with `2`, even
/// when `limit < 2`.
///
/// Time complexity: O(limit log log limit + candidates * rotations)
/// Space complexity: O(limit)
pub fn findCircularPrimes(limit: u32, allocator: std.mem.Allocator) Problem035Error![]u32 {
    const sieve = try buildSieve(1_000_000, allocator);
    defer allocator.free(sieve);

    var result = std.ArrayListUnmanaged(u32){};
    defer result.deinit(allocator);
    try result.append(allocator, 2);

    var num: u32 = 3;
    while (num <= limit) : (num += 2) {
        if (!sieve[num] or containsAnEvenDigit(num)) continue;

        var rotation = num;
        var all_prime = true;
        var idx: u32 = 0;
        const digit_count: u32 = blk: {
            var digits: u32 = 1;
            var t = num;
            while (t >= 10) : (digits += 1) t /= 10;
            break :blk digits;
        };

        while (idx < digit_count) : (idx += 1) {
            if (!sieve[rotation]) {
                all_prime = false;
                break;
            }
            rotation = rotateLeftDecimal(rotation);
        }

        if (all_prime) {
            try result.append(allocator, num);
        }
    }

    return try allocator.dupe(u32, result.items);
}

/// Returns the number of circular primes below one million.
pub fn solution(allocator: std.mem.Allocator) Problem035Error!u32 {
    const primes = try findCircularPrimes(1_000_000, allocator);
    defer allocator.free(primes);
    return @intCast(primes.len);
}

test "problem 035: python reference" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(u32, 55), try solution(allocator));

    const under_100 = try findCircularPrimes(100, allocator);
    defer allocator.free(under_100);
    try testing.expectEqual(@as(usize, 13), under_100.len);
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 3, 5, 7, 11, 13, 17, 31, 37, 71, 73, 79, 97 }, under_100);
}

test "problem 035: helper semantics and edge cases" {
    const allocator = testing.allocator;

    try testing.expect(containsAnEvenDigit(0));
    try testing.expect(!containsAnEvenDigit(975317933));
    try testing.expect(containsAnEvenDigit(-245679));
    try testing.expect(containsAnEvenDigit(101));
    try testing.expect(!containsAnEvenDigit(11));

    const weird = try findCircularPrimes(1, allocator);
    defer allocator.free(weird);
    try testing.expectEqual(@as(usize, 1), weird.len);
    try testing.expectEqual(@as(u32, 2), weird[0]);
}
