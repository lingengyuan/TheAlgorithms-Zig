//! Project Euler Problem 48: Self Powers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_048/sol1.py

const std = @import("std");
const testing = std.testing;

const last_digits = 10;
const modulus: u64 = 10_000_000_000;

fn powMod(base: u64, exponent: u64) u64 {
    var result: u64 = 1;
    var current = base % modulus;
    var power = exponent;
    while (power > 0) : (power >>= 1) {
        if ((power & 1) == 1) {
            result = @intCast((@as(u128, result) * @as(u128, current)) % modulus);
        }
        current = @intCast((@as(u128, current) * @as(u128, current)) % modulus);
    }
    return result;
}

fn formatLastDigits(allocator: std.mem.Allocator, value: u64) ![]u8 {
    const out = try allocator.alloc(u8, last_digits);
    var current = value;
    var index = out.len;
    while (index > 0) {
        index -= 1;
        out[index] = @as(u8, @intCast(current % 10)) + '0';
        current /= 10;
    }
    return out;
}

/// Returns the last ten digits of `1^1 + 2^2 + ... + limit^limit`.
/// Caller owns the returned string.
///
/// Time complexity: O(limit log limit)
/// Space complexity: O(1) excluding the returned buffer
pub fn solution(allocator: std.mem.Allocator, limit: u32) ![]u8 {
    var total: u64 = 0;
    var i: u32 = 1;
    while (i <= limit) : (i += 1) {
        total = (total + powMod(i, i)) % modulus;
    }
    return formatLastDigits(allocator, total);
}

test "problem 048: python reference" {
    const allocator = testing.allocator;

    const answer = try solution(allocator, 1000);
    defer allocator.free(answer);
    try testing.expectEqualStrings("9110846700", answer);
}

test "problem 048: helper semantics and extremes" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 1), powMod(1, 1));
    try testing.expectEqual(@as(u64, 1024), powMod(2, 10));
    try testing.expectEqual(@as(u64, 0), powMod(10, 10));

    const example = try solution(allocator, 10);
    defer allocator.free(example);
    try testing.expectEqualStrings("0405071317", example);

    const one = try solution(allocator, 1);
    defer allocator.free(one);
    try testing.expectEqualStrings("0000000001", one);

    const zero = try solution(allocator, 0);
    defer allocator.free(zero);
    try testing.expectEqualStrings("0000000000", zero);
}
