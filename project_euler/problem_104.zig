//! Project Euler Problem 104: Pandigital Fibonacci Ends - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_104/sol1.py

const std = @import("std");
const testing = std.testing;

fn isPandigitalNine(value: u32) bool {
    if (value < 123_456_789 or value > 987_654_321) return false;

    var seen: u16 = 0;
    var number = value;
    var count: usize = 0;
    while (count < 9) : (count += 1) {
        const digit: u4 = @intCast(number % 10);
        if (digit == 0) return false;
        const bit: u16 = @as(u16, 1) << digit;
        if ((seen & bit) != 0) return false;
        seen |= bit;
        number /= 10;
    }
    return number == 0 and seen == 0b11_1111_1110;
}

fn leadingNineFib(index: u64) u32 {
    const sqrt5 = std.math.sqrt(@as(f64, 5.0));
    const phi = (1.0 + sqrt5) / 2.0;
    const exponent = @as(f64, @floatFromInt(index)) * std.math.log10(phi) - std.math.log10(sqrt5);
    const fractional = exponent - std.math.floor(exponent);
    return @intFromFloat(std.math.pow(f64, 10.0, fractional + 8.0));
}

fn checkLast(number: u128) bool {
    return isPandigitalNine(@intCast(number % 1_000_000_000));
}

fn check(number: u128) bool {
    if (!checkLast(number)) return false;

    var buffer: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{number}) catch return false;
    if (text.len < 9) return false;

    const leading = std.fmt.parseInt(u32, text[0..9], 10) catch return false;
    return isPandigitalNine(leading);
}

/// Returns the smallest Fibonacci index whose first and last nine digits are
/// pandigital.
/// Time complexity: O(answer)
/// Space complexity: O(1)
pub fn solution() u64 {
    var prev: u32 = 1;
    var curr: u32 = 1;
    var index: u64 = 2;

    while (true) {
        index += 1;
        const next = @as(u64, prev) + curr;
        prev = curr;
        curr = @intCast(next % 1_000_000_000);

        if (isPandigitalNine(curr) and isPandigitalNine(leadingNineFib(index))) {
            return index;
        }
    }
}

test "problem 104: python reference" {
    try testing.expectEqual(@as(u64, 329468), solution());
}

test "problem 104: pandigital checks" {
    try testing.expect(check(123456789987654321));
    try testing.expect(!check(120000987654321));
    try testing.expect(check(1234567895765677987654321));
    try testing.expect(checkLast(123456789987654321));
    try testing.expect(checkLast(120000987654321));
    try testing.expect(!checkLast(12345678957656779870004321));
    try testing.expect(isPandigitalNine(123456789));
    try testing.expect(!isPandigitalNine(112345678));
}

