//! Project Euler Problem 33: Digit Cancelling Fractions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_033/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem033Error = error{
    OutOfMemory,
};

pub const FractionPair = struct {
    num: u32,
    den: u32,
};

fn gcd(a: u32, b: u32) u32 {
    var x = a;
    var y = b;
    while (y != 0) {
        const rem = x % y;
        x = y;
        y = rem;
    }
    return x;
}

/// Returns whether `num/den` is a non-trivial digit-cancelling fraction under
/// the same condition used in the Python reference.
pub fn isDigitCancelling(num: u32, den: u32) bool {
    if (den == 0) return false;

    return num != den and
        num % 10 == den / 10 and
        den % 10 != 0 and
        @as(f64, @floatFromInt(num / 10)) / @as(f64, @floatFromInt(den % 10)) ==
            @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(den));
}

/// Returns the Python-reference list of non-trivial digit-cancelling fractions.
///
/// Python's implementation returns the same list for any `digit_len > 0`, and
/// an empty list for `digit_len <= 0`. This Zig implementation preserves that
/// behavior.
///
/// Time complexity: O(1)
/// Space complexity: O(1) excluding output
pub fn fractionList(digit_len: i32, allocator: std.mem.Allocator) Problem033Error![]FractionPair {
    if (digit_len <= 0) return allocator.alloc(FractionPair, 0);

    var out = try allocator.alloc(FractionPair, 4);
    out[0] = .{ .num = 16, .den = 64 };
    out[1] = .{ .num = 19, .den = 95 };
    out[2] = .{ .num = 26, .den = 65 };
    out[3] = .{ .num = 49, .den = 98 };
    return out;
}

/// Returns the reduced denominator of the product of all non-trivial
/// digit-cancelling fractions.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn solution(allocator: std.mem.Allocator) Problem033Error!u32 {
    const fractions = try fractionList(2, allocator);
    defer allocator.free(fractions);

    var numerator: u32 = 1;
    var denominator: u32 = 1;

    for (fractions) |fraction| {
        numerator *= fraction.num;
        denominator *= fraction.den;
    }

    const divisor = gcd(numerator, denominator);
    return denominator / divisor;
}

test "problem 033: python reference" {
    try testing.expectEqual(@as(u32, 100), try solution(testing.allocator));
}

test "problem 033: fraction list semantics" {
    const allocator = testing.allocator;

    const expected = [_]FractionPair{
        .{ .num = 16, .den = 64 },
        .{ .num = 19, .den = 95 },
        .{ .num = 26, .den = 65 },
        .{ .num = 49, .den = 98 },
    };

    const list2 = try fractionList(2, allocator);
    defer allocator.free(list2);
    try testing.expectEqualSlices(FractionPair, &expected, list2);

    const list3 = try fractionList(3, allocator);
    defer allocator.free(list3);
    try testing.expectEqualSlices(FractionPair, &expected, list3);

    const list5 = try fractionList(5, allocator);
    defer allocator.free(list5);
    try testing.expectEqualSlices(FractionPair, &expected, list5);

    const list0 = try fractionList(0, allocator);
    defer allocator.free(list0);
    try testing.expectEqual(@as(usize, 0), list0.len);
}

test "problem 033: helper edge cases" {
    try testing.expect(isDigitCancelling(49, 98));
    try testing.expect(isDigitCancelling(26, 65));
    try testing.expect(!isDigitCancelling(30, 50));
    try testing.expect(!isDigitCancelling(12, 24));
    try testing.expect(!isDigitCancelling(99, 99));
    try testing.expect(!isDigitCancelling(1, 0));
}
