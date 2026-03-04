//! Lucas Series - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/lucas_series.py

const std = @import("std");
const testing = std.testing;

pub const LucasError = error{ InvalidInput, Overflow };

/// Returns the nth Lucas number with a recursive implementation.
/// Time complexity: O(2^n), Space complexity: O(n) recursion depth.
pub fn recursiveLucasNumber(n_th_number: i64) LucasError!u128 {
    if (n_th_number < 0) return LucasError.InvalidInput;
    return recursiveLucasNumberNonNegative(@intCast(n_th_number));
}

/// Returns the nth Lucas number with a dynamic iterative implementation.
/// Time complexity: O(n), Space complexity: O(1)
pub fn dynamicLucasNumber(n_th_number: i64) LucasError!u128 {
    if (n_th_number < 0) return LucasError.InvalidInput;

    var a: u128 = 2;
    var b: u128 = 1;

    var i: i64 = 0;
    while (i < n_th_number) : (i += 1) {
        const next = @addWithOverflow(a, b);
        if (next[1] != 0) return LucasError.Overflow;
        a = b;
        b = next[0];
    }

    return a;
}

fn recursiveLucasNumberNonNegative(n_th_number: u64) LucasError!u128 {
    if (n_th_number == 0) return 2;
    if (n_th_number == 1) return 1;

    const a = try recursiveLucasNumberNonNegative(n_th_number - 1);
    const b = try recursiveLucasNumberNonNegative(n_th_number - 2);
    const next = @addWithOverflow(a, b);
    if (next[1] != 0) return LucasError.Overflow;
    return next[0];
}

test "lucas series: python reference examples" {
    try testing.expectEqual(@as(u128, 1), try recursiveLucasNumber(1));
    try testing.expectEqual(@as(u128, 15_127), try recursiveLucasNumber(20));
    try testing.expectEqual(@as(u128, 2), try recursiveLucasNumber(0));
    try testing.expectEqual(@as(u128, 167_761), try recursiveLucasNumber(25));

    try testing.expectEqual(@as(u128, 1), try dynamicLucasNumber(1));
    try testing.expectEqual(@as(u128, 15_127), try dynamicLucasNumber(20));
    try testing.expectEqual(@as(u128, 2), try dynamicLucasNumber(0));
    try testing.expectEqual(@as(u128, 167_761), try dynamicLucasNumber(25));
}

test "lucas series: recursive and dynamic agree on small range" {
    var i: i64 = 0;
    while (i <= 30) : (i += 1) {
        try testing.expectEqual(try dynamicLucasNumber(i), try recursiveLucasNumber(i));
    }
}

test "lucas series: invalid and extreme cases" {
    try testing.expectError(LucasError.InvalidInput, recursiveLucasNumber(-1));
    try testing.expectError(LucasError.InvalidInput, dynamicLucasNumber(-1));

    // u128 overflows for Lucas numbers at sufficiently large n.
    try testing.expectError(LucasError.Overflow, dynamicLucasNumber(200));
}
