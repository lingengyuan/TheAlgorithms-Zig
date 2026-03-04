//! Fast Fibonacci (Doubling) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/fast_fibonacci.py

const std = @import("std");
const testing = std.testing;

pub const FastFibonacciError = error{
    NegativeInput,
    Overflow,
};

const Pair = struct {
    fn_: u128, // F(n)
    fn1: u128, // F(n+1)
};

/// Returns Fibonacci F(n) using fast doubling in O(log n).
pub fn fastFibonacci(n: i64) FastFibonacciError!u128 {
    if (n < 0) return FastFibonacciError.NegativeInput;
    return (try fibPair(@intCast(n))).fn_;
}

fn fibPair(n: u64) FastFibonacciError!Pair {
    if (n == 0) return .{ .fn_ = 0, .fn1 = 1 };

    const half = try fibPair(n / 2);
    const a = half.fn_;
    const b = half.fn1;

    const two_b = @mulWithOverflow(b, @as(u128, 2));
    if (two_b[1] != 0) return FastFibonacciError.Overflow;
    const two_b_minus_a = @subWithOverflow(two_b[0], a);
    if (two_b_minus_a[1] != 0) return FastFibonacciError.Overflow;

    const c = @mulWithOverflow(a, two_b_minus_a[0]); // F(2n)
    if (c[1] != 0) return FastFibonacciError.Overflow;

    const aa = @mulWithOverflow(a, a);
    if (aa[1] != 0) return FastFibonacciError.Overflow;
    const bb = @mulWithOverflow(b, b);
    if (bb[1] != 0) return FastFibonacciError.Overflow;
    const d = @addWithOverflow(aa[0], bb[0]); // F(2n+1)
    if (d[1] != 0) return FastFibonacciError.Overflow;

    if ((n & 1) == 0) {
        return .{ .fn_ = c[0], .fn1 = d[0] };
    }

    const cd = @addWithOverflow(c[0], d[0]);
    if (cd[1] != 0) return FastFibonacciError.Overflow;
    return .{ .fn_ = d[0], .fn1 = cd[0] };
}

test "fast fibonacci: python sequence sample" {
    const expected = [_]u128{ 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144 };
    for (expected, 0..) |value, i| {
        try testing.expectEqual(value, try fastFibonacci(@intCast(i)));
    }
}

test "fast fibonacci: known large value" {
    try testing.expectEqual(@as(u128, 354224848179261915075), try fastFibonacci(100));
}

test "fast fibonacci: invalid and overflow cases" {
    try testing.expectError(FastFibonacciError.NegativeInput, fastFibonacci(-1));
    try testing.expectEqual(@as(u128, 205697230343233228174223751303346572685), try fastFibonacci(185));
    try testing.expectError(FastFibonacciError.Overflow, fastFibonacci(186));
    try testing.expectError(FastFibonacciError.Overflow, fastFibonacci(187));
}
