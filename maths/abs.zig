//! Absolute Value Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/abs.py

const std = @import("std");
const testing = std.testing;

pub const AbsError = error{EmptyInput};

/// Returns absolute value of `num`.
/// Time complexity: O(1), Space complexity: O(1)
pub fn absVal(num: f64) f64 {
    return if (num < 0.0) -num else num;
}

/// Returns element with minimum absolute value from `x`.
/// Time complexity: O(n), Space complexity: O(1)
pub fn absMin(x: []const i64) AbsError!i64 {
    if (x.len == 0) return AbsError.EmptyInput;

    var best = x[0];
    for (x) |value| {
        if (absAsU128(value) < absAsU128(best)) {
            best = value;
        }
    }
    return best;
}

/// Returns element with maximum absolute value from `x`.
/// Time complexity: O(n), Space complexity: O(1)
pub fn absMax(x: []const i64) AbsError!i64 {
    if (x.len == 0) return AbsError.EmptyInput;

    var best = x[0];
    for (x) |value| {
        if (absAsU128(value) > absAsU128(best)) {
            best = value;
        }
    }
    return best;
}

/// Returns element with maximum absolute value, matching Python `sorted(x, key=abs)[-1]` tie behavior.
/// Time complexity: O(n), Space complexity: O(1)
pub fn absMaxSort(x: []const i64) AbsError!i64 {
    if (x.len == 0) return AbsError.EmptyInput;

    var best = x[0];
    var best_abs = absAsU128(best);
    for (x[1..]) |value| {
        const value_abs = absAsU128(value);
        // Python's stable sort then [-1] selects the last element among equal abs values.
        if (value_abs >= best_abs) {
            best = value;
            best_abs = value_abs;
        }
    }
    return best;
}

fn absAsU128(value: i64) u128 {
    const widened: i128 = value;
    return if (widened < 0) @intCast(-widened) else @intCast(widened);
}

test "abs: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 5.1), absVal(-5.1), 1e-12);
    try testing.expect(absVal(-5.0) == absVal(5.0));
    try testing.expectApproxEqAbs(@as(f64, 0.0), absVal(0.0), 1e-12);

    try testing.expectEqual(@as(i64, 0), try absMin(&[_]i64{ 0, 5, 1, 11 }));
    try testing.expectEqual(@as(i64, -2), try absMin(&[_]i64{ 3, -10, -2 }));

    try testing.expectEqual(@as(i64, 11), try absMax(&[_]i64{ 0, 5, 1, 11 }));
    try testing.expectEqual(@as(i64, -10), try absMax(&[_]i64{ 3, -10, -2 }));

    try testing.expectEqual(@as(i64, 11), try absMaxSort(&[_]i64{ 0, 5, 1, 11 }));
    try testing.expectEqual(@as(i64, -10), try absMaxSort(&[_]i64{ 3, -10, -2 }));
}

test "abs: edge and extreme cases" {
    try testing.expectError(AbsError.EmptyInput, absMin(&[_]i64{}));
    try testing.expectError(AbsError.EmptyInput, absMax(&[_]i64{}));
    try testing.expectError(AbsError.EmptyInput, absMaxSort(&[_]i64{}));

    try testing.expectEqual(@as(i64, std.math.minInt(i64)), try absMax(&[_]i64{ std.math.minInt(i64), std.math.maxInt(i64) }));
    try testing.expectEqual(@as(i64, 5), try absMaxSort(&[_]i64{ -5, 5 }));
}
