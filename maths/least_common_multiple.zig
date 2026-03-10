//! Least Common Multiple - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/least_common_multiple.py

const std = @import("std");
const testing = std.testing;
const lcm_mod = @import("lcm.zig");

/// Finds the least common multiple by repeated addition.
/// Time complexity: O(lcm / max(a, b)), Space complexity: O(1)
pub fn leastCommonMultipleSlow(first_num: i64, second_num: i64) u64 {
    if (first_num == 0 or second_num == 0) return 0;
    const abs_first = absI64ToU64(first_num);
    const abs_second = absI64ToU64(second_num);
    const max_num = @max(abs_first, abs_second);
    var common_mult = max_num;
    while (common_mult % abs_first != 0 or common_mult % abs_second != 0) {
        common_mult += max_num;
    }
    return common_mult;
}

/// Finds the least common multiple via the gcd-based fast formula.
/// Time complexity: O(log(min(a, b))), Space complexity: O(1)
pub fn leastCommonMultipleFast(first_num: i64, second_num: i64) u64 {
    return lcm_mod.lcm(first_num, second_num);
}

fn absI64ToU64(value: i64) u64 {
    const wide: i128 = value;
    return @intCast(if (wide < 0) -wide else wide);
}

test "least common multiple: python reference cases" {
    const inputs = [_][2]i64{
        .{ 10, 20 },
        .{ 13, 15 },
        .{ 4, 31 },
        .{ 10, 42 },
        .{ 43, 34 },
        .{ 5, 12 },
        .{ 12, 25 },
        .{ 10, 25 },
        .{ 6, 9 },
    };
    const expected = [_]u64{ 20, 195, 124, 210, 1462, 60, 300, 50, 18 };

    for (inputs, expected) |pair, want| {
        try testing.expectEqual(want, leastCommonMultipleSlow(pair[0], pair[1]));
        try testing.expectEqual(want, leastCommonMultipleFast(pair[0], pair[1]));
    }
}

test "least common multiple: zero and sign edge cases" {
    try testing.expectEqual(@as(u64, 0), leastCommonMultipleSlow(0, 9));
    try testing.expectEqual(@as(u64, 0), leastCommonMultipleFast(9, 0));
    try testing.expectEqual(@as(u64, 42), leastCommonMultipleSlow(-21, 6));
    try testing.expectEqual(@as(u64, 42), leastCommonMultipleFast(-21, 6));
}

