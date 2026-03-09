//! Persistence Of A Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/persistence.py

const std = @import("std");
const testing = std.testing;

pub const PersistenceError = error{NegativeInput};

fn multiplicativeStep(num: u64) u64 {
    if (num < 10) return num;
    var current = num;
    var total: u64 = 1;
    while (current > 0) {
        total *= current % 10;
        current /= 10;
    }
    return total;
}

fn additiveStep(num: u64) u64 {
    if (num < 10) return num;
    var current = num;
    var total: u64 = 0;
    while (current > 0) {
        total += current % 10;
        current /= 10;
    }
    return total;
}

/// Returns the multiplicative persistence of a non-negative integer.
/// Time complexity: O(steps * digits), Space complexity: O(1)
pub fn multiplicativePersistence(num: i64) PersistenceError!u32 {
    if (num < 0) return error.NegativeInput;
    var current: u64 = @intCast(num);
    var steps: u32 = 0;
    while (current >= 10) {
        current = multiplicativeStep(current);
        steps += 1;
    }
    return steps;
}

/// Returns the additive persistence of a non-negative integer.
/// Time complexity: O(steps * digits), Space complexity: O(1)
pub fn additivePersistence(num: i64) PersistenceError!u32 {
    if (num < 0) return error.NegativeInput;
    var current: u64 = @intCast(num);
    var steps: u32 = 0;
    while (current >= 10) {
        current = additiveStep(current);
        steps += 1;
    }
    return steps;
}

test "persistence: python reference examples" {
    try testing.expectEqual(@as(u32, 2), try multiplicativePersistence(217));
    try testing.expectEqual(@as(u32, 3), try additivePersistence(199));
}

test "persistence: edge and extreme cases" {
    try testing.expectEqual(@as(u32, 0), try multiplicativePersistence(0));
    try testing.expectEqual(@as(u32, 0), try additivePersistence(7));
    try testing.expectEqual(@as(u32, 11), try multiplicativePersistence(277777788888899));
    try testing.expectError(error.NegativeInput, multiplicativePersistence(-1));
    try testing.expectError(error.NegativeInput, additivePersistence(-1));
}
