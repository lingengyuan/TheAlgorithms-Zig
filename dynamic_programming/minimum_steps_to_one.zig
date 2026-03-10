//! Minimum Steps To One - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_steps_to_one.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const base = @import("min_steps_to_one.zig");

pub const MinimumStepsError = base.MinStepsError;

/// Compatibility wrapper for the Python file name.
/// Returns the minimum steps needed to reduce `number` to 1.
/// Time complexity: O(n), Space complexity: O(n)
pub fn minimumStepsToOne(
    allocator: Allocator,
    number: i32,
) (MinimumStepsError || Allocator.Error)!u32 {
    return base.minStepsToOne(allocator, number);
}

test "minimum steps to one: python samples" {
    try testing.expectEqual(@as(u32, 3), try minimumStepsToOne(testing.allocator, 10));
    try testing.expectEqual(@as(u32, 4), try minimumStepsToOne(testing.allocator, 15));
    try testing.expectEqual(@as(u32, 2), try minimumStepsToOne(testing.allocator, 6));
}

test "minimum steps to one: boundary and invalid input" {
    try testing.expectEqual(@as(u32, 0), try minimumStepsToOne(testing.allocator, 1));
    try testing.expectError(MinimumStepsError.InvalidInput, minimumStepsToOne(testing.allocator, 0));
    try testing.expectError(MinimumStepsError.InvalidInput, minimumStepsToOne(testing.allocator, -4));
}

test "minimum steps to one: extreme powers" {
    try testing.expectEqual(@as(u32, 10), try minimumStepsToOne(testing.allocator, 59_049));
    try testing.expectEqual(@as(u32, 20), try minimumStepsToOne(testing.allocator, 1_048_576));
}
