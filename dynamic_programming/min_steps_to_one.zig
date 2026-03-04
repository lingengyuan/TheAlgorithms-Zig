//! Minimum Steps To One - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_steps_to_one.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MinStepsError = error{
    InvalidInput,
    Overflow,
};

/// Returns the minimum steps to reduce `number` to 1 using:
/// - decrement by 1
/// - divide by 2 when divisible
/// - divide by 3 when divisible
/// Time complexity: O(n), Space complexity: O(n)
pub fn minStepsToOne(
    allocator: Allocator,
    number: i32,
) (MinStepsError || Allocator.Error)!u32 {
    if (number <= 0) return MinStepsError.InvalidInput;
    if (number == 1) return 0;

    const n: usize = @intCast(number);
    const n_plus_one = @addWithOverflow(n, @as(usize, 1));
    if (n_plus_one[1] != 0) return MinStepsError.Overflow;

    const sentinel_add = @addWithOverflow(number, @as(i32, 1));
    if (sentinel_add[1] != 0) return MinStepsError.Overflow;
    const sentinel: u32 = @intCast(sentinel_add[0]);

    const table = try allocator.alloc(u32, n_plus_one[0]);
    defer allocator.free(table);
    @memset(table, sentinel);

    table[1] = 0;
    for (1..n) |i| {
        const base = table[i];
        const next = @addWithOverflow(base, @as(u32, 1));
        if (next[1] != 0) return MinStepsError.Overflow;

        if (next[0] < table[i + 1]) table[i + 1] = next[0];

        const double = @mulWithOverflow(i, @as(usize, 2));
        if (double[1] == 0 and double[0] <= n and next[0] < table[double[0]]) {
            table[double[0]] = next[0];
        }

        const triple = @mulWithOverflow(i, @as(usize, 3));
        if (triple[1] == 0 and triple[0] <= n and next[0] < table[triple[0]]) {
            table[triple[0]] = next[0];
        }
    }

    return table[n];
}

test "min steps to one: python samples" {
    try testing.expectEqual(@as(u32, 3), try minStepsToOne(testing.allocator, 10));
    try testing.expectEqual(@as(u32, 4), try minStepsToOne(testing.allocator, 15));
    try testing.expectEqual(@as(u32, 2), try minStepsToOne(testing.allocator, 6));
}

test "min steps to one: boundary and invalid input" {
    try testing.expectEqual(@as(u32, 0), try minStepsToOne(testing.allocator, 1));
    try testing.expectError(MinStepsError.InvalidInput, minStepsToOne(testing.allocator, 0));
    try testing.expectError(MinStepsError.InvalidInput, minStepsToOne(testing.allocator, -8));
}

test "min steps to one: extreme powers" {
    // 3^10 -> 10 repeated divisions by 3
    try testing.expectEqual(@as(u32, 10), try minStepsToOne(testing.allocator, 59049));
    // 2^20 -> 20 repeated divisions by 2
    try testing.expectEqual(@as(u32, 20), try minStepsToOne(testing.allocator, 1_048_576));
}
