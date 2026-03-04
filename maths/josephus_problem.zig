//! Josephus Problem - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/josephus_problem.py

const std = @import("std");
const testing = std.testing;

pub const JosephusError = error{InvalidInput};

/// Solves Josephus in recursive form and returns zero-based survivor index.
/// Time complexity: O(n), Space complexity: O(n) recursion depth.
pub fn josephusRecursive(num_people: i64, step_size: i64) JosephusError!u64 {
    if (num_people <= 0 or step_size <= 0) return JosephusError.InvalidInput;
    return josephusRecursiveNonNegative(@intCast(num_people), @intCast(step_size));
}

/// Solves Josephus and returns one-based winner index.
/// Time complexity: O(n), Space complexity: O(n) recursion depth.
pub fn findWinner(num_people: i64, step_size: i64) JosephusError!u64 {
    const zero_based = try josephusRecursive(num_people, step_size);
    return zero_based + 1;
}

/// Iterative Josephus solver returning one-based winner index.
/// Time complexity: O(n), Space complexity: O(1)
pub fn josephusIterative(num_people: i64, step_size: i64) JosephusError!u64 {
    if (num_people <= 0 or step_size <= 0) return JosephusError.InvalidInput;

    const n: u64 = @intCast(num_people);
    const k: u64 = @intCast(step_size);

    var result: u64 = 0; // zero-based
    var i: u64 = 1;
    while (i <= n) : (i += 1) {
        result = @intCast((@as(u128, result) + @as(u128, k)) % @as(u128, i));
    }

    return result + 1;
}

fn josephusRecursiveNonNegative(num_people: u64, step_size: u64) u64 {
    if (num_people == 1) return 0;
    const prev = josephusRecursiveNonNegative(num_people - 1, step_size);
    return @intCast((@as(u128, prev) + @as(u128, step_size)) % @as(u128, num_people));
}

test "josephus problem: python reference examples" {
    try testing.expectEqual(@as(u64, 3), try josephusRecursive(7, 3));
    try testing.expectEqual(@as(u64, 4), try josephusRecursive(10, 2));

    try testing.expectEqual(@as(u64, 4), try findWinner(7, 3));
    try testing.expectEqual(@as(u64, 5), try findWinner(10, 2));

    try testing.expectEqual(@as(u64, 3), try josephusIterative(5, 2));
    try testing.expectEqual(@as(u64, 4), try josephusIterative(7, 3));
}

test "josephus problem: invalid input handling" {
    try testing.expectError(JosephusError.InvalidInput, josephusRecursive(0, 2));
    try testing.expectError(JosephusError.InvalidInput, josephusRecursive(7, 0));
    try testing.expectError(JosephusError.InvalidInput, josephusRecursive(-2, 2));
    try testing.expectError(JosephusError.InvalidInput, josephusRecursive(7, -2));

    try testing.expectError(JosephusError.InvalidInput, josephusIterative(0, 2));
    try testing.expectError(JosephusError.InvalidInput, josephusIterative(7, 0));
}

test "josephus problem: edge and extreme cases" {
    try testing.expectEqual(@as(u64, 1), try josephusIterative(1, 1));
    try testing.expectEqual(@as(u64, 1), try josephusIterative(5, 8));
    try testing.expectEqual(@as(u64, 1_000), try josephusIterative(1_000, 1));

    // Closed-form reference for k=2: 2*(n-2^floor(log2(n)))+1
    try testing.expectEqual(@as(u64, 951_425), try josephusIterative(1_000_000, 2));
}
