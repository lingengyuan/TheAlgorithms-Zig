//! Project Euler Problem 207: Integer Partition Equations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_207/sol1.py

const std = @import("std");
const testing = std.testing;

fn isPowerOfTwo(value: u64) bool {
    return value != 0 and (value & (value - 1)) == 0;
}

pub fn checkPartitionPerfect(positive_integer: u64) bool {
    const discriminant = 4 * positive_integer + 1;
    const root: u64 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(discriminant))));
    if (root * root != discriminant) return false;
    return isPowerOfTwo(root + 1);
}

/// Returns the smallest m for which the proportion of perfect partitions drops below `max_proportion`.
/// Time complexity: O(sqrt(answer))
/// Space complexity: O(1)
pub fn solution(max_proportion: f64) u64 {
    var total_partitions: u64 = 0;
    var perfect_partitions: u64 = 0;
    var odd_integer: u64 = 3;
    while (true) : (odd_integer += 2) {
        const partition_candidate = (odd_integer * odd_integer - 1) / 4;
        total_partitions += 1;
        if (isPowerOfTwo(odd_integer + 1)) perfect_partitions += 1;
        if (perfect_partitions > 0 and @as(f64, @floatFromInt(perfect_partitions)) / @as(f64, @floatFromInt(total_partitions)) < max_proportion) {
            return partition_candidate;
        }
    }
}

test "problem 207: perfect partition helper" {
    try testing.expect(checkPartitionPerfect(2));
    try testing.expect(!checkPartitionPerfect(6));
}

test "problem 207: python reference" {
    try testing.expect(solution(1.0) > 5);
    try testing.expect(solution(0.5) > 10);
    try testing.expect(solution(3.0 / 13.0) > 185);
    try testing.expectEqual(@as(u64, 44043947822), solution(1.0 / 12345.0));
}
