//! Project Euler Problem 114: Counting Block Combinations I - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_114/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of ways to fill a row of `length` with red blocks of minimum size 3.
/// Time complexity: O(length^3)
/// Space complexity: O(length)
pub fn solution(allocator: std.mem.Allocator, length: u32) !u64 {
    const ways = try allocator.alloc(u64, length + 1);
    defer allocator.free(ways);
    @memset(ways, 1);

    var row_length: u32 = 3;
    while (row_length <= length) : (row_length += 1) {
        var block_length: u32 = 3;
        while (block_length <= row_length) : (block_length += 1) {
            var block_start: u32 = 0;
            while (block_start < row_length - block_length) : (block_start += 1) {
                ways[row_length] += ways[row_length - block_start - block_length - 1];
            }
            ways[row_length] += 1;
        }
    }
    return ways[length];
}

test "problem 114: python reference" {
    try testing.expectEqual(@as(u64, 17), try solution(testing.allocator, 7));
    try testing.expectEqual(@as(u64, 16475640049), try solution(testing.allocator, 50));
}

test "problem 114: edge lengths" {
    try testing.expectEqual(@as(u64, 1), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(u64, 1), try solution(testing.allocator, 2));
    try testing.expectEqual(@as(u64, 2), try solution(testing.allocator, 3));
}
