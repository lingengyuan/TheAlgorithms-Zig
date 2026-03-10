//! Project Euler Problem 115: Counting Block Combinations II - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_115/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem115Error = error{InvalidInput};

/// Returns the least row length for which the fill-count function first exceeds one million.
/// Time complexity: O(answer^3) in the direct translated formulation
/// Space complexity: O(answer)
pub fn solution(allocator: std.mem.Allocator, min_block_length: u32) (std.mem.Allocator.Error || Problem115Error)!u32 {
    if (min_block_length == 0) return error.InvalidInput;

    var fill_counts = std.ArrayListUnmanaged(u64){};
    defer fill_counts.deinit(allocator);

    try fill_counts.ensureTotalCapacity(allocator, min_block_length + 16);
    for (0..min_block_length) |_| try fill_counts.append(allocator, 1);

    var n = min_block_length;
    while (true) : (n += 1) {
        try fill_counts.append(allocator, 1);
        var block_length = min_block_length;
        while (block_length <= n) : (block_length += 1) {
            var block_start: u32 = 0;
            while (block_start < n - block_length) : (block_start += 1) {
                fill_counts.items[n] += fill_counts.items[n - block_start - block_length - 1];
            }
            fill_counts.items[n] += 1;
        }
        if (fill_counts.items[n] > 1_000_000) return n;
    }
}

test "problem 115: python reference" {
    try testing.expectEqual(@as(u32, 30), try solution(testing.allocator, 3));
    try testing.expectEqual(@as(u32, 57), try solution(testing.allocator, 10));
    try testing.expectEqual(@as(u32, 168), try solution(testing.allocator, 50));
}

test "problem 115: smaller minimum blocks and invalid input" {
    try testing.expectEqual(@as(u32, 20), try solution(testing.allocator, 1));
    try testing.expectEqual(@as(u32, 26), try solution(testing.allocator, 2));
    try testing.expectError(error.InvalidInput, solution(testing.allocator, 0));
}
