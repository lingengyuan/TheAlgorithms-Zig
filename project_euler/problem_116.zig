//! Project Euler Problem 116: Red, Green or Blue Tiles - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_116/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of single-colour tilings using tile lengths 2, 3, or 4, with at least one coloured tile.
/// Time complexity: O(length^2)
/// Space complexity: O(length)
pub fn solution(allocator: std.mem.Allocator, length: u32) !u64 {
    const ways = try allocator.alloc([3]u64, length + 1);
    defer allocator.free(ways);
    for (ways) |*entry| entry.* = .{ 0, 0, 0 };

    var row_length: u32 = 0;
    while (row_length <= length) : (row_length += 1) {
        var tile_length: u32 = 2;
        while (tile_length <= 4) : (tile_length += 1) {
            var tile_start: u32 = 0;
            while (tile_start + tile_length <= row_length) : (tile_start += 1) {
                ways[row_length][tile_length - 2] += ways[row_length - tile_start - tile_length][tile_length - 2] + 1;
            }
        }
    }

    const final = ways[length];
    return final[0] + final[1] + final[2];
}

test "problem 116: python reference" {
    try testing.expectEqual(@as(u64, 12), try solution(testing.allocator, 5));
    try testing.expectEqual(@as(u64, 20492570929), try solution(testing.allocator, 50));
}

test "problem 116: short rows" {
    try testing.expectEqual(@as(u64, 0), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(u64, 0), try solution(testing.allocator, 1));
    try testing.expectEqual(@as(u64, 1), try solution(testing.allocator, 2));
}
