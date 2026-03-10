//! Project Euler Problem 117: Red, Green, and Blue Tiles - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_117/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of tilings using grey squares and coloured tiles of length 2, 3, and 4.
/// Time complexity: O(length^2)
/// Space complexity: O(length)
pub fn solution(allocator: std.mem.Allocator, length: u32) !u64 {
    const ways = try allocator.alloc(u64, length + 1);
    defer allocator.free(ways);
    @memset(ways, 1);

    var row_length: u32 = 0;
    while (row_length <= length) : (row_length += 1) {
        var tile_length: u32 = 2;
        while (tile_length <= 4) : (tile_length += 1) {
            var tile_start: u32 = 0;
            while (tile_start + tile_length <= row_length) : (tile_start += 1) {
                ways[row_length] += ways[row_length - tile_start - tile_length];
            }
        }
    }
    return ways[length];
}

test "problem 117: python reference" {
    try testing.expectEqual(@as(u64, 15), try solution(testing.allocator, 5));
    try testing.expectEqual(@as(u64, 100808458960497), try solution(testing.allocator, 50));
}

test "problem 117: edge lengths" {
    try testing.expectEqual(@as(u64, 1), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(u64, 1), try solution(testing.allocator, 1));
    try testing.expectEqual(@as(u64, 2), try solution(testing.allocator, 2));
}
