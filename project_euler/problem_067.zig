//! Project Euler Problem 67: Maximum Path Sum II - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_067/sol1.py

const std = @import("std");
const testing = std.testing;

const triangle_file = @embedFile("problem_067_triangle.txt");

/// Finds the maximum path sum in a triangle encoded as space-separated rows.
/// Time complexity: O(values)
/// Space complexity: O(row_width)
pub fn maxPathSum(allocator: std.mem.Allocator, data: []const u8) !u32 {
    var previous = std.ArrayListUnmanaged(u32){};
    defer previous.deinit(allocator);

    var current = std.ArrayListUnmanaged(u32){};
    defer current.deinit(allocator);

    var lines = std.mem.tokenizeAny(u8, data, "\r\n");
    while (lines.next()) |line| {
        current.clearRetainingCapacity();
        var values = std.mem.tokenizeScalar(u8, line, ' ');
        var index: usize = 0;
        while (values.next()) |token| : (index += 1) {
            const value = try std.fmt.parseInt(u32, token, 10);
            const left = if (index > 0) previous.items[index - 1] else 0;
            const right = if (index < previous.items.len) previous.items[index] else 0;
            try current.append(allocator, value + @max(left, right));
        }
        std.mem.swap(std.ArrayListUnmanaged(u32), &previous, &current);
    }

    var best: u32 = 0;
    for (previous.items) |value| best = @max(best, value);
    return best;
}

pub fn solution(allocator: std.mem.Allocator) !u32 {
    return maxPathSum(allocator, triangle_file);
}

test "problem 067: python reference dataset" {
    try testing.expectEqual(@as(u32, 7273), try solution(testing.allocator));
}

test "problem 067: sample triangle and edge row" {
    const sample =
        \\3
        \\7 4
        \\2 4 6
        \\8 5 9 3
    ;
    try testing.expectEqual(@as(u32, 23), try maxPathSum(testing.allocator, sample));
    try testing.expectEqual(@as(u32, 59), try maxPathSum(testing.allocator, "59\n"));
}
