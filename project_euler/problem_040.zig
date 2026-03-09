//! Project Euler Problem 40: Champernowne's Constant - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_040/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem040Error = error{
    InvalidPosition,
    OutOfMemory,
};

/// Multiplies the digits at the given 1-based Champernowne positions.
///
/// Time complexity: O(max_position)
/// Space complexity: O(max_position)
pub fn digitProduct(positions: []const usize, allocator: std.mem.Allocator) Problem040Error!u64 {
    if (positions.len == 0) return 1;

    var max_position: usize = 0;
    for (positions) |position| {
        if (position == 0) return error.InvalidPosition;
        max_position = @max(max_position, position);
    }

    var digits = std.ArrayListUnmanaged(u8){};
    defer digits.deinit(allocator);

    var value: usize = 1;
    var buffer: [32]u8 = undefined;
    while (digits.items.len < max_position) : (value += 1) {
        const chunk = std.fmt.bufPrint(&buffer, "{}", .{value}) catch unreachable;
        try digits.appendSlice(allocator, chunk);
    }

    var product: u64 = 1;
    for (positions) |position| {
        product *= digits.items[position - 1] - '0';
    }
    return product;
}

/// Returns d1 * d10 * d100 * d1000 * d10000 * d100000 * d1000000.
pub fn solution(allocator: std.mem.Allocator) Problem040Error!u64 {
    return digitProduct(&[_]usize{ 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000 }, allocator);
}

test "problem 040: python reference" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(u64, 210), try solution(allocator));
}

test "problem 040: helper positions and extremes" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(u64, 2), try digitProduct(&[_]usize{ 1, 10, 12, 15 }, allocator));
    try testing.expectEqual(@as(u64, 1), try digitProduct(&[_]usize{1}, allocator));
    try testing.expectEqual(@as(u64, 1), try digitProduct(&[_]usize{}, allocator));
    try testing.expectError(error.InvalidPosition, digitProduct(&[_]usize{0}, allocator));
}
