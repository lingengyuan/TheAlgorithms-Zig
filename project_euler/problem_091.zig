//! Project Euler Problem 91: Right Triangles with Integer Coordinates - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_091/sol1.py

const std = @import("std");
const testing = std.testing;

fn isRight(x1: u32, y1: u32, x2: u32, y2: u32) bool {
    if (x1 == 0 and y1 == 0) return false;
    if (x2 == 0 and y2 == 0) return false;

    const a_square = x1 * x1 + y1 * y1;
    const b_square = x2 * x2 + y2 * y2;
    const dx = @as(i32, @intCast(x1)) - @as(i32, @intCast(x2));
    const dy = @as(i32, @intCast(y1)) - @as(i32, @intCast(y2));
    const c_square = @as(u32, @intCast(dx * dx + dy * dy));
    return a_square + b_square == c_square or a_square + c_square == b_square or b_square + c_square == a_square;
}

/// Returns the number of right triangles OPQ with coordinates in `[0, limit]`.
/// Time complexity: O(limit^4)
/// Space complexity: O(1)
pub fn solution(limit: u32) u32 {
    var count: u32 = 0;
    var x1: u32 = 0;
    while (x1 <= limit) : (x1 += 1) {
        var y1: u32 = 0;
        while (y1 <= limit) : (y1 += 1) {
            var x2: u32 = x1;
            while (x2 <= limit) : (x2 += 1) {
                var y2: u32 = 0;
                while (y2 <= limit) : (y2 += 1) {
                    if (x1 == x2 and y2 <= y1) continue;
                    if (isRight(x1, y1, x2, y2)) count += 1;
                }
            }
        }
    }
    return count;
}

test "problem 091: python reference" {
    try testing.expectEqual(@as(u32, 14), solution(2));
    try testing.expectEqual(@as(u32, 448), solution(10));
    try testing.expectEqual(@as(u32, 14234), solution(50));
}

test "problem 091: helper semantics and degenerate cases" {
    try testing.expect(isRight(0, 1, 2, 0));
    try testing.expect(!isRight(1, 0, 2, 2));
    try testing.expect(!isRight(0, 0, 2, 0));
    try testing.expectEqual(@as(u32, 0), solution(0));
}
