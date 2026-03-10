//! Project Euler Problem 102: Triangle Containment - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_102/sol1.py

const std = @import("std");
const testing = std.testing;

const triangles_file = @embedFile("problem_102_triangles.txt");
const triangles_test_file = @embedFile("problem_102_test_triangles.txt");

pub const Problem102Error = error{ InvalidLine, InvalidNumber };

fn vectorProduct(point1: [2]i32, point2: [2]i32) i64 {
    return @as(i64, point1[0]) * point2[1] - @as(i64, point1[1]) * point2[0];
}

fn containsOrigin(x1: i32, y1: i32, x2: i32, y2: i32, x3: i32, y3: i32) bool {
    const point_a = [2]i32{ x1, y1 };
    const point_a_to_b = [2]i32{ x2 - x1, y2 - y1 };
    const point_a_to_c = [2]i32{ x3 - x1, y3 - y1 };
    const denominator = vectorProduct(point_a_to_c, point_a_to_b);
    if (denominator == 0) return false;

    const alpha_num = -vectorProduct(point_a, point_a_to_b);
    const beta_num = vectorProduct(point_a, point_a_to_c);

    if (denominator > 0) {
        return alpha_num > 0 and beta_num > 0 and alpha_num + beta_num < denominator;
    }
    return alpha_num < 0 and beta_num < 0 and alpha_num + beta_num > denominator;
}

/// Returns the number of triangles whose interior contains the origin.
/// Time complexity: O(lines)
/// Space complexity: O(1)
pub fn solution(data: []const u8) Problem102Error!u32 {
    var ret: u32 = 0;
    var lines = std.mem.tokenizeAny(u8, data, "\r\n");
    while (lines.next()) |line| {
        var items = std.mem.tokenizeScalar(u8, line, ',');
        var coords: [6]i32 = undefined;
        var idx: usize = 0;
        while (items.next()) |item| : (idx += 1) {
            if (idx >= coords.len) return error.InvalidLine;
            coords[idx] = std.fmt.parseInt(i32, item, 10) catch return error.InvalidNumber;
        }
        if (idx != 6) return error.InvalidLine;
        if (containsOrigin(coords[0], coords[1], coords[2], coords[3], coords[4], coords[5])) ret += 1;
    }
    return ret;
}

test "problem 102: python reference" {
    try testing.expectEqual(@as(u32, 228), try solution(triangles_file));
}

test "problem 102: examples and invalid input" {
    try testing.expect(containsOrigin(-340, 495, -153, -910, 835, -947));
    try testing.expect(!containsOrigin(-175, 41, -421, -714, 574, -645));
    try testing.expectEqual(@as(u32, 1), try solution(triangles_test_file));
    try testing.expectError(error.InvalidLine, solution("1,2,3\n"));
}
