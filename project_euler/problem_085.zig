//! Project Euler Problem 85: Counting Rectangles - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_085/sol1.py

const std = @import("std");
const testing = std.testing;

fn triangleNumber(n: usize) usize {
    return n * (n + 1) / 2;
}

/// Finds the area of the grid whose rectangle count is closest to `target`.
/// Time complexity: O(sqrt(target))
/// Space complexity: O(1)
pub fn solution(target: usize) usize {
    var best_product: usize = 0;
    var best_area: usize = 0;

    const max_side = @as(usize, @intFromFloat(@ceil(std.math.sqrt(@as(f64, @floatFromInt(target * 2))) * 1.1)));
    var a: usize = 1;
    while (a < max_side) : (a += 1) {
        const triangle_a = triangleNumber(a);
        const estimate = (-1.0 + std.math.sqrt(1.0 + 8.0 * @as(f64, @floatFromInt(target)) / @as(f64, @floatFromInt(triangle_a)))) / 2.0;
        const b_floor = @as(usize, @intFromFloat(@floor(estimate)));
        const b_ceil = @as(usize, @intFromFloat(@ceil(estimate)));

        const options = [_]usize{ b_floor, b_ceil };
        for (options) |b| {
            if (b == 0) continue;
            const product = triangle_a * triangleNumber(b);
            if (best_product == 0 or absDiff(target, product) < absDiff(target, best_product)) {
                best_product = product;
                best_area = a * b;
            }
        }
    }

    return best_area;
}

fn absDiff(a: usize, b: usize) usize {
    return if (a > b) a - b else b - a;
}

test "problem 085: python reference" {
    try testing.expectEqual(@as(usize, 6), solution(20));
    try testing.expectEqual(@as(usize, 72), solution(2000));
    try testing.expectEqual(@as(usize, 2772), solution(2_000_000));
    try testing.expectEqual(@as(usize, 86595), solution(2_000_000_000));
}

test "problem 085: tiny targets" {
    try testing.expectEqual(@as(usize, 1), solution(1));
    try testing.expectEqual(@as(usize, 1), solution(2));
}
