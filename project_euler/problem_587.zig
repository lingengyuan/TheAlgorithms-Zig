//! Project Euler Problem 587: Concave Triangle - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_587/sol1.py

const std = @import("std");
const testing = std.testing;

fn circleBottomArcIntegral(point: f64) f64 {
    return ((1.0 - 2.0 * point) * std.math.sqrt(point - point * point) + 2.0 * point + std.math.asin(std.math.sqrt(1.0 - point))) / 4.0;
}

fn concaveTriangleArea(circles_number: u64) f64 {
    const n = @as(f64, @floatFromInt(circles_number));
    const intersection_y = (n + 1.0 - std.math.sqrt(2.0 * n)) / (2.0 * (n * n + 1.0));
    const intersection_x = n * intersection_y;

    const triangle_area = intersection_x * intersection_y / 2.0;
    const concave_region_area = circleBottomArcIntegral(0.5) - circleBottomArcIntegral(intersection_x);
    return triangle_area + concave_region_area;
}

/// Returns the least `n` for which the concave triangle occupies less than
/// `fraction` of the L-section.
/// Time complexity: O(answer)
/// Space complexity: O(1)
pub fn solution(fraction: f64) u64 {
    const l_section_area = (1.0 - std.math.pi / 4.0) / 4.0;

    var n: u64 = 1;
    while (true) : (n += 1) {
        if (concaveTriangleArea(n) / l_section_area < fraction) return n;
    }
}

test "problem 587: python reference" {
    try testing.expectEqual(@as(u64, 15), solution(0.1));
    try testing.expectEqual(@as(u64, 2240), solution(0.001));
}

test "problem 587: integral and area edge cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.39269908169872414), circleBottomArcIntegral(0.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.44634954084936207), circleBottomArcIntegral(0.5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), circleBottomArcIntegral(1.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.026825229575318944), concaveTriangleArea(1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.01956236140083944), concaveTriangleArea(2), 1e-12);
}
