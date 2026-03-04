//! Quadratic Equations (Complex Roots) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/quadratic_equations_complex_numbers.py

const std = @import("std");
const testing = std.testing;

pub const QuadraticError = error{InvalidCoefficientA};

pub const ComplexNumber = struct {
    real: f64,
    imag: f64,
};

pub const QuadraticRoots = struct {
    root_1: ComplexNumber,
    root_2: ComplexNumber,
};

/// Solves `ax^2 + bx + c = 0` and returns two complex roots.
/// Time complexity: O(1), Space complexity: O(1)
pub fn quadraticRoots(a: f64, b: f64, c: f64) QuadraticError!QuadraticRoots {
    if (a == 0.0) return QuadraticError.InvalidCoefficientA;

    const delta = b * b - 4.0 * a * c;
    const denominator = 2.0 * a;

    if (delta >= 0.0) {
        const sqrt_delta = @sqrt(delta);
        return .{
            .root_1 = .{ .real = (-b + sqrt_delta) / denominator, .imag = 0.0 },
            .root_2 = .{ .real = (-b - sqrt_delta) / denominator, .imag = 0.0 },
        };
    }

    const sqrt_abs_delta = @sqrt(-delta);
    const real_part = -b / denominator;
    const imag_part = sqrt_abs_delta / denominator;
    return .{
        .root_1 = .{ .real = real_part, .imag = imag_part },
        .root_2 = .{ .real = real_part, .imag = -imag_part },
    };
}

fn expectComplexApproxEqual(expected: ComplexNumber, actual: ComplexNumber, tol: f64) !void {
    try testing.expectApproxEqAbs(expected.real, actual.real, tol);
    try testing.expectApproxEqAbs(expected.imag, actual.imag, tol);
}

test "quadratic roots: python reference examples" {
    const roots_1 = try quadraticRoots(1.0, 3.0, -4.0);
    try expectComplexApproxEqual(.{ .real = 1.0, .imag = 0.0 }, roots_1.root_1, 1e-12);
    try expectComplexApproxEqual(.{ .real = -4.0, .imag = 0.0 }, roots_1.root_2, 1e-12);

    const roots_2 = try quadraticRoots(5.0, 6.0, 1.0);
    try expectComplexApproxEqual(.{ .real = -0.2, .imag = 0.0 }, roots_2.root_1, 1e-12);
    try expectComplexApproxEqual(.{ .real = -1.0, .imag = 0.0 }, roots_2.root_2, 1e-12);

    const roots_3 = try quadraticRoots(1.0, -6.0, 25.0);
    try expectComplexApproxEqual(.{ .real = 3.0, .imag = 4.0 }, roots_3.root_1, 1e-12);
    try expectComplexApproxEqual(.{ .real = 3.0, .imag = -4.0 }, roots_3.root_2, 1e-12);
}

test "quadratic roots: edge and extreme cases" {
    try testing.expectError(QuadraticError.InvalidCoefficientA, quadraticRoots(0.0, 1.0, 2.0));

    const repeated = try quadraticRoots(1.0, -2.0, 1.0);
    try expectComplexApproxEqual(.{ .real = 1.0, .imag = 0.0 }, repeated.root_1, 1e-12);
    try expectComplexApproxEqual(.{ .real = 1.0, .imag = 0.0 }, repeated.root_2, 1e-12);

    const near_large = try quadraticRoots(1e12, -3e12, 2e12);
    try expectComplexApproxEqual(.{ .real = 2.0, .imag = 0.0 }, near_large.root_1, 1e-9);
    try expectComplexApproxEqual(.{ .real = 1.0, .imag = 0.0 }, near_large.root_2, 1e-9);
}
