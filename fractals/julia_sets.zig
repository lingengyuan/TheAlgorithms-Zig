//! Julia Sets Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/fractals/julia_sets.py

const std = @import("std");
const testing = std.testing;
const Complex = std.math.Complex(f64);

pub const JuliaSetsError = error{
    InvalidPixelCount,
};

/// Evaluates exp(z) + c.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn evalExponential(c_parameter: Complex, z_value: Complex) Complex {
    return std.math.complex.exp(z_value).add(c_parameter);
}

/// Evaluates z^2 + c.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn evalQuadraticPolynomial(c_parameter: Complex, z_value: Complex) Complex {
    return z_value.mul(z_value).add(c_parameter);
}

/// Creates square grid of complex values with real/imaginary ranges from
/// `-window_size` to `window_size` inclusive. Data is row-major, where row index
/// corresponds to real-axis sample and column index to imaginary-axis sample.
/// Caller owns returned buffer.
///
/// Time complexity: O(nb_pixels^2)
/// Space complexity: O(nb_pixels^2)
pub fn prepareGrid(allocator: std.mem.Allocator, window_size: f64, nb_pixels: usize) ![]Complex {
    if (nb_pixels == 0) return JuliaSetsError.InvalidPixelCount;

    const grid = try allocator.alloc(Complex, nb_pixels * nb_pixels);
    errdefer allocator.free(grid);

    const denom = if (nb_pixels > 1) @as(f64, @floatFromInt(nb_pixels - 1)) else 1.0;

    for (0..nb_pixels) |r| {
        const real = -window_size + 2.0 * window_size * @as(f64, @floatFromInt(r)) / denom;
        for (0..nb_pixels) |c| {
            const imag = -window_size + 2.0 * window_size * @as(f64, @floatFromInt(c)) / denom;
            grid[r * nb_pixels + c] = Complex.init(real, imag);
        }
    }

    return grid;
}

fn sanitize(value: Complex, infinity: ?f64) Complex {
    if (infinity) |limit| {
        if (!std.math.isFinite(value.re) or !std.math.isFinite(value.im)) {
            return Complex.init(limit, 0.0);
        }
    }
    return value;
}

/// Iterates function over each initial point for fixed number of iterations.
/// Caller owns returned buffer.
///
/// Time complexity: O(nb_iterations * n)
/// Space complexity: O(n)
pub fn iterateFunction(
    comptime eval_function: fn (Complex, Complex) Complex,
    allocator: std.mem.Allocator,
    function_params: Complex,
    nb_iterations: usize,
    z_0: []const Complex,
    infinity: ?f64,
) ![]Complex {
    const z_n = try allocator.alloc(Complex, z_0.len);
    errdefer allocator.free(z_n);
    @memcpy(z_n, z_0);

    for (0..nb_iterations) |_| {
        for (z_n) |*z| {
            z.* = sanitize(eval_function(function_params, z.*), infinity);
        }
    }

    return z_n;
}

test "julia sets: python eval doctests" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), evalExponential(Complex.init(0, 0), Complex.init(0, 0)).re, 1e-12);

    const exp_pi = evalExponential(Complex.init(1, 0), Complex.init(0, std.math.pi));
    try testing.expectApproxEqAbs(@as(f64, 0.0), exp_pi.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), exp_pi.im, 1e-12);

    const exp_i = evalExponential(Complex.init(0, 1), Complex.init(0, 0));
    try testing.expectApproxEqAbs(@as(f64, 1.0), exp_i.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), exp_i.im, 1e-12);

    const q1 = evalQuadraticPolynomial(Complex.init(0, 0), Complex.init(2, 0));
    try testing.expectApproxEqAbs(@as(f64, 4.0), q1.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), q1.im, 1e-12);

    const q2 = evalQuadraticPolynomial(Complex.init(-1, 0), Complex.init(1, 0));
    try testing.expectApproxEqAbs(@as(f64, 0.0), q2.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), q2.im, 1e-12);

    const q3 = evalQuadraticPolynomial(Complex.init(0, 1), Complex.init(0, 0));
    try testing.expectApproxEqAbs(@as(f64, 0.0), q3.re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), q3.im, 1e-12);
}

test "julia sets: python grid and iterate examples" {
    const alloc = testing.allocator;

    const grid = try prepareGrid(alloc, 1.0, 3);
    defer alloc.free(grid);

    const expected = [_]Complex{
        Complex.init(-1, -1),
        Complex.init(-1, 0),
        Complex.init(-1, 1),
        Complex.init(0, -1),
        Complex.init(0, 0),
        Complex.init(0, 1),
        Complex.init(1, -1),
        Complex.init(1, 0),
        Complex.init(1, 1),
    };

    for (expected, 0..) |v, i| {
        try testing.expectApproxEqAbs(v.re, grid[i].re, 1e-12);
        try testing.expectApproxEqAbs(v.im, grid[i].im, 1e-12);
    }

    const z0 = [_]Complex{ Complex.init(0, 0), Complex.init(1, 0), Complex.init(2, 0) };
    const out = try iterateFunction(evalQuadraticPolynomial, alloc, Complex.init(0, 0), 3, &z0, null);
    defer alloc.free(out);

    try testing.expectApproxEqAbs(@as(f64, 0.0), out[0].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), out[1].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 256.0), out[2].re, 1e-9);
}

test "julia sets: boundary and extreme behavior" {
    const alloc = testing.allocator;

    try testing.expectError(JuliaSetsError.InvalidPixelCount, prepareGrid(alloc, 2.0, 0));

    const huge = [_]Complex{ Complex.init(std.math.inf(f64), 0), Complex.init(0, 0) };
    const out_inf = try iterateFunction(evalQuadraticPolynomial, alloc, Complex.init(0, 0), 1, &huge, 12_345.0);
    defer alloc.free(out_inf);

    try testing.expectApproxEqAbs(@as(f64, 12_345.0), out_inf[0].re, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), out_inf[0].im, 1e-12);

    const dense = try prepareGrid(alloc, 2.0, 128);
    defer alloc.free(dense);

    const out = try iterateFunction(evalQuadraticPolynomial, alloc, Complex.init(0.25, 0.0), 32, dense, 1.0e10);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 128 * 128), out.len);
    for (out) |z| {
        try testing.expect(std.math.isFinite(z.re));
        try testing.expect(std.math.isFinite(z.im));
    }
}
