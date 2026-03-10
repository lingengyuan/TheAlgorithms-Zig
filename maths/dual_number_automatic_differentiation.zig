//! Dual Number Automatic Differentiation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/dual_number_automatic_differentiation.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

pub const DifferentiateError = error{
    InvalidOrder,
    OutOfMemory,
};

pub const Dual = struct {
    real: f64,
    duals: []f64,

    pub fn initSeed(allocator: Allocator, real: f64, rank: usize) DifferentiateError!Dual {
        const duals = try allocator.alloc(f64, rank);
        @memset(duals, 1.0);
        return .{ .real = real, .duals = duals };
    }

    pub fn initCoefficients(allocator: Allocator, real: f64, coeffs: []const f64) DifferentiateError!Dual {
        const duals = try allocator.alloc(f64, coeffs.len);
        @memcpy(duals, coeffs);
        return .{ .real = real, .duals = duals };
    }

    pub fn add(self: Dual, allocator: Allocator, other: Dual) DifferentiateError!Dual {
        const max_len = @max(self.duals.len, other.duals.len);
        const coeffs = try allocator.alloc(f64, max_len);
        for (0..max_len) |i| {
            const left = if (i < self.duals.len) self.duals[i] else 0.0;
            const right = if (i < other.duals.len) other.duals[i] else 0.0;
            coeffs[i] = left + right;
        }
        return .{ .real = self.real + other.real, .duals = coeffs };
    }

    pub fn addScalar(self: Dual, allocator: Allocator, value: f64) DifferentiateError!Dual {
        const coeffs = try allocator.alloc(f64, self.duals.len);
        @memcpy(coeffs, self.duals);
        return .{ .real = self.real + value, .duals = coeffs };
    }

    pub fn sub(self: Dual, allocator: Allocator, other: Dual) DifferentiateError!Dual {
        return self.add(allocator, try other.mulScalar(allocator, -1.0));
    }

    pub fn mul(self: Dual, allocator: Allocator, other: Dual) DifferentiateError!Dual {
        const coeffs = try allocator.alloc(f64, self.duals.len + other.duals.len + 1);
        @memset(coeffs, 0.0);

        for (self.duals, 0..) |left, i| {
            for (other.duals, 0..) |right, j| {
                coeffs[i + j + 1] += left * right;
            }
        }
        for (self.duals, 0..) |value, i| coeffs[i] += value * other.real;
        for (other.duals, 0..) |value, i| coeffs[i] += value * self.real;

        return .{ .real = self.real * other.real, .duals = coeffs };
    }

    pub fn mulScalar(self: Dual, allocator: Allocator, value: f64) DifferentiateError!Dual {
        const coeffs = try allocator.alloc(f64, self.duals.len);
        for (self.duals, 0..) |dual, i| coeffs[i] = dual * value;
        return .{ .real = self.real * value, .duals = coeffs };
    }

    pub fn divScalar(self: Dual, allocator: Allocator, value: f64) DifferentiateError!Dual {
        const coeffs = try allocator.alloc(f64, self.duals.len);
        for (self.duals, 0..) |dual, i| coeffs[i] = dual / value;
        return .{ .real = self.real / value, .duals = coeffs };
    }

    pub fn pow(self: Dual, allocator: Allocator, exponent: usize) DifferentiateError!Dual {
        if (exponent == 0) return initCoefficients(allocator, 1.0, &[_]f64{});
        if (exponent == 1) return initCoefficients(allocator, self.real, self.duals);

        var result = try initCoefficients(allocator, self.real, self.duals);
        var i: usize = 1;
        while (i < exponent) : (i += 1) {
            result = try result.mul(allocator, self);
        }
        return result;
    }
};

pub fn differentiate(
    allocator: Allocator,
    comptime func: fn (Allocator, Dual) DifferentiateError!Dual,
    position: f64,
    order: usize,
) DifferentiateError!f64 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const seed = try Dual.initSeed(alloc, position, 1);
    const result = try func(alloc, seed);
    if (order == 0) return result.real;
    if (order > result.duals.len) return 0.0;
    return result.duals[order - 1] * @as(f64, @floatFromInt(factorial(order)));
}

fn factorial(n: usize) usize {
    var result: usize = 1;
    for (1..n + 1) |value| result *= value;
    return result;
}

fn squareFunction(allocator: Allocator, x: Dual) DifferentiateError!Dual {
    return x.pow(allocator, 2);
}

fn sixthPowerFunction(allocator: Allocator, x: Dual) DifferentiateError!Dual {
    const square = try x.pow(allocator, 2);
    const fourth = try x.pow(allocator, 4);
    return square.mul(allocator, fourth);
}

fn shiftedSixthPowerFunction(allocator: Allocator, y: Dual) DifferentiateError!Dual {
    const shifted = try y.addScalar(allocator, 3.0);
    const power = try shifted.pow(allocator, 6);
    return power.mulScalar(allocator, 0.5);
}

test "dual autodiff: python reference examples" {
    try testing.expectEqual(@as(f64, 2.0), try differentiate(testing.allocator, squareFunction, 2.0, 2));
    try testing.expectEqual(@as(f64, 196830.0), try differentiate(testing.allocator, sixthPowerFunction, 9.0, 2));
    try testing.expectEqual(@as(f64, 7605.0), try differentiate(testing.allocator, shiftedSixthPowerFunction, 3.5, 4));
    try testing.expectEqual(@as(f64, 0.0), try differentiate(testing.allocator, squareFunction, 4.0, 3));
}

test "dual autodiff: order zero and extreme polynomial degree" {
    try testing.expectEqual(@as(f64, 16.0), try differentiate(testing.allocator, squareFunction, 4.0, 0));
    try testing.expectEqual(@as(f64, 360.0), try differentiate(testing.allocator, shiftedSixthPowerFunction, -3.0, 6));
}

test "dual autodiff: high derivative beyond polynomial degree is zero" {
    try testing.expectEqual(@as(f64, 0.0), try differentiate(testing.allocator, sixthPowerFunction, 2.0, 8));
}

test "dual autodiff: add treats missing coefficients as zero" {
    const alloc = testing.allocator;
    var lhs = try Dual.initCoefficients(alloc, 2.0, &[_]f64{ 1.0, 2.0 });
    defer alloc.free(lhs.duals);
    const rhs = try Dual.initCoefficients(alloc, 5.0, &[_]f64{3.0});
    defer alloc.free(rhs.duals);

    const sum = try lhs.add(alloc, rhs);
    defer alloc.free(sum.duals);

    try testing.expectEqual(@as(f64, 7.0), sum.real);
    try testing.expectEqualSlices(f64, &[_]f64{ 4.0, 2.0 }, sum.duals);
}
