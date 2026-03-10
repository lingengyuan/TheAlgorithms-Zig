//! Radix-2 FFT Polynomial Multiplication - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/radix2_fft.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

pub const FftError = error{OutOfMemory};

pub const Complex = struct {
    real: f64,
    imag: f64,

    fn add(self: Complex, other: Complex) Complex {
        return .{ .real = self.real + other.real, .imag = self.imag + other.imag };
    }

    fn sub(self: Complex, other: Complex) Complex {
        return .{ .real = self.real - other.real, .imag = self.imag - other.imag };
    }

    fn mul(self: Complex, other: Complex) Complex {
        return .{
            .real = self.real * other.real - self.imag * other.imag,
            .imag = self.real * other.imag + self.imag * other.real,
        };
    }

    fn scale(self: Complex, factor: f64) Complex {
        return .{ .real = self.real * factor, .imag = self.imag * factor };
    }
};

/// Multiplies two polynomials with complex coefficients using iterative radix-2 FFT.
/// Caller owns the returned slice.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn multiplyPolynomials(allocator: Allocator, poly_a: []const Complex, poly_b: []const Complex) FftError![]Complex {
    const trimmed_a = trimLength(poly_a);
    const trimmed_b = trimLength(poly_b);

    const len_a = if (trimmed_a == 0) 1 else trimmed_a;
    const len_b = if (trimmed_b == 0) 1 else trimmed_b;
    const result_len = len_a + len_b - 1;
    const fft_len = nextPowerOfTwo(result_len);

    var fa = try allocator.alloc(Complex, fft_len);
    defer allocator.free(fa);
    var fb = try allocator.alloc(Complex, fft_len);
    defer allocator.free(fb);

    @memset(fa, .{ .real = 0.0, .imag = 0.0 });
    @memset(fb, .{ .real = 0.0, .imag = 0.0 });

    if (trimmed_a == 0) {
        fa[0] = .{ .real = 0.0, .imag = 0.0 };
    } else {
        for (poly_a[0..trimmed_a], 0..) |value, i| fa[i] = value;
    }
    if (trimmed_b == 0) {
        fb[0] = .{ .real = 0.0, .imag = 0.0 };
    } else {
        for (poly_b[0..trimmed_b], 0..) |value, i| fb[i] = value;
    }

    fft(fa, false);
    fft(fb, false);
    for (fa, fb, 0..) |left, right, i| {
        fa[i] = left.mul(right);
    }
    fft(fa, true);

    const rounded = try allocator.alloc(Complex, result_len);
    errdefer allocator.free(rounded);
    for (0..result_len) |i| rounded[i] = roundComplex(fa[i]);

    return trimProduct(allocator, rounded);
}

fn fft(values: []Complex, invert: bool) void {
    const n = values.len;

    var j: usize = 0;
    for (1..n) |i| {
        var bit = n >> 1;
        while (j & bit != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std.mem.swap(Complex, &values[i], &values[j]);
    }

    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        const direction: f64 = if (invert) -1.0 else 1.0;
        const angle = 2.0 * std.math.pi / @as(f64, @floatFromInt(len)) * direction;
        const wlen = Complex{ .real = @cos(angle), .imag = @sin(angle) };

        var start: usize = 0;
        while (start < n) : (start += len) {
            var w = Complex{ .real = 1.0, .imag = 0.0 };
            for (0..len / 2) |offset| {
                const u = values[start + offset];
                const v = values[start + offset + len / 2].mul(w);
                values[start + offset] = u.add(v);
                values[start + offset + len / 2] = u.sub(v);
                w = w.mul(wlen);
            }
        }
    }

    if (invert) {
        const scale = 1.0 / @as(f64, @floatFromInt(n));
        for (values, 0..) |value, i| values[i] = value.scale(scale);
    }
}

fn trimLength(poly: []const Complex) usize {
    var len = poly.len;
    while (len > 0 and isComplexZero(poly[len - 1])) : (len -= 1) {}
    return len;
}

fn trimProduct(allocator: Allocator, product: []Complex) FftError![]Complex {
    var len = product.len;
    while (len > 1 and isComplexZero(product[len - 1])) : (len -= 1) {}
    if (len == product.len) return product;

    const trimmed = try allocator.alloc(Complex, len);
    @memcpy(trimmed, product[0..len]);
    allocator.free(product);
    return trimmed;
}

fn roundComplex(value: Complex) Complex {
    return .{
        .real = normalizeSignedZero(round8(value.real)),
        .imag = normalizeSignedZero(round8(value.imag)),
    };
}

fn round8(value: f64) f64 {
    return @round(value * 100_000_000.0) / 100_000_000.0;
}

fn normalizeSignedZero(value: f64) f64 {
    if (@abs(value) < 1e-12) return 0.0;
    return value;
}

fn isComplexZero(value: Complex) bool {
    return @abs(value.real) < 1e-9 and @abs(value.imag) < 1e-9;
}

fn nextPowerOfTwo(value: usize) usize {
    var n: usize = 1;
    while (n < value) : (n <<= 1) {}
    return n;
}

fn naiveMultiply(allocator: Allocator, a: []const Complex, b: []const Complex) ![]Complex {
    const result = try allocator.alloc(Complex, a.len + b.len - 1);
    @memset(result, .{ .real = 0.0, .imag = 0.0 });
    for (a, 0..) |left, i| {
        for (b, 0..) |right, j| {
            result[i + j] = result[i + j].add(left.mul(right));
        }
    }
    return result;
}

fn expectComplexApproxEq(expected: Complex, actual: Complex, tolerance: f64) !void {
    try testing.expectApproxEqAbs(expected.real, actual.real, tolerance);
    try testing.expectApproxEqAbs(expected.imag, actual.imag, tolerance);
}

test "radix2 fft: python reference product example" {
    const alloc = testing.allocator;
    const a = [_]Complex{
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 1.0, .imag = 0.0 },
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 2.0, .imag = 0.0 },
    };
    const b = [_]Complex{
        .{ .real = 2.0, .imag = 0.0 },
        .{ .real = 3.0, .imag = 0.0 },
        .{ .real = 4.0, .imag = 0.0 },
        .{ .real = 0.0, .imag = 0.0 },
    };

    const product = try multiplyPolynomials(alloc, &a, &b);
    defer alloc.free(product);

    const expected = [_]Complex{
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 2.0, .imag = 0.0 },
        .{ .real = 3.0, .imag = 0.0 },
        .{ .real = 8.0, .imag = 0.0 },
        .{ .real = 6.0, .imag = 0.0 },
        .{ .real = 8.0, .imag = 0.0 },
    };
    try testing.expectEqual(expected.len, product.len);
    for (expected, product) |want, got| try expectComplexApproxEq(want, got, 1e-8);
}

test "radix2 fft: agrees with naive convolution on complex coefficients" {
    const alloc = testing.allocator;
    const a = [_]Complex{
        .{ .real = 1.0, .imag = 1.0 },
        .{ .real = -2.0, .imag = 0.5 },
        .{ .real = 3.0, .imag = -1.0 },
        .{ .real = 0.0, .imag = 2.0 },
    };
    const b = [_]Complex{
        .{ .real = 0.5, .imag = -1.5 },
        .{ .real = 4.0, .imag = 0.0 },
        .{ .real = -1.0, .imag = 3.0 },
    };

    const fft_product = try multiplyPolynomials(alloc, &a, &b);
    defer alloc.free(fft_product);
    const naive_product = try naiveMultiply(alloc, &a, &b);
    defer alloc.free(naive_product);

    try testing.expectEqual(naive_product.len, fft_product.len);
    for (naive_product, fft_product) |want, got| try expectComplexApproxEq(want, got, 1e-6);
}

test "radix2 fft: trims trailing zeros but preserves zero polynomial" {
    const alloc = testing.allocator;

    const zero_product = try multiplyPolynomials(alloc, &[_]Complex{}, &[_]Complex{});
    defer alloc.free(zero_product);
    try testing.expectEqual(@as(usize, 1), zero_product.len);
    try expectComplexApproxEq(.{ .real = 0.0, .imag = 0.0 }, zero_product[0], 1e-12);

    const a = [_]Complex{
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 1.0, .imag = 0.0 },
    };
    const b = [_]Complex{
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 2.0, .imag = 0.0 },
        .{ .real = 0.0, .imag = 0.0 },
    };
    const shifted = try multiplyPolynomials(alloc, &a, &b);
    defer alloc.free(shifted);
    const expected = [_]Complex{
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 0.0, .imag = 0.0 },
        .{ .real = 2.0, .imag = 0.0 },
    };
    try testing.expectEqual(expected.len, shifted.len);
    for (expected, shifted) |want, got| try expectComplexApproxEq(want, got, 1e-8);
}

test "radix2 fft: extreme power-of-two padding case remains accurate" {
    const alloc = testing.allocator;
    var a: [17]Complex = undefined;
    var b: [11]Complex = undefined;
    for (a, 0..) |_, i| {
        a[i] = .{
            .real = @as(f64, @floatFromInt(@as(i64, @intCast(i % 5)) - 2)),
            .imag = @as(f64, @floatFromInt(@as(i64, @intCast(i % 3)) - 1)),
        };
    }
    for (b, 0..) |_, i| {
        b[i] = .{
            .real = @as(f64, @floatFromInt(@as(i64, @intCast(i % 7)) - 3)),
            .imag = @as(f64, @floatFromInt(@as(i64, @intCast(i % 4)) - 2)),
        };
    }

    const fft_product = try multiplyPolynomials(alloc, &a, &b);
    defer alloc.free(fft_product);
    const naive_product = try naiveMultiply(alloc, &a, &b);
    defer alloc.free(naive_product);

    const trimmed_len = trimLength(naive_product);
    try testing.expectEqual(trimmed_len, fft_product.len);
    for (naive_product[0..trimmed_len], fft_product) |want, got| try expectComplexApproxEq(want, got, 1e-5);
}
