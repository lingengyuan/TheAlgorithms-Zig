//! Mandelbrot Set Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/fractals/mandelbrot.py

const std = @import("std");
const testing = std.testing;

pub const MandelbrotError = error{
    InvalidMaxStep,
};

fn toRgbTuple(r: f64, g: f64, b: f64) [3]u8 {
    return .{
        @as(u8, @intFromFloat(@round(r * 255.0))),
        @as(u8, @intFromFloat(@round(g * 255.0))),
        @as(u8, @intFromFloat(@round(b * 255.0))),
    };
}

fn hsvToRgb(h: f64, s: f64, v: f64) [3]u8 {
    if (s == 0.0) {
        return toRgbTuple(v, v, v);
    }

    const hh = h * 6.0;
    const sector_floor = @floor(hh);
    const i: i32 = @intFromFloat(sector_floor);
    const f = hh - sector_floor;
    const p = v * (1.0 - s);
    const q = v * (1.0 - s * f);
    const t = v * (1.0 - s * (1.0 - f));

    return switch (@mod(i, 6)) {
        0 => toRgbTuple(v, t, p),
        1 => toRgbTuple(q, v, p),
        2 => toRgbTuple(p, v, t),
        3 => toRgbTuple(p, q, v),
        4 => toRgbTuple(t, p, v),
        else => toRgbTuple(v, p, q),
    };
}

/// Returns relative divergence distance for Mandelbrot iteration.
/// Members of set return 1.0, diverging values return step/(max_step-1).
///
/// Time complexity: O(max_step)
/// Space complexity: O(1)
pub fn getDistance(x: f64, y: f64, max_step: usize) MandelbrotError!f64 {
    if (max_step < 2) {
        return MandelbrotError.InvalidMaxStep;
    }

    var a = x;
    var b = y;
    var step: usize = 0;

    while (step < max_step) : (step += 1) {
        const a_new = a * a - b * b + x;
        b = 2.0 * a * b + y;
        a = a_new;

        if (a * a + b * b > 4.0) {
            break;
        }
    }

    const effective_step = if (step == max_step) max_step - 1 else step;
    return @as(f64, @floatFromInt(effective_step)) / @as(f64, @floatFromInt(max_step - 1));
}

/// Black/white color coding (Mandelbrot set points black).
pub fn getBlackAndWhiteRgb(distance: f64) [3]u8 {
    if (distance == 1.0) {
        return .{ 0, 0, 0 };
    }
    return .{ 255, 255, 255 };
}

/// HSV color coding based on divergence distance (Mandelbrot set points black).
pub fn getColorCodedRgb(distance: f64) [3]u8 {
    if (distance == 1.0) {
        return .{ 0, 0, 0 };
    }
    return hsvToRgb(distance, 1.0, 1.0);
}

test "mandelbrot: python distance examples" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), try getDistance(0, 0, 50), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.061224489795918366), try getDistance(0.5, 0.5, 50), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try getDistance(2, 0, 50), 1e-12);
}

test "mandelbrot: python color examples" {
    try testing.expectEqualSlices(u8, &[_]u8{ 255, 255, 255 }, &getBlackAndWhiteRgb(0));
    try testing.expectEqualSlices(u8, &[_]u8{ 255, 255, 255 }, &getBlackAndWhiteRgb(0.5));
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, &getBlackAndWhiteRgb(1));

    try testing.expectEqualSlices(u8, &[_]u8{ 255, 0, 0 }, &getColorCodedRgb(0));
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 255, 255 }, &getColorCodedRgb(0.5));
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, &getColorCodedRgb(1));
}

test "mandelbrot: validation and extreme values" {
    try testing.expectError(MandelbrotError.InvalidMaxStep, getDistance(0, 0, 1));

    const d = try getDistance(-0.75, 0.1, 2_000);
    try testing.expect(std.math.isFinite(d));
    try testing.expect(d >= 0 and d <= 1);
}
