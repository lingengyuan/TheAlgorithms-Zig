//! Koch Snowflake Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/fractals/koch_snowflake.py

const std = @import("std");
const testing = std.testing;

pub const KochSnowflakeError = error{
    InvalidVectorCount,
};

pub const Vector2 = struct {
    x: f64,
    y: f64,
};

/// Rotates a 2D vector by angle degrees.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn rotate(vector: Vector2, angle_in_degrees: f64) Vector2 {
    const theta = angle_in_degrees * std.math.pi / 180.0;
    const c = @cos(theta);
    const s = @sin(theta);
    return .{
        .x = c * vector.x - s * vector.y,
        .y = s * vector.x + c * vector.y,
    };
}

/// Performs one Koch iteration step over polyline vectors.
/// Caller owns returned slice.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn iterationStep(allocator: std.mem.Allocator, vectors: []const Vector2) ![]Vector2 {
    if (vectors.len < 2) {
        return KochSnowflakeError.InvalidVectorCount;
    }

    const out_len = (vectors.len - 1) * 4 + 1;
    const out = try allocator.alloc(Vector2, out_len);

    var idx: usize = 0;
    for (0..vectors.len - 1) |i| {
        const start = vectors[i];
        const end = vectors[i + 1];
        const diff = Vector2{ .x = end.x - start.x, .y = end.y - start.y };
        const one_third = Vector2{ .x = diff.x / 3.0, .y = diff.y / 3.0 };
        const two_third = Vector2{ .x = diff.x * 2.0 / 3.0, .y = diff.y * 2.0 / 3.0 };

        out[idx] = start;
        idx += 1;
        out[idx] = .{ .x = start.x + one_third.x, .y = start.y + one_third.y };
        idx += 1;

        const tip = rotate(one_third, 60.0);
        out[idx] = .{ .x = start.x + one_third.x + tip.x, .y = start.y + one_third.y + tip.y };
        idx += 1;

        out[idx] = .{ .x = start.x + two_third.x, .y = start.y + two_third.y };
        idx += 1;
    }
    out[idx] = vectors[vectors.len - 1];

    return out;
}

/// Applies Koch iteration for `steps` rounds.
/// Caller owns returned slice.
///
/// Time complexity: O(m * 4^steps)
/// Space complexity: O(m * 4^steps)
pub fn iterate(allocator: std.mem.Allocator, initial_vectors: []const Vector2, steps: usize) ![]Vector2 {
    if (initial_vectors.len < 2) {
        return KochSnowflakeError.InvalidVectorCount;
    }

    var current = try allocator.alloc(Vector2, initial_vectors.len);
    std.mem.copyForwards(Vector2, current, initial_vectors);

    var s: usize = 0;
    while (s < steps) : (s += 1) {
        const next = try iterationStep(allocator, current);
        allocator.free(current);
        current = next;
    }

    return current;
}

test "koch snowflake: python rotate examples" {
    const r60 = rotate(.{ .x = 1, .y = 0 }, 60);
    try testing.expectApproxEqAbs(@as(f64, 0.5), r60.x, 1e-7);
    try testing.expectApproxEqAbs(@as(f64, 0.8660254), r60.y, 1e-7);

    const r90 = rotate(.{ .x = 1, .y = 0 }, 90);
    try testing.expectApproxEqAbs(@as(f64, 6.123234e-17), r90.x, 1e-15);
    try testing.expectApproxEqAbs(@as(f64, 1.0), r90.y, 1e-12);
}

test "koch snowflake: python iteration step example" {
    const alloc = testing.allocator;
    const line = [_]Vector2{ .{ .x = 0, .y = 0 }, .{ .x = 1, .y = 0 } };
    const out = try iterationStep(alloc, &line);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 5), out.len);
    try testing.expectApproxEqAbs(@as(f64, 0.0), out[0].x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), out[1].x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), out[2].x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.28867513), out[2].y, 1e-8);
    try testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), out[3].x, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), out[4].x, 1e-12);
}

test "koch snowflake: validation and extreme values" {
    const alloc = testing.allocator;

    try testing.expectError(KochSnowflakeError.InvalidVectorCount, iterationStep(alloc, &[_]Vector2{.{ .x = 0, .y = 0 }}));
    try testing.expectError(KochSnowflakeError.InvalidVectorCount, iterate(alloc, &[_]Vector2{}, 1));

    const line = [_]Vector2{ .{ .x = 0, .y = 0 }, .{ .x = 1, .y = 0 } };
    const out = try iterate(alloc, &line, 6);
    defer alloc.free(out);
    try testing.expectEqual(@as(usize, 4097), out.len);
    try testing.expect(std.math.isFinite(out[2048].x));
    try testing.expect(std.math.isFinite(out[2048].y));
}
