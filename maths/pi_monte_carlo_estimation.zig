//! Pi Monte Carlo Estimation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/pi_monte_carlo_estimation.py

const std = @import("std");
const testing = std.testing;

pub const PiEstimationError = error{InvalidSimulationCount};

pub const Point = struct {
    x: f64,
    y: f64,

    pub fn isInUnitCircle(self: Point) bool {
        return self.x * self.x + self.y * self.y <= 1.0;
    }

    pub fn randomUnitSquare(random: std.Random) Point {
        return .{
            .x = random.float(f64),
            .y = random.float(f64),
        };
    }
};

/// Estimates pi by sampling points uniformly in the unit square.
/// Time complexity: O(number_of_simulations), Space complexity: O(1)
pub fn estimatePi(number_of_simulations: usize, random: std.Random) PiEstimationError!f64 {
    if (number_of_simulations == 0) return error.InvalidSimulationCount;

    var in_circle: usize = 0;
    const rng = random;
    for (0..number_of_simulations) |_| {
        if (Point.randomUnitSquare(rng).isInUnitCircle()) in_circle += 1;
    }

    return 4.0 * @as(f64, @floatFromInt(in_circle)) / @as(f64, @floatFromInt(number_of_simulations));
}

test "pi monte carlo estimation: point inclusion basics" {
    try testing.expect((Point{ .x = 0.0, .y = 0.0 }).isInUnitCircle());
    try testing.expect((Point{ .x = 1.0, .y = 0.0 }).isInUnitCircle());
    try testing.expect(!(Point{ .x = 1.0, .y = 1.0 }).isInUnitCircle());
}

test "pi monte carlo estimation: deterministic with fixed seed" {
    var prng_a = std.Random.DefaultPrng.init(123);
    var prng_b = std.Random.DefaultPrng.init(123);
    try testing.expectEqual(
        try estimatePi(1000, prng_a.random()),
        try estimatePi(1000, prng_b.random()),
    );
}

test "pi monte carlo estimation: converges within tolerance on large sample" {
    var prng = std.Random.DefaultPrng.init(42);
    const estimate = try estimatePi(500_000, prng.random());
    try testing.expect(@abs(estimate - std.math.pi) < 0.02);
}

test "pi monte carlo estimation: invalid input and extreme minimal case" {
    var prng = std.Random.DefaultPrng.init(1);
    try testing.expectError(error.InvalidSimulationCount, estimatePi(0, prng.random()));

    var prng_single = std.Random.DefaultPrng.init(7);
    const estimate = try estimatePi(1, prng_single.random());
    try testing.expect(estimate == 0.0 or estimate == 4.0);
}
