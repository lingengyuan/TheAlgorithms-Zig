//! Monte Carlo Dice Probability Estimation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/monte_carlo_dice.py

const std = @import("std");
const testing = std.testing;

pub const MonteCarloDiceError = error{
    InvalidInput,
    OutOfMemory,
};

pub const Dice = struct {
    pub const num_sides: u8 = 6;

    pub fn roll(random: std.Random) u8 {
        return random.uintLessThan(u8, num_sides) + 1;
    }
};

/// Returns the estimated percentage distribution of all possible sums when throwing
/// `num_dice` six-sided dice `num_throws` times.
/// Caller owns the returned slice.
/// Time complexity: O(num_throws * num_dice), Space complexity: O(num_dice)
pub fn throwDice(allocator: std.mem.Allocator, num_throws: usize, num_dice: usize, random: std.Random) MonteCarloDiceError![]f64 {
    if (num_throws == 0 or num_dice == 0) return error.InvalidInput;

    const bucket_count = num_dice * Dice.num_sides + 1;
    const counts = try allocator.alloc(usize, bucket_count);
    defer allocator.free(counts);
    @memset(counts, 0);

    const rng = random;
    for (0..num_throws) |_| {
        var sum: usize = 0;
        for (0..num_dice) |_| sum += Dice.roll(rng);
        counts[sum] += 1;
    }

    const probabilities = try allocator.alloc(f64, (Dice.num_sides - 1) * num_dice + 1);
    errdefer allocator.free(probabilities);
    for (0..probabilities.len) |i| {
        const count = counts[num_dice + i];
        probabilities[i] = round2(@as(f64, @floatFromInt(count)) * 100.0 / @as(f64, @floatFromInt(num_throws)));
    }
    return probabilities;
}

fn round2(value: f64) f64 {
    return @round(value * 100.0) / 100.0;
}

fn sumSlice(values: []const f64) f64 {
    var total: f64 = 0.0;
    for (values) |value| total += value;
    return total;
}

test "monte carlo dice: deterministic with same seed" {
    const alloc = testing.allocator;
    var prng_a = std.Random.DefaultPrng.init(0);
    var prng_b = std.Random.DefaultPrng.init(0);

    const first = try throwDice(alloc, 100, 1, prng_a.random());
    defer alloc.free(first);
    const second = try throwDice(alloc, 100, 1, prng_b.random());
    defer alloc.free(second);

    try testing.expectEqual(first.len, second.len);
    for (first, second) |left, right| try testing.expectEqual(left, right);
}

test "monte carlo dice: one die approaches uniform distribution" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const probabilities = try throwDice(alloc, 100_000, 1, prng.random());
    defer alloc.free(probabilities);

    try testing.expectEqual(@as(usize, 6), probabilities.len);
    for (probabilities) |probability| {
        try testing.expect(@abs(probability - 16.67) < 1.0);
    }
    try testing.expect(@abs(sumSlice(probabilities) - 100.0) < 0.05);
}

test "monte carlo dice: two dice distribution has expected shape" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(123);
    const probabilities = try throwDice(alloc, 200_000, 2, prng.random());
    defer alloc.free(probabilities);

    try testing.expectEqual(@as(usize, 11), probabilities.len);
    try testing.expect(probabilities[5] > probabilities[4]);
    try testing.expect(probabilities[5] > probabilities[6]);
    try testing.expect(@abs(probabilities[0] - probabilities[10]) < 0.5);
    try testing.expect(@abs(probabilities[1] - probabilities[9]) < 0.5);
    try testing.expect(@abs(sumSlice(probabilities) - 100.0) < 0.05);
}

test "monte carlo dice: invalid and extreme inputs" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    try testing.expectError(error.InvalidInput, throwDice(alloc, 0, 1, prng.random()));
    try testing.expectError(error.InvalidInput, throwDice(alloc, 10, 0, prng.random()));

    const probabilities = try throwDice(alloc, 1, 10, prng.random());
    defer alloc.free(probabilities);
    try testing.expectEqual(@as(usize, 51), probabilities.len);

    var non_zero_count: usize = 0;
    for (probabilities) |probability| {
        if (probability > 0.0) non_zero_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), non_zero_count);
    try testing.expect(@abs(sumSlice(probabilities) - 100.0) < 1e-9);
}
