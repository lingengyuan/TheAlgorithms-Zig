//! Simulated Annealing - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/simulated_annealing.py

const std = @import("std");
const testing = std.testing;
const hc = @import("hill_climbing.zig");

pub const SimulatedAnnealingOptions = struct {
    find_max: bool = true,
    bounds: hc.Bounds = .{},
    start_temperature: f64 = 100.0,
    rate_of_decrease: f64 = 0.01,
    threshold_temp: f64 = 1.0,
};

/// Performs simulated annealing using the Python reference acceptance logic.
/// Note: best-state tracking intentionally mirrors the Python implementation,
/// which always records the highest score seen so far.
/// Time complexity: O(iterations * neighbors), Space complexity: O(1)
pub fn simulatedAnnealing(search_prob: hc.SearchProblem, options: SimulatedAnnealingOptions, random: std.Random) hc.SearchProblem {
    var rng = random;
    var current_state = search_prob;
    var current_temp = options.start_temperature;
    var best_state: ?hc.SearchProblem = null;

    while (true) {
        const current_score = current_state.score();
        if (best_state == null or current_score > best_state.?.score()) {
            best_state = current_state;
        }

        var next_state: ?hc.SearchProblem = null;
        var neighbors = current_state.getNeighbors();
        var remaining = neighbors.len;

        while (next_state == null and remaining > 0) {
            const index = rng.uintLessThan(usize, remaining);
            const picked = neighbors[index];
            remaining -= 1;
            neighbors[index] = neighbors[remaining];
            neighbors[remaining] = picked;

            if (!options.bounds.contains(picked)) continue;

            var change = picked.score() - current_score;
            if (!options.find_max) change *= -1;

            if (change > 0) {
                next_state = picked;
            } else if (current_temp != 0.0) {
                const probability = std.math.exp(@as(f64, @floatFromInt(change)) / current_temp);
                if (rng.float(f64) < probability) next_state = picked;
            }
        }

        current_temp -= current_temp * options.rate_of_decrease;
        if (current_temp < options.threshold_temp or next_state == null) break;
        current_state = next_state.?;
    }

    return best_state orelse current_state;
}

fn paraboloid(x: i32, y: i32) i64 {
    return @as(i64, x) * x + @as(i64, y) * y;
}

fn linearPlane(x: i32, y: i32) i64 {
    return x + y;
}

test "simulated annealing: accepts improving moves to bounded maximum" {
    var prng = std.Random.DefaultPrng.init(0);
    const problem = hc.SearchProblem{ .x = 0, .y = 0, .step_size = 1, .function = linearPlane };
    const result = simulatedAnnealing(problem, .{
        .find_max = true,
        .bounds = .{ .max_x = 1, .min_x = 0, .max_y = 1, .min_y = 0 },
        .start_temperature = 10.0,
        .rate_of_decrease = 0.1,
        .threshold_temp = 0.1,
    }, prng.random());

    try testing.expectEqual(@as(i32, 1), result.x);
    try testing.expectEqual(@as(i32, 1), result.y);
    try testing.expectEqual(@as(i64, 2), result.score());
}

test "simulated annealing: threshold short-circuits before movement" {
    var prng = std.Random.DefaultPrng.init(1);
    const problem = hc.SearchProblem{ .x = 12, .y = 47, .step_size = 1, .function = paraboloid };
    const result = simulatedAnnealing(problem, .{
        .find_max = false,
        .bounds = .{ .max_x = 100, .min_x = 5, .max_y = 50, .min_y = -5 },
        .start_temperature = 0.5,
        .rate_of_decrease = 0.01,
        .threshold_temp = 1.0,
    }, prng.random());

    try testing.expectEqual(problem, result);
}

test "simulated annealing: extreme finite scores remain stable" {
    var prng = std.Random.DefaultPrng.init(2);
    const problem = hc.SearchProblem{ .x = 5, .y = 5, .step_size = 1, .function = paraboloid };
    const result = simulatedAnnealing(problem, .{
        .find_max = true,
        .bounds = .{ .max_x = 25, .min_x = -25, .max_y = 25, .min_y = -25 },
        .start_temperature = 100.0,
        .rate_of_decrease = 0.02,
        .threshold_temp = 0.5,
    }, prng.random());

    try testing.expect(std.math.isFinite(@as(f64, @floatFromInt(result.score()))));
    try testing.expect(result.x <= 25 and result.x >= -25);
    try testing.expect(result.y <= 25 and result.y >= -25);
}
