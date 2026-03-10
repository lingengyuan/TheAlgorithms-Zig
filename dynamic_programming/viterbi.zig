//! Viterbi - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/viterbi.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const ViterbiError = error{
    EmptyParameter,
    DimensionMismatch,
    UnknownObservation,
} || Allocator.Error;

pub const HiddenMarkovModel = struct {
    states: []const []const u8,
    symbols: []const []const u8,
    initial_probabilities: []const f64,
    transition_probabilities: []const []const f64,
    emission_probabilities: []const []const f64,
};

/// Returns the most likely hidden-state path for the observed symbols.
/// Time complexity: O(T * S^2), Space complexity: O(T * S)
pub fn viterbi(
    allocator: Allocator,
    observations: []const []const u8,
    model: HiddenMarkovModel,
) ViterbiError![][]const u8 {
    try validateModel(observations, model);

    const state_count = model.states.len;
    const time_steps = observations.len;

    var previous = try allocator.alloc(f64, state_count);
    defer allocator.free(previous);
    var current = try allocator.alloc(f64, state_count);
    defer allocator.free(current);
    const backpointers = try allocator.alloc(usize, time_steps * state_count);
    defer allocator.free(backpointers);

    const first_observation = try symbolIndex(model.symbols, observations[0]);
    for (0..state_count) |state_index| {
        previous[state_index] = model.initial_probabilities[state_index] * model.emission_probabilities[state_index][first_observation];
        backpointers[state_index] = 0;
    }

    for (1..time_steps) |time_index| {
        const observation_index = try symbolIndex(model.symbols, observations[time_index]);
        for (0..state_count) |state_index| {
            var arg_max: usize = 0;
            var max_probability: f64 = -1.0;
            for (0..state_count) |prior_index| {
                const probability = previous[prior_index] *
                    model.transition_probabilities[prior_index][state_index] *
                    model.emission_probabilities[state_index][observation_index];
                if (probability > max_probability) {
                    max_probability = probability;
                    arg_max = prior_index;
                }
            }
            current[state_index] = max_probability;
            backpointers[time_index * state_count + state_index] = arg_max;
        }
        @memcpy(previous, current);
    }

    var last_state: usize = 0;
    var best_probability = previous[0];
    for (1..state_count) |state_index| {
        if (previous[state_index] > best_probability) {
            best_probability = previous[state_index];
            last_state = state_index;
        }
    }

    const path = try allocator.alloc([]const u8, time_steps);
    errdefer allocator.free(path);

    var current_state = last_state;
    var time_index = time_steps;
    while (time_index > 0) {
        time_index -= 1;
        path[time_index] = model.states[current_state];
        if (time_index > 0) current_state = backpointers[time_index * state_count + current_state];
    }

    return path;
}

fn validateModel(observations: []const []const u8, model: HiddenMarkovModel) ViterbiError!void {
    if (observations.len == 0 or
        model.states.len == 0 or
        model.symbols.len == 0 or
        model.initial_probabilities.len == 0 or
        model.transition_probabilities.len == 0 or
        model.emission_probabilities.len == 0)
    {
        return ViterbiError.EmptyParameter;
    }

    const state_count = model.states.len;
    const symbol_count = model.symbols.len;

    if (model.initial_probabilities.len != state_count or
        model.transition_probabilities.len != state_count or
        model.emission_probabilities.len != state_count)
    {
        return ViterbiError.DimensionMismatch;
    }

    for (model.transition_probabilities) |row| {
        if (row.len != state_count) return ViterbiError.DimensionMismatch;
    }
    for (model.emission_probabilities) |row| {
        if (row.len != symbol_count) return ViterbiError.DimensionMismatch;
    }

    for (observations) |observation| _ = try symbolIndex(model.symbols, observation);
}

fn symbolIndex(symbols: []const []const u8, target: []const u8) ViterbiError!usize {
    for (symbols, 0..) |symbol, index| {
        if (std.mem.eql(u8, symbol, target)) return index;
    }
    return ViterbiError.UnknownObservation;
}

pub fn freePath(allocator: Allocator, path: []const []const u8) void {
    allocator.free(path);
}

test "viterbi: wikipedia example" {
    const states = [_][]const u8{ "Healthy", "Fever" };
    const symbols = [_][]const u8{ "normal", "cold", "dizzy" };
    const initial = [_]f64{ 0.6, 0.4 };
    const transition_healthy = [_]f64{ 0.7, 0.3 };
    const transition_fever = [_]f64{ 0.4, 0.6 };
    const transition = [_][]const f64{ &transition_healthy, &transition_fever };
    const emission_healthy = [_]f64{ 0.5, 0.4, 0.1 };
    const emission_fever = [_]f64{ 0.1, 0.3, 0.6 };
    const emission = [_][]const f64{ &emission_healthy, &emission_fever };
    const observations = [_][]const u8{ "normal", "cold", "dizzy" };

    const path = try viterbi(testing.allocator, &observations, .{
        .states = &states,
        .symbols = &symbols,
        .initial_probabilities = &initial,
        .transition_probabilities = &transition,
        .emission_probabilities = &emission,
    });
    defer freePath(testing.allocator, path);

    try testing.expectEqualSlices([]const u8, &[_][]const u8{ "Healthy", "Healthy", "Fever" }, path);
}

test "viterbi: empty observations" {
    const states = [_][]const u8{"Healthy"};
    const symbols = [_][]const u8{"normal"};
    const initial = [_]f64{1.0};
    const transition_row = [_]f64{1.0};
    const transition = [_][]const f64{&transition_row};
    const emission_row = [_]f64{1.0};
    const emission = [_][]const f64{&emission_row};
    const observations = [_][]const u8{};

    try testing.expectError(ViterbiError.EmptyParameter, viterbi(testing.allocator, &observations, .{
        .states = &states,
        .symbols = &symbols,
        .initial_probabilities = &initial,
        .transition_probabilities = &transition,
        .emission_probabilities = &emission,
    }));
}

test "viterbi: dimension mismatch" {
    const states = [_][]const u8{ "Healthy", "Fever" };
    const symbols = [_][]const u8{ "normal", "cold" };
    const initial = [_]f64{ 0.6, 0.4 };
    const transition_healthy = [_]f64{0.7};
    const transition_fever = [_]f64{ 0.4, 0.6 };
    const transition = [_][]const f64{ &transition_healthy, &transition_fever };
    const emission_healthy = [_]f64{ 0.5, 0.5 };
    const emission_fever = [_]f64{ 0.4, 0.6 };
    const emission = [_][]const f64{ &emission_healthy, &emission_fever };
    const observations = [_][]const u8{"normal"};

    try testing.expectError(ViterbiError.DimensionMismatch, viterbi(testing.allocator, &observations, .{
        .states = &states,
        .symbols = &symbols,
        .initial_probabilities = &initial,
        .transition_probabilities = &transition,
        .emission_probabilities = &emission,
    }));
}

test "viterbi: extreme single-state model" {
    const states = [_][]const u8{"Rainy"};
    const symbols = [_][]const u8{"walk"};
    const initial = [_]f64{1.0};
    const transition_row = [_]f64{1.0};
    const transition = [_][]const f64{&transition_row};
    const emission_row = [_]f64{1.0};
    const emission = [_][]const f64{&emission_row};
    const observations = [_][]const u8{ "walk", "walk", "walk", "walk" };

    const path = try viterbi(testing.allocator, &observations, .{
        .states = &states,
        .symbols = &symbols,
        .initial_probabilities = &initial,
        .transition_probabilities = &transition,
        .emission_probabilities = &emission,
    });
    defer freePath(testing.allocator, path);

    try testing.expectEqualSlices([]const u8, &[_][]const u8{ "Rainy", "Rainy", "Rainy", "Rainy" }, path);
}
