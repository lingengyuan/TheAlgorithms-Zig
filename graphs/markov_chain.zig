//! Markov Chain Transition Simulation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/markov_chain.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Transition = struct {
    from: usize,
    to: usize,
    probability: f64,
};

/// Runs Markov transitions for `steps` iterations and returns visit counts per node.
/// Invalid start/node/probability values return errors.
/// If outgoing probabilities sum to < 1, unmatched mass keeps the walker on current node.
/// Time complexity: O(steps * N), Space complexity: O(N^2)
pub fn getTransitions(
    allocator: Allocator,
    node_count: usize,
    start: usize,
    transitions: []const Transition,
    steps: usize,
    random: std.Random,
) ![]usize {
    if (start >= node_count) return error.InvalidStart;

    const matrix = try allocator.alloc(f64, node_count * node_count);
    defer allocator.free(matrix);
    @memset(matrix, 0.0);

    for (transitions) |t| {
        if (t.from >= node_count or t.to >= node_count) return error.InvalidNode;
        if (t.probability < 0 or t.probability > 1) return error.InvalidProbability;
        matrix[t.from * node_count + t.to] = t.probability;
    }

    const visited = try allocator.alloc(usize, node_count);
    @memset(visited, 0);

    var node = start;
    for (0..steps) |_| {
        const threshold = random.float(f64);
        var cumulative: f64 = 0.0;
        var moved = false;

        for (0..node_count) |to| {
            cumulative += matrix[node * node_count + to];
            if (cumulative > threshold) {
                node = to;
                moved = true;
                break;
            }
        }

        if (!moved) {
            // Keep current node when transition probability mass is incomplete.
        }

        const sum = @addWithOverflow(visited[node], 1);
        if (sum[1] != 0) {
            allocator.free(visited);
            return error.Overflow;
        }
        visited[node] = sum[0];
    }

    return visited;
}

fn sumCounts(values: []const usize) usize {
    var total: usize = 0;
    for (values) |v| total += v;
    return total;
}

test "markov chain: python sample ordering" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    const transitions = [_]Transition{
        .{ .from = 0, .to = 0, .probability = 0.9 },
        .{ .from = 0, .to = 1, .probability = 0.075 },
        .{ .from = 0, .to = 2, .probability = 0.025 },
        .{ .from = 1, .to = 0, .probability = 0.15 },
        .{ .from = 1, .to = 1, .probability = 0.8 },
        .{ .from = 1, .to = 2, .probability = 0.05 },
        .{ .from = 2, .to = 0, .probability = 0.25 },
        .{ .from = 2, .to = 1, .probability = 0.25 },
        .{ .from = 2, .to = 2, .probability = 0.5 },
    };

    const visited = try getTransitions(alloc, 3, 0, &transitions, 5000, prng.random());
    defer alloc.free(visited);

    try testing.expect(visited[0] > visited[1]);
    try testing.expect(visited[1] > visited[2]);
    try testing.expectEqual(@as(usize, 5000), sumCounts(visited));
}

test "markov chain: absorbing state" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);

    const transitions = [_]Transition{
        .{ .from = 0, .to = 1, .probability = 1.0 },
        .{ .from = 1, .to = 1, .probability = 1.0 },
    };

    const visited = try getTransitions(alloc, 2, 0, &transitions, 100, prng.random());
    defer alloc.free(visited);

    try testing.expectEqual(@as(usize, 0), visited[0]);
    try testing.expectEqual(@as(usize, 100), visited[1]);
}

test "markov chain: invalid start" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);

    try testing.expectError(
        error.InvalidStart,
        getTransitions(alloc, 2, 3, &[_]Transition{}, 1, prng.random()),
    );
}

test "markov chain: invalid transition values" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);

    try testing.expectError(
        error.InvalidNode,
        getTransitions(
            alloc,
            2,
            0,
            &[_]Transition{.{ .from = 0, .to = 3, .probability = 1.0 }},
            1,
            prng.random(),
        ),
    );

    try testing.expectError(
        error.InvalidProbability,
        getTransitions(
            alloc,
            2,
            0,
            &[_]Transition{.{ .from = 0, .to = 1, .probability = 1.2 }},
            1,
            prng.random(),
        ),
    );
}

test "markov chain: extreme step count" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(123);

    const transitions = [_]Transition{
        .{ .from = 0, .to = 0, .probability = 0.5 },
        .{ .from = 0, .to = 1, .probability = 0.5 },
        .{ .from = 1, .to = 0, .probability = 0.5 },
        .{ .from = 1, .to = 1, .probability = 0.5 },
    };

    const visited = try getTransitions(alloc, 2, 0, &transitions, 50_000, prng.random());
    defer alloc.free(visited);

    try testing.expectEqual(@as(usize, 50_000), sumCounts(visited));
}
