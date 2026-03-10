//! Hill Climbing - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/hill_climbing.py

const std = @import("std");
const testing = std.testing;

pub const SearchFunction = *const fn (x: i32, y: i32) i64;

pub const SearchProblem = struct {
    x: i32,
    y: i32,
    step_size: i32,
    function: SearchFunction,

    pub fn score(self: SearchProblem) i64 {
        return self.function(self.x, self.y);
    }

    pub fn getNeighbors(self: SearchProblem) [8]SearchProblem {
        const step = self.step_size;
        return .{
            .{ .x = self.x - step, .y = self.y - step, .step_size = step, .function = self.function },
            .{ .x = self.x - step, .y = self.y, .step_size = step, .function = self.function },
            .{ .x = self.x - step, .y = self.y + step, .step_size = step, .function = self.function },
            .{ .x = self.x, .y = self.y - step, .step_size = step, .function = self.function },
            .{ .x = self.x, .y = self.y + step, .step_size = step, .function = self.function },
            .{ .x = self.x + step, .y = self.y - step, .step_size = step, .function = self.function },
            .{ .x = self.x + step, .y = self.y, .step_size = step, .function = self.function },
            .{ .x = self.x + step, .y = self.y + step, .step_size = step, .function = self.function },
        };
    }
};

pub const Bounds = struct {
    max_x: i32 = std.math.maxInt(i32),
    min_x: i32 = std.math.minInt(i32),
    max_y: i32 = std.math.maxInt(i32),
    min_y: i32 = std.math.minInt(i32),

    pub fn contains(self: Bounds, problem: SearchProblem) bool {
        return problem.x <= self.max_x and problem.x >= self.min_x and problem.y <= self.max_y and problem.y >= self.min_y;
    }
};

pub const HillClimbingOptions = struct {
    find_max: bool = true,
    bounds: Bounds = .{},
    max_iter: usize = 10_000,
};

const Position = struct {
    x: i32,
    y: i32,
};

/// Performs hill climbing with the same neighbor ordering as the Python reference.
/// Time complexity: O(max_iter), Space complexity: O(max_iter)
pub fn hillClimbing(search_prob: SearchProblem, options: HillClimbingOptions, allocator: std.mem.Allocator) !SearchProblem {
    var current_state = search_prob;
    var iterations: usize = 0;
    var visited = std.AutoHashMap(Position, void).init(allocator);
    defer visited.deinit();

    while (iterations < options.max_iter) {
        try visited.put(.{ .x = current_state.x, .y = current_state.y }, {});
        iterations += 1;

        const current_score = current_state.score();
        const neighbors = current_state.getNeighbors();

        var next_state: ?SearchProblem = null;
        var max_change: i64 = std.math.minInt(i64);
        var min_change: i64 = std.math.maxInt(i64);

        for (neighbors) |neighbor| {
            if (visited.contains(.{ .x = neighbor.x, .y = neighbor.y })) continue;
            if (!options.bounds.contains(neighbor)) continue;

            const change = neighbor.score() - current_score;
            if (options.find_max) {
                if (change > max_change and change > 0) {
                    max_change = change;
                    next_state = neighbor;
                }
            } else if (change < min_change and change < 0) {
                min_change = change;
                next_state = neighbor;
            }
        }

        if (next_state) |state| {
            current_state = state;
        } else {
            break;
        }
    }

    return current_state;
}

fn paraboloid(x: i32, y: i32) i64 {
    return @as(i64, x) * x + @as(i64, y) * y;
}

fn linearPlane(x: i32, y: i32) i64 {
    return x + y;
}

test "hill climbing: score and neighbors" {
    const problem = SearchProblem{ .x = 0, .y = 0, .step_size = 1, .function = linearPlane };
    try testing.expectEqual(@as(i64, 0), problem.score());

    const neighbors = problem.getNeighbors();
    try testing.expectEqual(SearchProblem{ .x = -1, .y = -1, .step_size = 1, .function = linearPlane }, neighbors[0]);
    try testing.expectEqual(SearchProblem{ .x = 1, .y = 1, .step_size = 1, .function = linearPlane }, neighbors[7]);
}

test "hill climbing: finds local minimum" {
    const problem = SearchProblem{ .x = 3, .y = 4, .step_size = 1, .function = paraboloid };
    const result = try hillClimbing(problem, .{ .find_max = false }, testing.allocator);
    try testing.expectEqual(@as(i32, 0), result.x);
    try testing.expectEqual(@as(i32, 0), result.y);
    try testing.expectEqual(@as(i64, 0), result.score());
}

test "hill climbing: finds bounded maximum" {
    const problem = SearchProblem{ .x = 0, .y = 0, .step_size = 1, .function = paraboloid };
    const result = try hillClimbing(problem, .{
        .find_max = true,
        .bounds = .{ .max_x = 5, .min_x = -5, .max_y = 5, .min_y = -5 },
    }, testing.allocator);
    try testing.expect(@abs(result.x) == 5);
    try testing.expect(@abs(result.y) == 5);
    try testing.expectEqual(@as(i64, 50), result.score());
}

test "hill climbing: respects bounds and degenerate iteration limits" {
    const bounded_problem = SearchProblem{ .x = 12, .y = 47, .step_size = 1, .function = paraboloid };
    const bounded = try hillClimbing(bounded_problem, .{
        .find_max = false,
        .bounds = .{ .max_x = 100, .min_x = 5, .max_y = 50, .min_y = -5 },
    }, testing.allocator);
    try testing.expectEqual(@as(i32, 5), bounded.x);
    try testing.expectEqual(@as(i32, 0), bounded.y);
    try testing.expectEqual(@as(i64, 25), bounded.score());

    const stuck_problem = SearchProblem{ .x = 7, .y = -9, .step_size = 0, .function = linearPlane };
    const stuck = try hillClimbing(stuck_problem, .{ .max_iter = 0 }, testing.allocator);
    try testing.expectEqual(stuck_problem, stuck);
}
