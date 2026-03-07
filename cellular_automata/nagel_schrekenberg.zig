//! Nagel-Schreckenberg Traffic Model - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/cellular_automata/nagel_schrekenberg.py

const std = @import("std");
const testing = std.testing;

pub const NagelSchreckenbergError = error{
    InvalidCellCount,
    InvalidFrequency,
    InvalidMaxSpeed,
    InvalidProbability,
    InvalidCarIndex,
    MissingRandomSource,
};

const HighwayError = std.mem.Allocator.Error || NagelSchreckenbergError;

fn cloneRow(allocator: std.mem.Allocator, row: []const i32) ![]i32 {
    const copy = try allocator.alloc(i32, row.len);
    @memcpy(copy, row);
    return copy;
}

fn shouldSlowDown(probability: f64, random: ?*std.Random) NagelSchreckenbergError!bool {
    if (probability <= 0.0) return false;
    if (probability >= 1.0) return true;

    const rng = random orelse return NagelSchreckenbergError.MissingRandomSource;
    return rng.float(f64) < probability;
}

/// Builds initial highway state with fixed spacing and speed.
/// Caller owns returned slice.
///
/// Time complexity: O(number_of_cells / frequency)
/// Space complexity: O(number_of_cells)
pub fn constructHighway(
    allocator: std.mem.Allocator,
    number_of_cells: usize,
    frequency: usize,
    initial_speed: i32,
) HighwayError![]i32 {
    if (number_of_cells == 0) return NagelSchreckenbergError.InvalidCellCount;
    if (frequency == 0) return NagelSchreckenbergError.InvalidFrequency;

    const highway = try allocator.alloc(i32, number_of_cells);
    @memset(highway, -1);

    const start_speed = @max(initial_speed, 0);
    var i: usize = 0;
    while (i < number_of_cells) : (i += frequency) {
        highway[i] = start_speed;
    }

    return highway;
}

/// Gets number of empty cells between a car and the next car in circular highway.
///
/// Time complexity: O(number_of_cells)
/// Space complexity: O(number_of_cells) recursion depth in worst case
pub fn getDistance(highway_now: []const i32, car_index: isize) NagelSchreckenbergError!usize {
    if (highway_now.len == 0) return NagelSchreckenbergError.InvalidCellCount;
    if (car_index < -1 or car_index >= @as(isize, @intCast(highway_now.len))) {
        return NagelSchreckenbergError.InvalidCarIndex;
    }

    var distance: usize = 0;
    var index: usize = @intCast(car_index + 1);

    while (index < highway_now.len) : (index += 1) {
        if (highway_now[index] != -1) {
            return distance;
        }
        distance += 1;
    }

    if (car_index == -1) {
        return distance;
    }

    return distance + try getDistance(highway_now, -1);
}

/// Updates vehicle speeds before movement step.
/// Caller owns returned slice.
///
/// Time complexity: O(number_of_cells^2)
/// Space complexity: O(number_of_cells)
pub fn update(
    allocator: std.mem.Allocator,
    highway_now: []const i32,
    probability: f64,
    max_speed: i32,
    random: ?*std.Random,
) HighwayError![]i32 {
    if (probability < 0.0 or probability > 1.0) {
        return NagelSchreckenbergError.InvalidProbability;
    }
    if (max_speed < 0) {
        return NagelSchreckenbergError.InvalidMaxSpeed;
    }
    if (probability > 0.0 and probability < 1.0 and random == null) {
        return NagelSchreckenbergError.MissingRandomSource;
    }

    const next_highway = try allocator.alloc(i32, highway_now.len);
    @memset(next_highway, -1);

    for (highway_now, 0..) |speed, car_index| {
        if (speed == -1) continue;

        const accelerated = @min(speed + 1, max_speed);
        const distance = try getDistance(highway_now, @intCast(car_index));
        const dn = @as(i32, @intCast(distance)) - 1;

        var next_speed = @min(accelerated, dn);
        if (try shouldSlowDown(probability, random)) {
            next_speed = @max(next_speed - 1, 0);
        }

        next_highway[car_index] = next_speed;
    }

    return next_highway;
}

/// Simulates highway evolution for a number of update steps.
/// Caller owns each row and the outer slice; use `deinitHistory` to free.
///
/// Time complexity: O(number_of_update * number_of_cells^2)
/// Space complexity: O(number_of_update * number_of_cells)
pub fn simulate(
    allocator: std.mem.Allocator,
    initial_highway: []const i32,
    number_of_update: usize,
    probability: f64,
    max_speed: i32,
    random: ?*std.Random,
) HighwayError![][]i32 {
    if (probability < 0.0 or probability > 1.0) {
        return NagelSchreckenbergError.InvalidProbability;
    }
    if (max_speed < 0) {
        return NagelSchreckenbergError.InvalidMaxSpeed;
    }

    var history = std.ArrayListUnmanaged([]i32){};
    errdefer {
        for (history.items) |row| allocator.free(row);
        history.deinit(allocator);
    }

    try history.append(allocator, try cloneRow(allocator, initial_highway));

    const number_of_cells = initial_highway.len;
    for (0..number_of_update) |step| {
        const next_speeds = try update(allocator, history.items[step], probability, max_speed, random);
        defer allocator.free(next_speeds);

        const real_next = try allocator.alloc(i32, number_of_cells);
        @memset(real_next, -1);

        for (next_speeds, 0..) |speed, car_index| {
            if (speed == -1) continue;
            const target = (car_index + @as(usize, @intCast(speed))) % number_of_cells;
            real_next[target] = speed;
        }

        try history.append(allocator, real_next);
    }

    return history.toOwnedSlice(allocator);
}

pub fn deinitHistory(allocator: std.mem.Allocator, history: [][]i32) void {
    for (history) |row| allocator.free(row);
    allocator.free(history);
}

fn expectHistoryEqual(expected: []const []const i32, actual: [][]i32) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, 0..) |row, i| {
        try testing.expectEqualSlices(i32, row, actual[i]);
    }
}

test "nagel schreckenberg: python construct/get_distance/update examples" {
    const alloc = testing.allocator;

    const h1 = try constructHighway(alloc, 10, 2, 6);
    defer alloc.free(h1);
    try testing.expectEqualSlices(i32, &[_]i32{ 6, -1, 6, -1, 6, -1, 6, -1, 6, -1 }, h1);

    const h2 = try constructHighway(alloc, 10, 10, 2);
    defer alloc.free(h2);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, h2);

    try testing.expectEqual(@as(usize, 1), try getDistance(&[_]i32{ 6, -1, 6, -1, 6 }, 2));
    try testing.expectEqual(@as(usize, 3), try getDistance(&[_]i32{ 2, -1, -1, -1, 3, 1, 0, 1, 3, 2 }, 0));
    try testing.expectEqual(@as(usize, 4), try getDistance(&[_]i32{ -1, -1, -1, -1, 2, -1, -1, -1, 3 }, -1));

    const updated_1 = try update(alloc, &[_]i32{ -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, 3 }, 0.0, 5, null);
    defer alloc.free(updated_1);
    try testing.expectEqualSlices(i32, &[_]i32{ -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, 4 }, updated_1);

    const updated_2 = try update(alloc, &[_]i32{ -1, -1, 2, -1, -1, -1, -1, 3 }, 0.0, 5, null);
    defer alloc.free(updated_2);
    try testing.expectEqualSlices(i32, &[_]i32{ -1, -1, 3, -1, -1, -1, -1, 1 }, updated_2);
}

test "nagel schreckenberg: python simulate examples" {
    const alloc = testing.allocator;

    const sim1 = try simulate(alloc, &[_]i32{ -1, 2, -1, -1, -1, 3 }, 2, 0.0, 3, null);
    defer deinitHistory(alloc, sim1);
    try expectHistoryEqual(&[_][]const i32{
        &[_]i32{ -1, 2, -1, -1, -1, 3 },
        &[_]i32{ -1, -1, -1, 2, -1, 0 },
        &[_]i32{ 1, -1, -1, 0, -1, -1 },
    }, sim1);

    const sim2 = try simulate(alloc, &[_]i32{ -1, 2, -1, 3 }, 4, 0.0, 3, null);
    defer deinitHistory(alloc, sim2);
    try expectHistoryEqual(&[_][]const i32{
        &[_]i32{ -1, 2, -1, 3 },
        &[_]i32{ -1, 0, -1, 0 },
        &[_]i32{ -1, 0, -1, 0 },
        &[_]i32{ -1, 0, -1, 0 },
        &[_]i32{ -1, 0, -1, 0 },
    }, sim2);

    const base3 = try constructHighway(alloc, 6, 3, 0);
    defer alloc.free(base3);
    const sim3 = try simulate(alloc, base3, 2, 0.0, 2, null);
    defer deinitHistory(alloc, sim3);
    try expectHistoryEqual(&[_][]const i32{
        &[_]i32{ 0, -1, -1, 0, -1, -1 },
        &[_]i32{ -1, 1, -1, -1, 1, -1 },
        &[_]i32{ -1, -1, 1, -1, -1, 1 },
    }, sim3);

    const base4 = try constructHighway(alloc, 5, 2, -2);
    defer alloc.free(base4);
    const sim4 = try simulate(alloc, base4, 3, 0.0, 2, null);
    defer deinitHistory(alloc, sim4);
    try expectHistoryEqual(&[_][]const i32{
        &[_]i32{ 0, -1, 0, -1, 0 },
        &[_]i32{ 0, -1, 0, -1, -1 },
        &[_]i32{ 0, -1, -1, 1, -1 },
        &[_]i32{ -1, 1, -1, 0, -1 },
    }, sim4);
}

test "nagel schreckenberg: boundary and extreme cases" {
    const alloc = testing.allocator;

    try testing.expectError(NagelSchreckenbergError.InvalidCellCount, constructHighway(alloc, 0, 2, 1));
    try testing.expectError(NagelSchreckenbergError.InvalidFrequency, constructHighway(alloc, 8, 0, 1));
    try testing.expectError(NagelSchreckenbergError.InvalidCarIndex, getDistance(&[_]i32{ 1, -1, 2 }, 3));

    try testing.expectError(NagelSchreckenbergError.InvalidProbability, update(alloc, &[_]i32{ 1, -1, 2 }, 1.2, 5, null));
    try testing.expectError(NagelSchreckenbergError.MissingRandomSource, update(alloc, &[_]i32{ 1, -1, 2 }, 0.3, 5, null));

    const large = try constructHighway(alloc, 1_000, 25, 0);
    defer alloc.free(large);

    const history = try simulate(alloc, large, 250, 0.0, 5, null);
    defer deinitHistory(alloc, history);

    try testing.expectEqual(@as(usize, 251), history.len);
    for (history) |row| {
        try testing.expectEqual(@as(usize, 1_000), row.len);
        for (row) |speed| {
            try testing.expect(speed == -1 or (speed >= 0 and speed <= 5));
        }
    }
}
