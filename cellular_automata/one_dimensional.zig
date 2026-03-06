//! One-Dimensional Cellular Automata - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/cellular_automata/one_dimensional.py

const std = @import("std");
const testing = std.testing;

pub const OneDimensionalError = error{
    InvalidRuleset,
    EmptyCells,
    InvalidTime,
    InvalidCellState,
};

/// Converts decimal digit-form ruleset (e.g. 11100 -> "00011100") to rule array.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn formatRuleset(ruleset: u32) OneDimensionalError![8]u8 {
    if (ruleset > 99_999_999) {
        return OneDimensionalError.InvalidRuleset;
    }

    var buf: [8]u8 = [_]u8{'0'} ** 8;
    var value = ruleset;
    var idx: isize = 7;
    while (idx >= 0) : (idx -= 1) {
        const digit = value % 10;
        if (digit > 1) {
            return OneDimensionalError.InvalidRuleset;
        }
        buf[@as(usize, @intCast(idx))] = @as(u8, @intCast('0' + digit));
        value /= 10;
    }

    var out: [8]u8 = undefined;
    for (buf, 0..) |ch, i| {
        out[i] = ch - '0';
    }
    return out;
}

/// Computes next generation from `cells[time]` using given rule table.
/// Caller owns returned slice.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn newGeneration(
    allocator: std.mem.Allocator,
    cells: []const []const u8,
    rule: [8]u8,
    time: usize,
) (std.mem.Allocator.Error || OneDimensionalError)![]u8 {
    if (cells.len == 0 or cells[0].len == 0) {
        return OneDimensionalError.EmptyCells;
    }
    if (time >= cells.len) {
        return OneDimensionalError.InvalidTime;
    }

    const population = cells[0].len;
    const current = cells[time];
    if (current.len != population) {
        return OneDimensionalError.EmptyCells;
    }

    const next = try allocator.alloc(u8, population);
    errdefer allocator.free(next);

    for (0..population) |i| {
        const left = if (i == 0) 0 else current[i - 1];
        const center = current[i];
        const right = if (i == population - 1) 0 else current[i + 1];

        if (left > 1 or center > 1 or right > 1) {
            return OneDimensionalError.InvalidCellState;
        }

        const pattern: u8 = (left << 2) | (center << 1) | right;
        const situation = 7 - pattern;
        next[i] = rule[situation];
    }

    return next;
}

test "one dimensional automata: python ruleset examples" {
    const r1 = try formatRuleset(11100);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 1, 1, 1, 0, 0 }, &r1);

    const r2 = try formatRuleset(0);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 }, &r2);

    const r3 = try formatRuleset(11111111);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1, 1, 1, 1, 1, 1 }, &r3);
}

test "one dimensional automata: generation and edge cases" {
    const alloc = testing.allocator;
    const rule = try formatRuleset(11100);

    const cells = [_][]const u8{&[_]u8{ 0, 0, 1, 0, 0 }};
    const next = try newGeneration(alloc, &cells, rule, 0);
    defer alloc.free(next);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 1, 1, 0 }, next);

    try testing.expectError(OneDimensionalError.InvalidRuleset, formatRuleset(12345678));
    try testing.expectError(OneDimensionalError.EmptyCells, newGeneration(alloc, &[_][]const u8{}, rule, 0));
    try testing.expectError(OneDimensionalError.InvalidTime, newGeneration(alloc, &cells, rule, 1));

    const bad_cells = [_][]const u8{&[_]u8{ 0, 2, 1 }};
    try testing.expectError(OneDimensionalError.InvalidCellState, newGeneration(alloc, &bad_cells, rule, 0));
}
