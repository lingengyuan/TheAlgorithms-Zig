//! Spearman Rank Correlation Coefficient - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/spearman_rank_correlation_coefficient.py

const std = @import("std");
const testing = std.testing;

const RankedValue = struct {
    value: f64,
    index: usize,
};

fn lessThan(_: void, lhs: RankedValue, rhs: RankedValue) bool {
    return lhs.value < rhs.value;
}

/// Assigns ranks to the input data.
/// Caller owns the returned slice.
pub fn assignRanks(allocator: std.mem.Allocator, data: []const f64) ![]usize {
    const ranked = try allocator.alloc(RankedValue, data.len);
    defer allocator.free(ranked);
    for (data, 0..) |value, index| {
        ranked[index] = .{ .value = value, .index = index };
    }
    std.sort.heap(RankedValue, ranked, {}, lessThan);

    const ranks = try allocator.alloc(usize, data.len);
    for (ranked, 0..) |pair, position| {
        ranks[pair.index] = position + 1;
    }
    return ranks;
}

/// Calculates Spearman's rank correlation coefficient.
pub fn calculateSpearmanRankCorrelation(
    allocator: std.mem.Allocator,
    variable_1: []const f64,
    variable_2: []const f64,
) !f64 {
    const rank_var1 = try assignRanks(allocator, variable_1);
    defer allocator.free(rank_var1);
    const rank_var2 = try assignRanks(allocator, variable_2);
    defer allocator.free(rank_var2);

    const n = @as(f64, @floatFromInt(variable_1.len));
    var d_squared: f64 = 0;
    for (rank_var1, rank_var2) |rx, ry| {
        const d = @as(f64, @floatFromInt(rx)) - @as(f64, @floatFromInt(ry));
        d_squared += d * d;
    }
    return 1.0 - (6.0 * d_squared) / (n * (n * n - 1.0));
}

test "spearman rank correlation: python reference examples" {
    const alloc = testing.allocator;
    const ranks = try assignRanks(alloc, &[_]f64{ 3.2, 1.5, 4.0, 2.7, 5.1 });
    defer alloc.free(ranks);
    try testing.expectEqualSlices(usize, &[_]usize{ 3, 1, 4, 2, 5 }, ranks);

    try testing.expectApproxEqAbs(@as(f64, -1.0), try calculateSpearmanRankCorrelation(alloc, &[_]f64{ 1, 2, 3, 4, 5 }, &[_]f64{ 5, 4, 3, 2, 1 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try calculateSpearmanRankCorrelation(alloc, &[_]f64{ 1, 2, 3, 4, 5 }, &[_]f64{ 2, 4, 6, 8, 10 }), 1e-12);
}

test "spearman rank correlation: edge cases" {
    const alloc = testing.allocator;
    try testing.expectApproxEqAbs(@as(f64, 0.6), try calculateSpearmanRankCorrelation(alloc, &[_]f64{ 1, 2, 3, 4, 5 }, &[_]f64{ 5, 1, 2, 9, 5 }), 1e-12);
}
