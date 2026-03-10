//! Edmonds-Karp with Multiple Sources and Sinks - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/edmonds_karp_multiple_source_and_sink.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const flow = @import("ford_fulkerson.zig");

/// Computes the maximum flow of a capacity matrix with one or more sources and sinks.
/// When multiple sources or sinks are provided, a super-source/super-sink pair is added
/// following the Python reference normalization strategy.
/// Returns `0` when either source or sink list is empty.
/// Time complexity: O(V^5) worst-case after matrix expansion, Space complexity: O(V^2)
pub fn maxFlowMultipleSourcesSinks(
    allocator: Allocator,
    capacity: []const []const i64,
    sources: []const usize,
    sinks: []const usize,
) !i64 {
    const n = capacity.len;
    const elem_count = @mulWithOverflow(n, n);
    if (elem_count[1] != 0) return error.Overflow;

    for (capacity) |row| {
        if (row.len != n) return error.InvalidMatrix;
        for (row) |cap| {
            if (cap < 0) return error.NegativeCapacity;
        }
    }

    for (sources) |source| {
        if (source >= n) return error.InvalidNode;
    }
    for (sinks) |sink| {
        if (sink >= n) return error.InvalidNode;
    }

    if (sources.len == 0 or sinks.len == 0) return 0;
    if (sources.len == 1 and sinks.len == 1) {
        return flow.fordFulkersonMaxFlow(allocator, capacity, sources[0], sinks[0]);
    }

    var super_capacity = try allocator.alloc(i64, (n + 2) * (n + 2));
    defer allocator.free(super_capacity);
    @memset(super_capacity, 0);

    const expanded_n = n + 2;
    const super_source: usize = n;
    const super_sink: usize = n + 1;

    for (0..n) |i| {
        for (0..n) |j| {
            super_capacity[i * expanded_n + j] = capacity[i][j];
        }
    }

    var max_input_flow: i64 = 0;
    for (sources) |source| {
        for (capacity[source]) |cap| {
            const sum = @addWithOverflow(max_input_flow, cap);
            if (sum[1] != 0) return error.Overflow;
            max_input_flow = sum[0];
        }
    }

    for (sources) |source| {
        super_capacity[super_source * expanded_n + source] = max_input_flow;
    }
    for (sinks) |sink| {
        super_capacity[sink * expanded_n + super_sink] = max_input_flow;
    }

    const rows = try allocator.alloc([]const i64, expanded_n);
    defer allocator.free(rows);
    for (0..expanded_n) |i| {
        rows[i] = super_capacity[i * expanded_n .. (i + 1) * expanded_n];
    }

    return flow.fordFulkersonMaxFlow(allocator, rows, super_source, super_sink);
}

test "edmonds karp multiple source sink: python single source sample" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 7, 0, 0 },
        &[_]i64{ 0, 0, 6, 0 },
        &[_]i64{ 0, 0, 0, 8 },
        &[_]i64{ 9, 0, 0, 0 },
    };

    try testing.expectEqual(@as(i64, 6), try maxFlowMultipleSourcesSinks(alloc, &capacity, &[_]usize{0}, &[_]usize{3}));
}

test "edmonds karp multiple source sink: multiple sources and sinks" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 0, 5, 0, 0 },
        &[_]i64{ 0, 0, 4, 0, 0 },
        &[_]i64{ 0, 0, 0, 6, 3 },
        &[_]i64{ 0, 0, 0, 0, 0 },
        &[_]i64{ 0, 0, 0, 0, 0 },
    };

    try testing.expectEqual(@as(i64, 9), try maxFlowMultipleSourcesSinks(alloc, &capacity, &[_]usize{ 0, 1 }, &[_]usize{ 3, 4 }));
}

test "edmonds karp multiple source sink: empty source or sink returns zero" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 1 },
        &[_]i64{ 0, 0 },
    };

    try testing.expectEqual(@as(i64, 0), try maxFlowMultipleSourcesSinks(alloc, &capacity, &[_]usize{}, &[_]usize{1}));
    try testing.expectEqual(@as(i64, 0), try maxFlowMultipleSourcesSinks(alloc, &capacity, &[_]usize{0}, &[_]usize{}));
}

test "edmonds karp multiple source sink: invalid and negative inputs" {
    const alloc = testing.allocator;
    const capacity = [_][]const i64{
        &[_]i64{ 0, 1 },
        &[_]i64{ 0, 0 },
    };
    try testing.expectError(error.InvalidNode, maxFlowMultipleSourcesSinks(alloc, &capacity, &[_]usize{2}, &[_]usize{1}));

    const negative = [_][]const i64{
        &[_]i64{ 0, -1 },
        &[_]i64{ 0, 0 },
    };
    try testing.expectError(error.NegativeCapacity, maxFlowMultipleSourcesSinks(alloc, &negative, &[_]usize{0}, &[_]usize{1}));
}

test "edmonds karp multiple source sink: extreme parallel chains" {
    const alloc = testing.allocator;
    const n: usize = 34;
    const data = try alloc.alloc(i64, n * n);
    defer alloc.free(data);
    @memset(data, 0);

    for (0..16) |i| {
        data[i * n + (16 + i)] = 1;
        data[(16 + i) * n + (n - 1)] = 1;
    }

    const rows = try alloc.alloc([]const i64, n);
    defer alloc.free(rows);
    for (0..n) |i| rows[i] = data[i * n .. (i + 1) * n];

    const sources = try alloc.alloc(usize, 16);
    defer alloc.free(sources);
    const sinks = [_]usize{n - 1};
    for (0..16) |i| sources[i] = i;

    try testing.expectEqual(@as(i64, 16), try maxFlowMultipleSourcesSinks(alloc, rows, sources, &sinks));
}
