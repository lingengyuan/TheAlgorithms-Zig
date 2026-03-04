//! Stock Span Problem - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/stock_span_problem.py

const std = @import("std");
const testing = std.testing;

/// Calculates stock spans for each day.
/// Time complexity: O(n), Space complexity: O(n)
pub fn calculateSpan(allocator: std.mem.Allocator, price: []const i64) ![]usize {
    if (price.len == 0) return allocator.alloc(usize, 0);

    const spans = try allocator.alloc(usize, price.len);
    var st = std.ArrayListUnmanaged(usize){};
    defer st.deinit(allocator);

    spans[0] = 1;
    try st.append(allocator, 0);

    var i: usize = 1;
    while (i < price.len) : (i += 1) {
        while (st.items.len > 0 and price[st.items[st.items.len - 1]] <= price[i]) {
            _ = st.pop();
        }

        spans[i] = if (st.items.len == 0) i + 1 else i - st.items[st.items.len - 1];
        try st.append(allocator, i);
    }

    return spans;
}

test "stock span problem: python samples" {
    {
        const span = try calculateSpan(testing.allocator, &[_]i64{ 10, 4, 5, 90, 120, 80 });
        defer testing.allocator.free(span);
        try testing.expectEqualSlices(usize, &[_]usize{ 1, 1, 2, 4, 5, 1 }, span);
    }
    {
        const span = try calculateSpan(testing.allocator, &[_]i64{ 100, 50, 60, 70, 80, 90 });
        defer testing.allocator.free(span);
        try testing.expectEqualSlices(usize, &[_]usize{ 1, 1, 2, 3, 4, 5 }, span);
    }
    {
        const span = try calculateSpan(testing.allocator, &[_]i64{ 100, 80, 60, 70, 60, 75, 85 });
        defer testing.allocator.free(span);
        try testing.expectEqualSlices(usize, &[_]usize{ 1, 1, 1, 2, 1, 4, 6 }, span);
    }
}

test "stock span problem: empty and extreme" {
    const empty = try calculateSpan(testing.allocator, &[_]i64{});
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const n: usize = 50_000;
    var increasing = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(increasing);
    for (0..n) |i| increasing[i] = @intCast(i + 1);

    const spans = try calculateSpan(testing.allocator, increasing);
    defer testing.allocator.free(spans);

    try testing.expectEqual(@as(usize, 1), spans[0]);
    try testing.expectEqual(n, spans[n - 1]);
}
