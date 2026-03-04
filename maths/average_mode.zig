//! Average Mode - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/average_mode.py

const std = @import("std");
const testing = std.testing;

/// Returns sorted mode values from `input_list`.
/// For empty input, returns empty slice.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn mode(allocator: std.mem.Allocator, input_list: []const i64) std.mem.Allocator.Error![]i64 {
    if (input_list.len == 0) return try allocator.alloc(i64, 0);

    var counts = std.AutoHashMap(i64, usize).init(allocator);
    defer counts.deinit();

    for (input_list) |value| {
        const entry = try counts.getOrPut(value);
        if (entry.found_existing) {
            entry.value_ptr.* += 1;
        } else {
            entry.value_ptr.* = 1;
        }
    }

    var max_count: usize = 0;
    var it_find = counts.iterator();
    while (it_find.next()) |entry| {
        if (entry.value_ptr.* > max_count) {
            max_count = entry.value_ptr.*;
        }
    }

    var modes = std.ArrayListUnmanaged(i64){};
    errdefer modes.deinit(allocator);

    var it_collect = counts.iterator();
    while (it_collect.next()) |entry| {
        if (entry.value_ptr.* == max_count) {
            try modes.append(allocator, entry.key_ptr.*);
        }
    }

    std.mem.sort(i64, modes.items, {}, std.sort.asc(i64));
    return try modes.toOwnedSlice(allocator);
}

test "average mode: python reference examples" {
    const alloc = testing.allocator;

    const m1 = try mode(alloc, &[_]i64{ 2, 3, 4, 5, 3, 4, 2, 5, 2, 2, 4, 2, 2, 2 });
    defer alloc.free(m1);
    try testing.expectEqualSlices(i64, &[_]i64{2}, m1);

    const m2 = try mode(alloc, &[_]i64{ 3, 4, 5, 3, 4, 2, 5, 2, 2, 4, 4, 2, 2, 2 });
    defer alloc.free(m2);
    try testing.expectEqualSlices(i64, &[_]i64{2}, m2);

    const m3 = try mode(alloc, &[_]i64{ 3, 4, 5, 3, 4, 2, 5, 2, 2, 4, 4, 4, 2, 2, 4, 2 });
    defer alloc.free(m3);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 4 }, m3);
}

test "average mode: edge and extreme cases" {
    const empty = try mode(testing.allocator, &[_]i64{});
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var large: [10_000]i64 = undefined;
    for (&large, 0..) |*slot, idx| {
        slot.* = @intCast(idx % 10);
    }
    const result = try mode(testing.allocator, &large);
    defer testing.allocator.free(result);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, result);
}
