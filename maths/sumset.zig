//! Sumset - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sumset.py

const std = @import("std");
const testing = std.testing;

/// Returns set {a + b | a in set_a, b in set_b} as sorted unique slice.
/// Caller owns returned slice.
/// Time complexity: O(n * m), Space complexity: O(n * m)
pub fn sumset(
    allocator: std.mem.Allocator,
    set_a: []const i64,
    set_b: []const i64,
) std.mem.Allocator.Error![]i64 {
    var map = std.AutoHashMap(i64, void).init(allocator);
    defer map.deinit();

    for (set_a) |a| {
        for (set_b) |b| {
            try map.put(a + b, {});
        }
    }

    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    var it = map.iterator();
    while (it.next()) |entry| {
        try out.append(allocator, entry.key_ptr.*);
    }

    std.mem.sort(i64, out.items, {}, std.sort.asc(i64));
    return try out.toOwnedSlice(allocator);
}

test "sumset: python reference examples" {
    const r1 = try sumset(testing.allocator, &[_]i64{ 1, 2, 3 }, &[_]i64{ 4, 5, 6 });
    defer testing.allocator.free(r1);
    try testing.expectEqualSlices(i64, &[_]i64{ 5, 6, 7, 8, 9 }, r1);

    const r2 = try sumset(testing.allocator, &[_]i64{ 1, 2, 3 }, &[_]i64{ 4, 5, 6, 7 });
    defer testing.allocator.free(r2);
    try testing.expectEqualSlices(i64, &[_]i64{ 5, 6, 7, 8, 9, 10 }, r2);
}

test "sumset: edge and extreme cases" {
    const e1 = try sumset(testing.allocator, &[_]i64{}, &[_]i64{ 1, 2, 3 });
    defer testing.allocator.free(e1);
    try testing.expectEqual(@as(usize, 0), e1.len);

    const e2 = try sumset(testing.allocator, &[_]i64{ -10, 0, 10 }, &[_]i64{ -10, 0, 10 });
    defer testing.allocator.free(e2);
    try testing.expectEqualSlices(i64, &[_]i64{ -20, -10, 0, 10, 20 }, e2);
}
