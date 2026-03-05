//! Karnaugh Map Simplification (2 variables) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/karnaugh_map_simplification.py

const std = @import("std");
const testing = std.testing;

/// Simplifies a 2-variable K-map by enumerating truthy cells.
///
/// Time complexity: O(r * c)
/// Space complexity: O(r * c) output dependent
pub fn simplifyKmap(
    allocator: std.mem.Allocator,
    kmap: []const []const i64,
) std.mem.Allocator.Error![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);

    var first = true;
    for (kmap, 0..) |row, a| {
        for (row, 0..) |item, b| {
            if (item == 0) continue;
            if (!first) try out.appendSlice(allocator, " + ");

            if (a == 0) {
                try out.appendSlice(allocator, "A'");
            } else {
                try out.append(allocator, 'A');
            }

            if (b == 0) {
                try out.appendSlice(allocator, "B'");
            } else {
                try out.append(allocator, 'B');
            }

            first = false;
        }
    }

    return out.toOwnedSlice(allocator);
}

test "karnaugh map simplification: python examples" {
    const alloc = testing.allocator;

    const k1 = [_][]const i64{
        &[_]i64{ 0, 1 },
        &[_]i64{ 1, 1 },
    };
    const s1 = try simplifyKmap(alloc, &k1);
    defer alloc.free(s1);
    try testing.expectEqualStrings("A'B + AB' + AB", s1);

    const k2 = [_][]const i64{
        &[_]i64{ 0, 0 },
        &[_]i64{ 0, 0 },
    };
    const s2 = try simplifyKmap(alloc, &k2);
    defer alloc.free(s2);
    try testing.expectEqualStrings("", s2);

    const k3 = [_][]const i64{
        &[_]i64{ 0, 1 },
        &[_]i64{ 1, -1 },
    };
    const s3 = try simplifyKmap(alloc, &k3);
    defer alloc.free(s3);
    try testing.expectEqualStrings("A'B + AB' + AB", s3);

    const k4 = [_][]const i64{
        &[_]i64{ 0, 1 },
        &[_]i64{ 1, 2 },
    };
    const s4 = try simplifyKmap(alloc, &k4);
    defer alloc.free(s4);
    try testing.expectEqualStrings("A'B + AB' + AB", s4);
}

test "karnaugh map simplification: boundary and extreme dense map" {
    const alloc = testing.allocator;
    const empty = [_][]const i64{};
    const s0 = try simplifyKmap(alloc, &empty);
    defer alloc.free(s0);
    try testing.expectEqualStrings("", s0);

    const dense = [_][]const i64{
        &[_]i64{ 1, 1 },
        &[_]i64{ 1, 1 },
    };
    const s = try simplifyKmap(alloc, &dense);
    defer alloc.free(s);
    try testing.expectEqualStrings("A'B' + A'B + AB' + AB", s);
}
