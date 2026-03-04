//! Jaccard Similarity - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/jaccard_similarity.py

const std = @import("std");
const testing = std.testing;

pub const JaccardError = error{ IncompatibleTypes, EmptyUnion };

pub const Collection = union(enum) {
    set: []const []const u8,
    sequence: []const []const u8,
};

/// Returns Jaccard similarity between two collections.
/// For `.set`, duplicates are ignored. For `.sequence`, Python list/tuple semantics are followed.
/// Time complexity: O(n + m) average with hash maps, Space complexity: O(n + m)
pub fn jaccardSimilarity(
    allocator: std.mem.Allocator,
    set_a: Collection,
    set_b: Collection,
    alternative_union: bool,
) (JaccardError || std.mem.Allocator.Error)!f64 {
    switch (set_a) {
        .set => |a_items| switch (set_b) {
            .set => |b_items| return jaccardForSets(allocator, a_items, b_items, alternative_union),
            else => return JaccardError.IncompatibleTypes,
        },
        .sequence => |a_items| switch (set_b) {
            .sequence => |b_items| return jaccardForSequences(allocator, a_items, b_items, alternative_union),
            else => return JaccardError.IncompatibleTypes,
        },
    }
}

fn jaccardForSets(
    allocator: std.mem.Allocator,
    a_items: []const []const u8,
    b_items: []const []const u8,
    alternative_union: bool,
) (JaccardError || std.mem.Allocator.Error)!f64 {
    var a_map = std.StringHashMap(void).init(allocator);
    defer a_map.deinit();
    for (a_items) |item| {
        try a_map.put(item, {});
    }

    var b_map = std.StringHashMap(void).init(allocator);
    defer b_map.deinit();
    for (b_items) |item| {
        try b_map.put(item, {});
    }

    var intersection_len: usize = 0;
    var it = a_map.iterator();
    while (it.next()) |entry| {
        if (b_map.contains(entry.key_ptr.*)) intersection_len += 1;
    }

    const union_len: usize = if (alternative_union)
        a_map.count() + b_map.count()
    else blk: {
        var count = a_map.count();
        var b_it = b_map.iterator();
        while (b_it.next()) |entry| {
            if (!a_map.contains(entry.key_ptr.*)) count += 1;
        }
        break :blk count;
    };

    if (union_len == 0) return JaccardError.EmptyUnion;
    return @as(f64, @floatFromInt(intersection_len)) / @as(f64, @floatFromInt(union_len));
}

fn jaccardForSequences(
    allocator: std.mem.Allocator,
    a_items: []const []const u8,
    b_items: []const []const u8,
    alternative_union: bool,
) (JaccardError || std.mem.Allocator.Error)!f64 {
    var intersection_len: usize = 0;
    for (a_items) |item| {
        if (containsString(b_items, item)) intersection_len += 1;
    }

    var union_len: usize = undefined;
    if (alternative_union) {
        union_len = a_items.len + b_items.len;
    } else {
        var merged = std.ArrayListUnmanaged([]const u8){};
        defer merged.deinit(allocator);
        try merged.appendSlice(allocator, a_items);
        for (b_items) |item| {
            if (!containsString(a_items, item)) {
                try merged.append(allocator, item);
            }
        }
        union_len = merged.items.len;
    }

    if (union_len == 0) return JaccardError.EmptyUnion;
    return @as(f64, @floatFromInt(intersection_len)) / @as(f64, @floatFromInt(union_len));
}

fn containsString(items: []const []const u8, target: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item, target)) return true;
    }
    return false;
}

test "jaccard similarity: python reference examples for sets" {
    const a = [_][]const u8{ "a", "b", "c", "d", "e" };
    const b = [_][]const u8{ "c", "d", "e", "f", "h", "i" };
    try testing.expectApproxEqAbs(@as(f64, 0.375), try jaccardSimilarity(testing.allocator, .{ .set = &a }, .{ .set = &b }, false), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try jaccardSimilarity(testing.allocator, .{ .set = &a }, .{ .set = &a }, false), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), try jaccardSimilarity(testing.allocator, .{ .set = &a }, .{ .set = &a }, true), 1e-12);
}

test "jaccard similarity: python reference examples for sequences" {
    const a = [_][]const u8{ "a", "b", "c", "d", "e" };
    const b = [_][]const u8{ "c", "d", "e", "f", "h", "i" };
    try testing.expectApproxEqAbs(@as(f64, 0.375), try jaccardSimilarity(testing.allocator, .{ .sequence = &a }, .{ .sequence = &b }, false), 1e-12);

    const c = [_][]const u8{ "c", "d", "e", "f", "h", "i" };
    const d = [_][]const u8{ "a", "b", "c", "d" };
    try testing.expectApproxEqAbs(@as(f64, 0.2), try jaccardSimilarity(testing.allocator, .{ .sequence = &c }, .{ .sequence = &d }, true), 1e-12);
}

test "jaccard similarity: edge and error cases" {
    const a = [_][]const u8{ "a", "b" };
    const b = [_][]const u8{ "c", "d" };
    try testing.expectError(JaccardError.IncompatibleTypes, jaccardSimilarity(testing.allocator, .{ .set = &a }, .{ .sequence = &b }, false));

    const empty = [_][]const u8{};
    try testing.expectError(JaccardError.EmptyUnion, jaccardSimilarity(testing.allocator, .{ .set = &empty }, .{ .set = &empty }, false));
}
