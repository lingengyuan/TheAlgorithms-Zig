//! Natural Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/natural_sort.py

const std = @import("std");
const testing = std.testing;

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn compareNumericToken(a: []const u8, b: []const u8) std.math.Order {
    var ia: usize = 0;
    while (ia < a.len and a[ia] == '0') : (ia += 1) {}
    var ib: usize = 0;
    while (ib < b.len and b[ib] == '0') : (ib += 1) {}

    const ta = a[ia..];
    const tb = b[ib..];
    if (ta.len != tb.len) return std.math.order(ta.len, tb.len);
    if (ta.len > 0) {
        const ord = std.mem.order(u8, ta, tb);
        if (ord != .eq) return ord;
    }

    // Numeric value equal (e.g. "001" vs "1"), fallback to original length.
    return std.math.order(a.len, b.len);
}

fn naturalCompare(a: []const u8, b: []const u8) std.math.Order {
    var i: usize = 0;
    var j: usize = 0;

    while (i < a.len and j < b.len) {
        const ad = isDigit(a[i]);
        const bd = isDigit(b[j]);

        if (ad and bd) {
            const si = i;
            while (i < a.len and isDigit(a[i])) : (i += 1) {}
            const sj = j;
            while (j < b.len and isDigit(b[j])) : (j += 1) {}
            const ord_num = compareNumericToken(a[si..i], b[sj..j]);
            if (ord_num != .eq) return ord_num;
            continue;
        }

        const si = i;
        while (i < a.len and !isDigit(a[i])) : (i += 1) {}
        const sj = j;
        while (j < b.len and !isDigit(b[j])) : (j += 1) {}

        const sa = a[si..i];
        const sb = b[sj..j];
        const limit = @min(sa.len, sb.len);
        var k: usize = 0;
        while (k < limit) : (k += 1) {
            const ca = std.ascii.toLower(sa[k]);
            const cb = std.ascii.toLower(sb[k]);
            if (ca != cb) return std.math.order(ca, cb);
        }
        if (sa.len != sb.len) return std.math.order(sa.len, sb.len);
    }

    return std.math.order(a.len, b.len);
}

const IndexedString = struct {
    value: []const u8,
    index: usize,
};

fn lessIndexed(_: void, lhs: IndexedString, rhs: IndexedString) bool {
    const ord = naturalCompare(lhs.value, rhs.value);
    if (ord == .lt) return true;
    if (ord == .gt) return false;
    // Keep stable order for ties.
    return lhs.index < rhs.index;
}

/// Returns natural-sorted copy of input string list.
/// Caller owns returned outer slice; inner slices borrow input.
/// Time complexity: O(n log n * m), Space complexity: O(n)
pub fn naturalSort(allocator: std.mem.Allocator, input_list: []const []const u8) ![][]const u8 {
    const indexed = try allocator.alloc(IndexedString, input_list.len);
    defer allocator.free(indexed);

    for (input_list, 0..) |s, i| indexed[i] = .{ .value = s, .index = i };
    std.mem.sort(IndexedString, indexed, {}, lessIndexed);

    const out = try allocator.alloc([]const u8, input_list.len);
    for (indexed, 0..) |entry, i| out[i] = entry.value;
    return out;
}

test "natural sort: python reference examples" {
    const alloc = testing.allocator;

    const example1 = [_][]const u8{ "2 ft 7 in", "1 ft 5 in", "10 ft 2 in", "2 ft 11 in", "7 ft 6 in" };
    const sorted1 = try naturalSort(alloc, &example1);
    defer alloc.free(sorted1);
    try testing.expectEqualStrings("1 ft 5 in", sorted1[0]);
    try testing.expectEqualStrings("2 ft 7 in", sorted1[1]);
    try testing.expectEqualStrings("2 ft 11 in", sorted1[2]);
    try testing.expectEqualStrings("7 ft 6 in", sorted1[3]);
    try testing.expectEqualStrings("10 ft 2 in", sorted1[4]);

    const example2 = [_][]const u8{ "Elm11", "Elm12", "Elm2", "elm0", "elm1", "elm10", "elm13", "elm9" };
    const sorted2 = try naturalSort(alloc, &example2);
    defer alloc.free(sorted2);
    try testing.expectEqualStrings("elm0", sorted2[0]);
    try testing.expectEqualStrings("elm1", sorted2[1]);
    try testing.expectEqualStrings("Elm2", sorted2[2]);
    try testing.expectEqualStrings("elm9", sorted2[3]);
    try testing.expectEqualStrings("elm10", sorted2[4]);
    try testing.expectEqualStrings("Elm11", sorted2[5]);
    try testing.expectEqualStrings("Elm12", sorted2[6]);
    try testing.expectEqualStrings("elm13", sorted2[7]);
}

test "natural sort: edge and tie cases" {
    const alloc = testing.allocator;

    const empty = [_][]const u8{};
    const sorted_empty = try naturalSort(alloc, &empty);
    defer alloc.free(sorted_empty);
    try testing.expectEqual(@as(usize, 0), sorted_empty.len);

    const ties = [_][]const u8{ "a001", "a1", "a0001", "a01" };
    const sorted_ties = try naturalSort(alloc, &ties);
    defer alloc.free(sorted_ties);
    // same numeric value, shorter numeric token first due fallback
    try testing.expectEqualStrings("a1", sorted_ties[0]);
}

test "natural sort: extreme large list" {
    const alloc = testing.allocator;
    const n: usize = 50_000;
    const data = try alloc.alloc([]const u8, n);
    defer alloc.free(data);
    const backing = try alloc.alloc(u8, n * 8);
    defer alloc.free(backing);

    // Create tokens like x00000..x49999 without extra heap allocations per entry.
    for (0..n) |i| {
        const start = i * 8;
        backing[start] = 'x';
        const num = i;
        backing[start + 1] = @as(u8, '0' + @as(u8, @intCast((num / 10000) % 10)));
        backing[start + 2] = @as(u8, '0' + @as(u8, @intCast((num / 1000) % 10)));
        backing[start + 3] = @as(u8, '0' + @as(u8, @intCast((num / 100) % 10)));
        backing[start + 4] = @as(u8, '0' + @as(u8, @intCast((num / 10) % 10)));
        backing[start + 5] = @as(u8, '0' + @as(u8, @intCast(num % 10)));
        backing[start + 6] = '_';
        backing[start + 7] = 'z';
        data[i] = backing[start .. start + 8];
    }
    std.mem.reverse([]const u8, data);

    const sorted = try naturalSort(alloc, data);
    defer alloc.free(sorted);
    for (1..sorted.len) |i| {
        try testing.expect(naturalCompare(sorted[i - 1], sorted[i]) != .gt);
    }
}
