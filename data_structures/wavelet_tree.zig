//! Wavelet Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/wavelet_tree.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    minn: i64,
    maxx: i64,
    map_left: []usize,
    left: ?*Node = null,
    right: ?*Node = null,
};

pub fn freeTree(allocator: std.mem.Allocator, root: ?*Node) void {
    const start = root orelse return;

    var stack = std.ArrayListUnmanaged(*Node){};
    defer stack.deinit(allocator);
    stack.append(allocator, start) catch return;

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        if (node.left) |left| stack.append(allocator, left) catch {};
        if (node.right) |right| stack.append(allocator, right) catch {};
        allocator.free(node.map_left);
        allocator.destroy(node);
    }
}

/// Builds wavelet tree from input array.
/// Time complexity: O(n log sigma), Space complexity: O(n log sigma)
pub fn buildTree(allocator: std.mem.Allocator, arr: []const i64) !?*Node {
    if (arr.len == 0) return null;

    var min_v = arr[0];
    var max_v = arr[0];
    for (arr[1..]) |v| {
        min_v = @min(min_v, v);
        max_v = @max(max_v, v);
    }

    const node = try allocator.create(Node);
    errdefer allocator.destroy(node);

    const map_left = try allocator.alloc(usize, arr.len);
    errdefer allocator.free(map_left);

    node.* = .{
        .minn = min_v,
        .maxx = max_v,
        .map_left = map_left,
    };

    if (min_v == max_v) {
        @memset(node.map_left, arr.len);
        return node;
    }

    const pivot = @divFloor(min_v + max_v, 2);

    var left_values = std.ArrayListUnmanaged(i64){};
    defer left_values.deinit(allocator);
    var right_values = std.ArrayListUnmanaged(i64){};
    defer right_values.deinit(allocator);

    for (arr, 0..) |num, index| {
        if (num <= pivot) {
            try left_values.append(allocator, num);
        } else {
            try right_values.append(allocator, num);
        }
        node.map_left[index] = left_values.items.len;
    }

    const left_slice = try left_values.toOwnedSlice(allocator);
    defer allocator.free(left_slice);
    const right_slice = try right_values.toOwnedSlice(allocator);
    defer allocator.free(right_slice);

    node.left = try buildTree(allocator, left_slice);
    node.right = try buildTree(allocator, right_slice);

    return node;
}

fn clampIndex(node: *const Node, index: isize) ?usize {
    if (index < 0 or node.map_left.len == 0) return null;
    if (index >= @as(isize, @intCast(node.map_left.len))) return node.map_left.len - 1;
    return @intCast(index);
}

/// Returns number of occurrences of `num` in range [0, index].
/// Time complexity: O(log sigma), Space complexity: O(log sigma)
pub fn rankTillIndex(node: ?*const Node, num: i64, index: isize) usize {
    if (index < 0 or node == null) return 0;

    const n = node.?;
    const idx = clampIndex(n, index) orelse return 0;

    if (n.minn == n.maxx) {
        return if (n.minn == num) idx + 1 else 0;
    }

    const pivot = @divFloor(n.minn + n.maxx, 2);
    if (num <= pivot) {
        const mapped = @as(isize, @intCast(n.map_left[idx])) - 1;
        return rankTillIndex(n.left, num, mapped);
    }

    const mapped = @as(isize, @intCast(idx)) - @as(isize, @intCast(n.map_left[idx]));
    return rankTillIndex(n.right, num, mapped);
}

/// Returns number of occurrences of `num` in range [start, end].
/// Time complexity: O(log sigma), Space complexity: O(log sigma)
pub fn rank(node: ?*const Node, num: i64, start: isize, end: isize) usize {
    if (start > end) return 0;
    const till_end = rankTillIndex(node, num, end);
    const before_start = rankTillIndex(node, num, start - 1);
    return till_end - before_start;
}

/// Returns `index`-th smallest element in range [start, end], index is 0-based.
/// Returns -1 for invalid arguments.
/// Time complexity: O(log sigma), Space complexity: O(log sigma)
pub fn quantile(node: ?*const Node, index: isize, start: isize, end: isize) i64 {
    const n = node orelse return -1;

    if (start > end or index < 0 or index > (end - start)) return -1;
    if (start < 0 or end < 0) return -1;

    const start_u: usize = @intCast(start);
    const end_u: usize = @intCast(end);
    if (end_u >= n.map_left.len or start_u >= n.map_left.len) return -1;

    if (n.minn == n.maxx) return n.minn;

    const left_before: usize = if (start_u > 0) n.map_left[start_u - 1] else 0;
    const left_end: usize = n.map_left[end_u];
    const num_left: isize = @intCast(left_end - left_before);

    if (num_left > index) {
        return quantile(
            n.left,
            index,
            @intCast(left_before),
            @intCast(left_end - 1),
        );
    }

    return quantile(
        n.right,
        index - num_left,
        start - @as(isize, @intCast(left_before)),
        end - @as(isize, @intCast(left_end)),
    );
}

/// Returns count of numbers in [start_num, end_num] over interval [start, end].
/// Time complexity: O(log sigma), Space complexity: O(log sigma)
pub fn rangeCounting(
    node: ?*const Node,
    start: isize,
    end: isize,
    start_num: i64,
    end_num: i64,
) usize {
    const n = node orelse return 0;

    if (start > end or start_num > end_num or start < 0 or end < 0) return 0;

    const start_u: usize = @intCast(start);
    const end_u: usize = @intCast(end);
    if (end_u >= n.map_left.len or start_u >= n.map_left.len) return 0;

    if (n.minn > end_num or n.maxx < start_num) return 0;
    if (start_num <= n.minn and n.maxx <= end_num) {
        return @intCast(end - start + 1);
    }

    const left_before: usize = if (start_u > 0) n.map_left[start_u - 1] else 0;
    const left_end: usize = n.map_left[end_u];

    const left = rangeCounting(
        n.left,
        @intCast(left_before),
        @as(isize, @intCast(left_end)) - 1,
        start_num,
        end_num,
    );

    const right = rangeCounting(
        n.right,
        start - @as(isize, @intCast(left_before)),
        end - @as(isize, @intCast(left_end)),
        start_num,
        end_num,
    );

    return left + right;
}

fn lessI64(_: void, a: i64, b: i64) bool {
    return a < b;
}

fn naiveRank(values: []const i64, num: i64, l: usize, r: usize) usize {
    var count: usize = 0;
    for (values[l .. r + 1]) |v| {
        if (v == num) count += 1;
    }
    return count;
}

fn naiveRangeCount(values: []const i64, l: usize, r: usize, lo: i64, hi: i64) usize {
    var count: usize = 0;
    for (values[l .. r + 1]) |v| {
        if (lo <= v and v <= hi) count += 1;
    }
    return count;
}

fn naiveQuantile(allocator: std.mem.Allocator, values: []const i64, l: usize, r: usize, idx: usize) !i64 {
    if (idx > r - l) return -1;

    const tmp = try allocator.alloc(i64, r - l + 1);
    defer allocator.free(tmp);

    @memcpy(tmp, values[l .. r + 1]);
    std.mem.sort(i64, tmp, {}, lessI64);
    return tmp[idx];
}

test "wavelet tree: python sample queries" {
    const sample = [_]i64{ 2, 1, 4, 5, 6, 0, 8, 9, 1, 2, 0, 6, 4, 2, 0, 6, 5, 3, 2, 7 };

    const root = (try buildTree(testing.allocator, &sample)).?;
    defer freeTree(testing.allocator, root);

    try testing.expectEqual(@as(i64, 0), root.minn);
    try testing.expectEqual(@as(i64, 9), root.maxx);

    try testing.expectEqual(@as(usize, 1), rankTillIndex(root, 6, 6));
    try testing.expectEqual(@as(usize, 1), rankTillIndex(root, 2, 0));
    try testing.expectEqual(@as(usize, 2), rankTillIndex(root, 1, 10));
    try testing.expectEqual(@as(usize, 0), rankTillIndex(root, 17, 7));
    try testing.expectEqual(@as(usize, 1), rankTillIndex(root, 0, 9));

    try testing.expectEqual(@as(usize, 2), rank(root, 6, 3, 13));
    try testing.expectEqual(@as(usize, 4), rank(root, 2, 0, 19));
    try testing.expectEqual(@as(usize, 0), rank(root, 9, 2, 2));
    try testing.expectEqual(@as(usize, 2), rank(root, 0, 5, 10));

    try testing.expectEqual(@as(i64, 5), quantile(root, 2, 2, 5));
    try testing.expectEqual(@as(i64, 4), quantile(root, 5, 2, 13));
    try testing.expectEqual(@as(i64, 8), quantile(root, 0, 6, 6));
    try testing.expectEqual(@as(i64, -1), quantile(root, 4, 2, 5));

    try testing.expectEqual(@as(usize, 3), rangeCounting(root, 1, 10, 3, 7));
    try testing.expectEqual(@as(usize, 1), rangeCounting(root, 2, 2, 1, 4));
    try testing.expectEqual(@as(usize, 20), rangeCounting(root, 0, 19, 0, 100));
    try testing.expectEqual(@as(usize, 0), rangeCounting(root, 1, 0, 1, 100));
    try testing.expectEqual(@as(usize, 0), rangeCounting(root, 0, 17, 100, 1));
}

test "wavelet tree: boundary null and invalid ranges" {
    try testing.expectEqual(@as(usize, 0), rankTillIndex(null, 1, 5));
    try testing.expectEqual(@as(usize, 0), rank(null, 1, 3, 5));
    try testing.expectEqual(@as(i64, -1), quantile(null, 0, 0, 0));
    try testing.expectEqual(@as(usize, 0), rangeCounting(null, 0, 0, 0, 10));
}

test "wavelet tree: extreme randomized parity" {
    const n: usize = 6_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    var prng = std.Random.DefaultPrng.init(0xBADC0FFEE0DDF00D);
    const random = prng.random();

    for (0..n) |i| {
        values[i] = @intCast(random.uintLessThan(u16, 1024));
    }

    const root = (try buildTree(testing.allocator, values)).?;
    defer freeTree(testing.allocator, root);

    var q: usize = 0;
    while (q < 600) : (q += 1) {
        const a = random.uintLessThan(usize, n);
        const b = random.uintLessThan(usize, n);
        const l = @min(a, b);
        const r = @max(a, b);

        const num: i64 = @intCast(random.uintLessThan(u16, 1024));
        try testing.expectEqual(naiveRank(values, num, l, r), rank(root, num, @intCast(l), @intCast(r)));

        const lo_u = random.uintLessThan(u16, 1024);
        const hi_u = random.uintLessThan(u16, 1024);
        const lo = @as(i64, @intCast(@min(lo_u, hi_u)));
        const hi = @as(i64, @intCast(@max(lo_u, hi_u)));
        try testing.expectEqual(naiveRangeCount(values, l, r, lo, hi), rangeCounting(root, @intCast(l), @intCast(r), lo, hi));

        const len = r - l + 1;
        const idx = random.uintLessThan(usize, len);
        try testing.expectEqual(
            try naiveQuantile(testing.allocator, values, l, r, idx),
            quantile(root, @intCast(idx), @intCast(l), @intCast(r)),
        );
    }
}
