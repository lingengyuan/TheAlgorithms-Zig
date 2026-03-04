//! Linked List Merge Two Lists - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/merge_two_lists.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Node = struct {
    data: i64,
    next: ?*Node,
};

pub const SortedLinkedList = struct {
    allocator: Allocator,
    head: ?*Node,
    len_: usize,

    pub fn deinit(self: *SortedLinkedList) void {
        var node_opt = self.head;
        while (node_opt) |node| {
            const next = node.next;
            self.allocator.destroy(node);
            node_opt = next;
        }
        self.* = undefined;
    }

    pub fn len(self: *const SortedLinkedList) usize {
        return self.len_;
    }

    pub fn toOwnedSlice(self: *const SortedLinkedList, allocator: Allocator) ![]i64 {
        const out = try allocator.alloc(i64, self.len_);
        var idx: usize = 0;
        var node_opt = self.head;
        while (node_opt) |node| {
            out[idx] = node.data;
            idx += 1;
            node_opt = node.next;
        }
        return out;
    }
};

/// Creates sorted linked list from any input sequence.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn initSortedLinkedList(allocator: Allocator, ints: []const i64) !SortedLinkedList {
    const sorted = try allocator.alloc(i64, ints.len);
    defer allocator.free(sorted);
    @memcpy(sorted, ints);
    std.mem.sort(i64, sorted, {}, std.sort.asc(i64));

    var head: ?*Node = null;
    var i = sorted.len;
    while (i > 0) {
        i -= 1;
        const node = try allocator.create(Node);
        node.* = .{ .data = sorted[i], .next = head };
        head = node;
    }

    return .{
        .allocator = allocator,
        .head = head,
        .len_ = sorted.len,
    };
}

/// Merges two sorted linked lists.
/// Time complexity: O((n+m) log(n+m)), Space complexity: O(n+m)
pub fn mergeLists(allocator: Allocator, one: *const SortedLinkedList, two: *const SortedLinkedList) !SortedLinkedList {
    const total = one.len_ + two.len_;
    var values = try allocator.alloc(i64, total);
    defer allocator.free(values);

    var idx: usize = 0;
    var node_opt = one.head;
    while (node_opt) |node| {
        values[idx] = node.data;
        idx += 1;
        node_opt = node.next;
    }

    node_opt = two.head;
    while (node_opt) |node| {
        values[idx] = node.data;
        idx += 1;
        node_opt = node.next;
    }

    return initSortedLinkedList(allocator, values);
}

test "linked list merge two lists: python sample" {
    const odd = [_]i64{ 3, 9, -11, 0, 7, 5, 1, -1 };
    const even = [_]i64{ 4, 6, 2, 0, 8, 10, 3, -2 };

    var l1 = try initSortedLinkedList(testing.allocator, &odd);
    defer l1.deinit();
    var l2 = try initSortedLinkedList(testing.allocator, &even);
    defer l2.deinit();

    var merged = try mergeLists(testing.allocator, &l1, &l2);
    defer merged.deinit();

    try testing.expectEqual(@as(usize, 16), merged.len());

    const out = try merged.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(i64, &[_]i64{ -11, -2, -1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10 }, out);
}

test "linked list merge two lists: empty and extreme" {
    var a = try initSortedLinkedList(testing.allocator, &[_]i64{});
    defer a.deinit();
    var b = try initSortedLinkedList(testing.allocator, &[_]i64{ 2, 1 });
    defer b.deinit();

    var merged_ab = try mergeLists(testing.allocator, &a, &b);
    defer merged_ab.deinit();

    const out_ab = try merged_ab.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out_ab);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2 }, out_ab);

    const n: usize = 10_000;
    var left = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(left);
    var right = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(right);

    for (0..n) |i| {
        left[i] = @intCast((n - i) * 2);
        right[i] = @intCast((n - i) * 2 - 1);
    }

    var l1 = try initSortedLinkedList(testing.allocator, left);
    defer l1.deinit();
    var l2 = try initSortedLinkedList(testing.allocator, right);
    defer l2.deinit();

    var merged = try mergeLists(testing.allocator, &l1, &l2);
    defer merged.deinit();
    try testing.expectEqual(@as(usize, 20_000), merged.len());

    const out = try merged.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(i64, 1), out[0]);
    try testing.expectEqual(@as(i64, 20_000), out[out.len - 1]);
}
