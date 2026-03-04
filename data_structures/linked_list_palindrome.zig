//! Linked List Palindrome - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/is_palindrome.py

const std = @import("std");
const testing = std.testing;

pub const ListNode = struct {
    val: i64,
    next: ?*ListNode,
};

pub const List = struct {
    allocator: std.mem.Allocator,
    head: ?*ListNode,
    tail: ?*ListNode,
    len_: usize,

    pub fn init(allocator: std.mem.Allocator) List {
        return .{ .allocator = allocator, .head = null, .tail = null, .len_ = 0 };
    }

    pub fn deinit(self: *List) void {
        var node_opt = self.head;
        while (node_opt) |node| {
            const next = node.next;
            self.allocator.destroy(node);
            node_opt = next;
        }
        self.* = undefined;
    }

    pub fn append(self: *List, value: i64) !void {
        const next_len = @addWithOverflow(self.len_, @as(usize, 1));
        if (next_len[1] != 0) return error.Overflow;

        const node = try self.allocator.create(ListNode);
        node.* = .{ .val = value, .next = null };

        if (self.tail) |tail_node| {
            tail_node.next = node;
        } else {
            self.head = node;
        }

        self.tail = node;
        self.len_ = next_len[0];
    }
};

pub fn listFromSlice(allocator: std.mem.Allocator, values: []const i64) !List {
    var list = List.init(allocator);
    for (values) |v| {
        try list.append(v);
    }
    return list;
}

/// Checks whether a linked list is palindrome by reversing second half.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isPalindrome(head: ?*ListNode) bool {
    if (head == null or head.?.next == null) return true;

    var slow = head.?;
    var fast = head.?.next;

    while (fast != null and fast.?.next != null) {
        fast = fast.?.next.?.next;
        slow = slow.next.?;
    }

    var second = slow.next;
    slow.next = null;

    var reversed: ?*ListNode = null;
    while (second) |node| {
        const next = node.next;
        node.next = reversed;
        reversed = node;
        second = next;
    }

    var p1 = head;
    var p2 = reversed;
    var ok = true;
    while (p2 != null and p1 != null) {
        if (p1.?.val != p2.?.val) {
            ok = false;
            break;
        }
        p1 = p1.?.next;
        p2 = p2.?.next;
    }

    // Restore list so tests can safely deinit all nodes without leaks.
    var restore_src = reversed;
    var restored: ?*ListNode = null;
    while (restore_src) |node| {
        const next = node.next;
        node.next = restored;
        restored = node;
        restore_src = next;
    }
    slow.next = restored;

    return ok;
}

/// Checks whether a linked list is palindrome using a stack.
/// Time complexity: O(n), Space complexity: O(n)
pub fn isPalindromeStack(allocator: std.mem.Allocator, head: ?*ListNode) !bool {
    if (head == null or head.?.next == null) return true;

    var slow = head;
    var fast = head;
    while (fast != null and fast.?.next != null) {
        fast = fast.?.next.?.next;
        slow = slow.?.next;
    }

    var stack = std.ArrayListUnmanaged(i64){};
    defer stack.deinit(allocator);

    var node_opt = slow;
    while (node_opt) |node| {
        try stack.append(allocator, node.val);
        node_opt = node.next;
    }

    var cur = head;
    while (stack.items.len > 0 and cur != null) {
        const top = stack.pop().?;
        if (top != cur.?.val) return false;
        cur = cur.?.next;
    }

    return true;
}

test "linked list palindrome: python examples" {
    try testing.expect(isPalindrome(null));

    var l1 = try listFromSlice(testing.allocator, &[_]i64{1});
    defer l1.deinit();
    try testing.expect(isPalindrome(l1.head));

    var l2 = try listFromSlice(testing.allocator, &[_]i64{ 1, 2 });
    defer l2.deinit();
    try testing.expect(!isPalindrome(l2.head));

    var l3 = try listFromSlice(testing.allocator, &[_]i64{ 1, 2, 1 });
    defer l3.deinit();
    try testing.expect(isPalindrome(l3.head));

    var l4 = try listFromSlice(testing.allocator, &[_]i64{ 1, 2, 2, 1 });
    defer l4.deinit();
    try testing.expect(isPalindrome(l4.head));
}

test "linked list palindrome: stack variant" {
    try testing.expect(try isPalindromeStack(testing.allocator, null));

    var l1 = try listFromSlice(testing.allocator, &[_]i64{ 1, 2, 1 });
    defer l1.deinit();
    try testing.expect(try isPalindromeStack(testing.allocator, l1.head));

    var l2 = try listFromSlice(testing.allocator, &[_]i64{ 1, 2, 3, 2, 1 });
    defer l2.deinit();
    try testing.expect(try isPalindromeStack(testing.allocator, l2.head));

    var l3 = try listFromSlice(testing.allocator, &[_]i64{ 1, 2, 3, 1 });
    defer l3.deinit();
    try testing.expect(!(try isPalindromeStack(testing.allocator, l3.head)));
}

test "linked list palindrome: extreme long list" {
    const half: usize = 10_000;
    var values = try testing.allocator.alloc(i64, half * 2 + 1);
    defer testing.allocator.free(values);

    for (0..half) |i| {
        values[i] = @intCast(i);
    }
    values[half] = 999_999;
    for (0..half) |i| {
        values[half + 1 + i] = @intCast(half - 1 - i);
    }

    var pal = try listFromSlice(testing.allocator, values);
    defer pal.deinit();
    try testing.expect(isPalindrome(pal.head));

    values[half + 3] = -1;
    var non_pal = try listFromSlice(testing.allocator, values);
    defer non_pal.deinit();
    try testing.expect(!isPalindrome(non_pal.head));
}
