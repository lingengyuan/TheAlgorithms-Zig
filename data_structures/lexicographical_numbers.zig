//! Lexicographical Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/lexicographical_numbers.py

const std = @import("std");
const testing = std.testing;

/// Generates numbers from 1..max_number in lexicographical order.
/// Time complexity: O(n), Space complexity: O(n)
pub fn lexicalOrder(allocator: std.mem.Allocator, max_number: usize) ![]usize {
    if (max_number == 0) return allocator.alloc(usize, 0);

    var stack = std.ArrayListUnmanaged(usize){};
    defer stack.deinit(allocator);
    try stack.append(allocator, 1);

    var out = std.ArrayListUnmanaged(usize){};
    errdefer out.deinit(allocator);

    while (stack.items.len > 0) {
        const num = stack.pop().?;
        if (num > max_number) continue;

        try out.append(allocator, num);

        if ((num % 10) != 9) {
            const plus = @addWithOverflow(num, @as(usize, 1));
            if (plus[1] == 0) try stack.append(allocator, plus[0]);
        }

        const mul = @mulWithOverflow(num, @as(usize, 10));
        if (mul[1] == 0) try stack.append(allocator, mul[0]);
    }

    return try out.toOwnedSlice(allocator);
}

test "lexicographical numbers: python examples" {
    {
        const out = try lexicalOrder(testing.allocator, 13);
        defer testing.allocator.free(out);
        try testing.expectEqualSlices(usize, &[_]usize{ 1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9 }, out);
    }
    {
        const out = try lexicalOrder(testing.allocator, 1);
        defer testing.allocator.free(out);
        try testing.expectEqualSlices(usize, &[_]usize{1}, out);
    }
    {
        const out = try lexicalOrder(testing.allocator, 20);
        defer testing.allocator.free(out);
        try testing.expectEqualSlices(usize, &[_]usize{ 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 3, 4, 5, 6, 7, 8, 9 }, out);
    }
}

test "lexicographical numbers: empty and extreme" {
    const empty = try lexicalOrder(testing.allocator, 0);
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const n: usize = 100_000;
    const out = try lexicalOrder(testing.allocator, n);
    defer testing.allocator.free(out);

    try testing.expectEqual(n, out.len);
    try testing.expectEqual(@as(usize, 1), out[0]);
    try testing.expectEqual(@as(usize, 10), out[1]);
    try testing.expectEqual(@as(usize, 100), out[2]);
}
