//! Generate Parentheses (Iterative) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/generate_parentheses_iterative.py

const std = @import("std");
const testing = std.testing;

const State = struct {
    current: []u8,
    open_count: usize,
    close_count: usize,
};

pub fn freeParenthesesList(allocator: std.mem.Allocator, list: [][]u8) void {
    for (list) |item| allocator.free(item);
    allocator.free(list);
}

/// Generates all valid parentheses strings of size `length` using an explicit stack.
/// Output order matches the Python iterative reference implementation.
///
/// Time complexity: O(2^(2n))
/// Space complexity: O(2^(2n))
pub fn generateParenthesesIterative(
    allocator: std.mem.Allocator,
    length: usize,
) std.mem.Allocator.Error![][]u8 {
    var result = std.ArrayListUnmanaged([]u8){};
    errdefer {
        for (result.items) |item| allocator.free(item);
        result.deinit(allocator);
    }

    var stack = std.ArrayListUnmanaged(State){};
    defer {
        for (stack.items) |state| allocator.free(state.current);
        stack.deinit(allocator);
    }

    try stack.append(allocator, .{
        .current = try allocator.dupe(u8, ""),
        .open_count = 0,
        .close_count = 0,
    });

    while (stack.pop()) |state| {
        var keep_current = false;
        if (state.current.len == 2 * length) {
            try result.append(allocator, state.current);
            keep_current = true;
        }

        if (state.open_count < length) {
            const next_open = try std.fmt.allocPrint(allocator, "{s}(", .{state.current});
            try stack.append(allocator, .{
                .current = next_open,
                .open_count = state.open_count + 1,
                .close_count = state.close_count,
            });
        }

        if (state.close_count < state.open_count) {
            const next_close = try std.fmt.allocPrint(allocator, "{s})", .{state.current});
            try stack.append(allocator, .{
                .current = next_close,
                .open_count = state.open_count,
                .close_count = state.close_count + 1,
            });
        }

        if (!keep_current) allocator.free(state.current);
    }

    return result.toOwnedSlice(allocator);
}

test "generate parentheses iterative: python examples" {
    const alloc = testing.allocator;

    const r3 = try generateParenthesesIterative(alloc, 3);
    defer freeParenthesesList(alloc, r3);
    const expected3 = [_][]const u8{ "()()()", "()(())", "(())()", "(()())", "((()))" };
    try testing.expectEqual(expected3.len, r3.len);
    for (expected3, 0..) |item, i| try testing.expectEqualStrings(item, r3[i]);

    const r2 = try generateParenthesesIterative(alloc, 2);
    defer freeParenthesesList(alloc, r2);
    const expected2 = [_][]const u8{ "()()", "(())" };
    try testing.expectEqual(expected2.len, r2.len);
    for (expected2, 0..) |item, i| try testing.expectEqualStrings(item, r2[i]);

    const r1 = try generateParenthesesIterative(alloc, 1);
    defer freeParenthesesList(alloc, r1);
    try testing.expectEqual(@as(usize, 1), r1.len);
    try testing.expectEqualStrings("()", r1[0]);

    const r0 = try generateParenthesesIterative(alloc, 0);
    defer freeParenthesesList(alloc, r0);
    try testing.expectEqual(@as(usize, 1), r0.len);
    try testing.expectEqualStrings("", r0[0]);
}

test "generate parentheses iterative: boundary and extreme size" {
    const alloc = testing.allocator;

    const r4 = try generateParenthesesIterative(alloc, 4);
    defer freeParenthesesList(alloc, r4);
    try testing.expectEqual(@as(usize, 14), r4.len); // Catalan(4)

    const r6 = try generateParenthesesIterative(alloc, 6);
    defer freeParenthesesList(alloc, r6);
    try testing.expectEqual(@as(usize, 132), r6.len); // Catalan(6)
}
