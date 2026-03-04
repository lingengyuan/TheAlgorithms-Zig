//! Balanced Parentheses - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/balanced_parentheses.py

const std = @import("std");
const testing = std.testing;

/// Returns whether parentheses/brackets/braces in `input` are balanced.
/// Time complexity: O(n), Space complexity: O(n)
pub fn balancedParentheses(allocator: std.mem.Allocator, input: []const u8) !bool {
    var stack = std.ArrayListUnmanaged(u8){};
    defer stack.deinit(allocator);

    for (input) |ch| {
        switch (ch) {
            '(', '[', '{' => try stack.append(allocator, ch),
            ')', ']', '}' => {
                if (stack.items.len == 0) return false;
                const top = stack.pop().?;
                if ((top == '(' and ch != ')') or (top == '[' and ch != ']') or (top == '{' and ch != '}')) {
                    return false;
                }
            },
            else => {},
        }
    }

    return stack.items.len == 0;
}

test "balanced parentheses: python examples" {
    try testing.expect(try balancedParentheses(testing.allocator, "([]{})"));
    try testing.expect(try balancedParentheses(testing.allocator, "[()]{}{[()()]()}"));
    try testing.expect(!(try balancedParentheses(testing.allocator, "[(])")));
    try testing.expect(try balancedParentheses(testing.allocator, "1+2*3-4"));
    try testing.expect(try balancedParentheses(testing.allocator, ""));
}

test "balanced parentheses: boundary and extreme" {
    try testing.expect(!(try balancedParentheses(testing.allocator, ")")));
    try testing.expect(!(try balancedParentheses(testing.allocator, "(")));

    const n: usize = 100_000;
    var buf = try testing.allocator.alloc(u8, n * 2);
    defer testing.allocator.free(buf);

    for (0..n) |i| {
        buf[i] = '(';
        buf[n + i] = ')';
    }
    try testing.expect(try balancedParentheses(testing.allocator, buf));

    buf[n * 2 - 1] = ']';
    try testing.expect(!(try balancedParentheses(testing.allocator, buf)));
}
