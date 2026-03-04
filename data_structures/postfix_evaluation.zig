//! Postfix Evaluation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/postfix_evaluation.py

const std = @import("std");
const testing = std.testing;

fn isOperator(token: []const u8) bool {
    return token.len == 1 and switch (token[0]) {
        '^', '*', '/', '+', '-' => true,
        else => false,
    };
}

fn applyBinaryOperator(op: u8, a: f64, b: f64) !f64 {
    return switch (op) {
        '^' => std.math.pow(f64, a, b),
        '*' => a * b,
        '/' => if (b == 0) error.DivisionByZero else a / b,
        '+' => a + b,
        '-' => a - b,
        else => error.InvalidOperator,
    };
}

/// Evaluates postfix expression token list.
/// Time complexity: O(n), Space complexity: O(n)
pub fn evaluatePostfix(allocator: std.mem.Allocator, post_fix: []const []const u8) !f64 {
    if (post_fix.len == 0) return 0;

    var stack = std.ArrayListUnmanaged(f64){};
    defer stack.deinit(allocator);

    for (post_fix) |token| {
        if (isOperator(token)) {
            const op = token[0];

            if ((op == '-' or op == '+') and stack.items.len < 2) {
                if (stack.items.len == 0) return error.InvalidExpression;
                var v = stack.pop().?;
                if (op == '-') v = -v;
                try stack.append(allocator, v);
                continue;
            }

            if (stack.items.len < 2) return error.InvalidExpression;
            const b = stack.pop().?;
            const a = stack.pop().?;
            try stack.append(allocator, try applyBinaryOperator(op, a, b));
        } else {
            const parsed = std.fmt.parseFloat(f64, token) catch return error.InvalidToken;
            try stack.append(allocator, parsed);
        }
    }

    if (stack.items.len != 1) return error.InvalidExpression;
    return stack.items[0];
}

test "postfix evaluation: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 0), try evaluatePostfix(testing.allocator, &[_][]const u8{"0"}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -1.1), try evaluatePostfix(testing.allocator, &[_][]const u8{"-1.1"}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 9), try evaluatePostfix(testing.allocator, &[_][]const u8{ "2", "1", "+", "3", "*" }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 11.7), try evaluatePostfix(testing.allocator, &[_][]const u8{ "2", "1.9", "+", "3", "*" }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 6.6), try evaluatePostfix(testing.allocator, &[_][]const u8{ "4", "13", "5", "/", "+" }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1), try evaluatePostfix(testing.allocator, &[_][]const u8{ "2", "-", "3", "+" }), 1e-9);
}

test "postfix evaluation: invalid input" {
    try testing.expectApproxEqAbs(@as(f64, 0), try evaluatePostfix(testing.allocator, &[_][]const u8{}), 1e-9);
    try testing.expectError(error.InvalidExpression, evaluatePostfix(testing.allocator, &[_][]const u8{ "4", "-", "6", "7", "/", "9", "8" }));
    try testing.expectError(error.InvalidToken, evaluatePostfix(testing.allocator, &[_][]const u8{ "3", "x", "+" }));
    try testing.expectError(error.DivisionByZero, evaluatePostfix(testing.allocator, &[_][]const u8{ "4", "0", "/" }));
}

test "postfix evaluation: extreme long chain" {
    const n: usize = 20_000;
    var tokens = std.ArrayListUnmanaged([]const u8){};
    defer tokens.deinit(testing.allocator);

    var storage = try testing.allocator.alloc([2]u8, n);
    defer testing.allocator.free(storage);

    // Build expression: 1 1 + 1 + 1 + ...
    storage[0] = .{ '1', 0 };
    try tokens.append(testing.allocator, storage[0][0..1]);

    var i: usize = 1;
    while (i < n) : (i += 1) {
        storage[i] = .{ '1', 0 };
        try tokens.append(testing.allocator, storage[i][0..1]);
        try tokens.append(testing.allocator, "+");
    }

    const value = try evaluatePostfix(testing.allocator, tokens.items);
    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(n)), value, 1e-6);
}
