//! Prefix Evaluation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/prefix_evaluation.py

const std = @import("std");
const testing = std.testing;

fn isOperator(token: []const u8) bool {
    return token.len == 1 and switch (token[0]) {
        '+', '-', '*', '/' => true,
        else => false,
    };
}

fn applyOp(op: u8, x: f64, y: f64) !f64 {
    return switch (op) {
        '+' => x + y,
        '-' => x - y,
        '*' => x * y,
        '/' => if (y == 0) error.DivisionByZero else x / y,
        else => error.InvalidOperator,
    };
}

fn parseTokens(allocator: std.mem.Allocator, expression: []const u8) !std.ArrayListUnmanaged([]const u8) {
    var tokens = std.ArrayListUnmanaged([]const u8){};
    errdefer tokens.deinit(allocator);

    var it = std.mem.tokenizeAny(u8, expression, " \t\n\r");
    while (it.next()) |token| {
        try tokens.append(allocator, token);
    }

    return tokens;
}

/// Evaluates prefix expression using stack.
/// Time complexity: O(n), Space complexity: O(n)
pub fn evaluatePrefix(allocator: std.mem.Allocator, expression: []const u8) !f64 {
    var tokens = try parseTokens(allocator, expression);
    defer tokens.deinit(allocator);

    if (tokens.items.len == 0) return error.InvalidExpression;

    var stack = std.ArrayListUnmanaged(f64){};
    defer stack.deinit(allocator);

    var i = tokens.items.len;
    while (i > 0) {
        i -= 1;
        const token = tokens.items[i];
        if (isOperator(token)) {
            if (stack.items.len < 2) return error.InvalidExpression;
            const o1 = stack.pop().?;
            const o2 = stack.pop().?;
            try stack.append(allocator, try applyOp(token[0], o1, o2));
        } else {
            const value = std.fmt.parseFloat(f64, token) catch return error.InvalidToken;
            try stack.append(allocator, value);
        }
    }

    if (stack.items.len != 1) return error.InvalidExpression;
    return stack.items[0];
}

fn evalRecursive(tokens: []const []const u8, idx: *usize) !f64 {
    if (idx.* >= tokens.len) return error.InvalidExpression;

    const token = tokens[idx.*];
    idx.* += 1;

    if (!isOperator(token)) {
        return std.fmt.parseFloat(f64, token) catch error.InvalidToken;
    }

    const a = try evalRecursive(tokens, idx);
    const b = try evalRecursive(tokens, idx);
    return applyOp(token[0], a, b);
}

/// Evaluates prefix expression recursively.
/// Time complexity: O(n), Space complexity: O(n)
pub fn evaluatePrefixRecursive(allocator: std.mem.Allocator, expression: []const u8) !f64 {
    var tokens = try parseTokens(allocator, expression);
    defer tokens.deinit(allocator);

    var idx: usize = 0;
    const value = try evalRecursive(tokens.items, &idx);
    if (idx != tokens.items.len) return error.InvalidExpression;
    return value;
}

test "prefix evaluation: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 21), try evaluatePrefix(testing.allocator, "+ 9 * 2 6"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4), try evaluatePrefix(testing.allocator, "/ * 10 2 + 4 1"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2), try evaluatePrefix(testing.allocator, "2"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 8), try evaluatePrefix(testing.allocator, "+ * 2 3 / 8 4"), 1e-9);
}

test "prefix evaluation: recursive examples" {
    try testing.expectApproxEqAbs(@as(f64, 2), try evaluatePrefixRecursive(testing.allocator, "2"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 8), try evaluatePrefixRecursive(testing.allocator, "+ * 2 3 / 8 4"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 21), try evaluatePrefixRecursive(testing.allocator, "+ 9 * 2 6"), 1e-9);
}

test "prefix evaluation: invalid and extreme" {
    try testing.expectError(error.InvalidExpression, evaluatePrefix(testing.allocator, ""));
    try testing.expectError(error.InvalidExpression, evaluatePrefix(testing.allocator, "+ 1"));
    try testing.expectError(error.InvalidToken, evaluatePrefix(testing.allocator, "+ 1 x"));

    // Build: + 1 + 1 + 1 ...  (right-skewed)
    const n: usize = 20_000;
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(testing.allocator);
    const w = out.writer(testing.allocator);

    var i: usize = 1;
    while (i < n) : (i += 1) {
        try w.writeAll("+ ");
    }
    var j: usize = 0;
    while (j < n) : (j += 1) {
        try w.writeAll("1 ");
    }

    const expr = try out.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(expr);

    const value = try evaluatePrefix(testing.allocator, expr);
    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(n)), value, 1e-6);
}
