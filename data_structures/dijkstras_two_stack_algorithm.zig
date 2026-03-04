//! Dijkstra's Two-Stack Algorithm - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/dijkstras_two_stack_algorithm.py

const std = @import("std");
const testing = std.testing;

fn apply(op: u8, a: f64, b: f64) !f64 {
    return switch (op) {
        '+' => a + b,
        '-' => a - b,
        '*' => a * b,
        '/' => if (b == 0) error.DivisionByZero else a / b,
        else => error.InvalidOperator,
    };
}

fn consumeOneOperation(operators: *std.ArrayListUnmanaged(u8), operands: *std.ArrayListUnmanaged(f64), allocator: std.mem.Allocator) !void {
    if (operators.items.len == 0) return error.InvalidExpression;
    if (operands.items.len < 2) return error.InvalidExpression;

    const op = operators.pop().?;
    const b = operands.pop().?;
    const a = operands.pop().?;
    const result = try apply(op, a, b);
    try operands.append(allocator, result);
}

/// Evaluates arithmetic expression using Dijkstra's two-stack algorithm.
/// Time complexity: O(n), Space complexity: O(n)
pub fn dijkstrasTwoStackAlgorithm(allocator: std.mem.Allocator, equation: []const u8) !f64 {
    var operands = std.ArrayListUnmanaged(f64){};
    defer operands.deinit(allocator);

    var operators = std.ArrayListUnmanaged(u8){};
    defer operators.deinit(allocator);

    var i: usize = 0;
    while (i < equation.len) {
        const ch = equation[i];

        if (std.ascii.isWhitespace(ch) or ch == '(') {
            i += 1;
            continue;
        }

        if (std.ascii.isDigit(ch)) {
            var value: f64 = 0;
            while (i < equation.len and std.ascii.isDigit(equation[i])) : (i += 1) {
                value = value * 10 + @as(f64, @floatFromInt(equation[i] - '0'));
            }
            try operands.append(allocator, value);
            continue;
        }

        if (ch == '+' or ch == '-' or ch == '*' or ch == '/') {
            try operators.append(allocator, ch);
            i += 1;
            continue;
        }

        if (ch == ')') {
            try consumeOneOperation(&operators, &operands, allocator);
            i += 1;
            continue;
        }

        return error.InvalidCharacter;
    }

    while (operators.items.len > 0) {
        try consumeOneOperation(&operators, &operands, allocator);
    }

    if (operands.items.len != 1) return error.InvalidExpression;
    return operands.items[0];
}

test "dijkstras two stack algorithm: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 8), try dijkstrasTwoStackAlgorithm(testing.allocator, "(5 + 3)"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 5), try dijkstrasTwoStackAlgorithm(testing.allocator, "((9 - (2 + 9)) + (8 - 1))"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -3), try dijkstrasTwoStackAlgorithm(testing.allocator, "((((3 - 2) - (2 + 3)) + (2 - 4)) + 3)"), 1e-9);
}

test "dijkstras two stack algorithm: invalid input" {
    try testing.expectError(error.InvalidExpression, dijkstrasTwoStackAlgorithm(testing.allocator, "(5 + )"));
    try testing.expectError(error.InvalidCharacter, dijkstrasTwoStackAlgorithm(testing.allocator, "(5 & 3)"));
    try testing.expectError(error.DivisionByZero, dijkstrasTwoStackAlgorithm(testing.allocator, "(8 / (2 - 2))"));
}

test "dijkstras two stack algorithm: extreme long addition chain" {
    const n: usize = 30_000;
    var expr = std.ArrayListUnmanaged(u8){};
    defer expr.deinit(testing.allocator);
    const writer = expr.writer(testing.allocator);

    try writer.writeByte('1');
    for (1..n) |_| {
        try writer.writeAll("+1");
    }

    const expression = try expr.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(expression);

    const result = try dijkstrasTwoStackAlgorithm(testing.allocator, expression);
    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(n)), result, 1e-6);
}
