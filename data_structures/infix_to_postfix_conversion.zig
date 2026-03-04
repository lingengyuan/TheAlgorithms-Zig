//! Infix To Postfix Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/infix_to_postfix_conversion.py

const std = @import("std");
const testing = std.testing;
const balanced = @import("balanced_parentheses.zig");

fn precedence(ch: u8) i32 {
    return switch (ch) {
        '+', '-' => 1,
        '*', '/' => 2,
        '^' => 3,
        else => -1,
    };
}

fn isRightAssociative(ch: u8) bool {
    return ch == '^';
}

fn isOperator(ch: u8) bool {
    return switch (ch) {
        '+', '-', '*', '/', '^' => true,
        else => false,
    };
}

fn appendToken(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, token: u8) !void {
    if (out.items.len > 0) try out.append(allocator, ' ');
    try out.append(allocator, token);
}

/// Converts infix expression to postfix (space-separated tokens).
/// Time complexity: O(n), Space complexity: O(n)
pub fn infixToPostfix(allocator: std.mem.Allocator, expression: []const u8) ![]u8 {
    if (expression.len == 0) return allocator.alloc(u8, 0);

    if (!(try balanced.balancedParentheses(allocator, expression))) {
        return error.MismatchedParentheses;
    }

    var stack = std.ArrayListUnmanaged(u8){};
    defer stack.deinit(allocator);

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (expression) |ch| {
        if (std.ascii.isWhitespace(ch)) continue;

        if (std.ascii.isAlphanumeric(ch)) {
            try appendToken(&out, allocator, ch);
            continue;
        }

        if (ch == '(') {
            try stack.append(allocator, ch);
            continue;
        }

        if (ch == ')') {
            while (stack.items.len > 0 and stack.items[stack.items.len - 1] != '(') {
                try appendToken(&out, allocator, stack.pop().?);
            }
            if (stack.items.len == 0) return error.MismatchedParentheses;
            _ = stack.pop();
            continue;
        }

        if (!isOperator(ch)) return error.InvalidCharacter;

        while (stack.items.len > 0) {
            const top = stack.items[stack.items.len - 1];
            if (top == '(') break;

            const char_prec = precedence(ch);
            const top_prec = precedence(top);

            if (char_prec > top_prec) break;
            if (char_prec == top_prec and isRightAssociative(ch)) break;

            try appendToken(&out, allocator, stack.pop().?);
        }

        try stack.append(allocator, ch);
    }

    while (stack.items.len > 0) {
        const top = stack.pop().?;
        if (top == '(') return error.MismatchedParentheses;
        try appendToken(&out, allocator, top);
    }

    return try out.toOwnedSlice(allocator);
}

test "infix to postfix conversion: python examples" {
    const e1 = try infixToPostfix(testing.allocator, "3+2");
    defer testing.allocator.free(e1);
    try testing.expectEqualStrings("3 2 +", e1);

    const e2 = try infixToPostfix(testing.allocator, "(3+4)*5-6");
    defer testing.allocator.free(e2);
    try testing.expectEqualStrings("3 4 + 5 * 6 -", e2);

    const e3 = try infixToPostfix(testing.allocator, "(1+2)*3/4-5");
    defer testing.allocator.free(e3);
    try testing.expectEqualStrings("1 2 + 3 * 4 / 5 -", e3);

    const e4 = try infixToPostfix(testing.allocator, "a+b*c+(d*e+f)*g");
    defer testing.allocator.free(e4);
    try testing.expectEqualStrings("a b c * + d e * f + g * +", e4);

    const e5 = try infixToPostfix(testing.allocator, "x^y/(5*z)+2");
    defer testing.allocator.free(e5);
    try testing.expectEqualStrings("x y ^ 5 z * / 2 +", e5);

    const e6 = try infixToPostfix(testing.allocator, "2^3^2");
    defer testing.allocator.free(e6);
    try testing.expectEqualStrings("2 3 2 ^ ^", e6);
}

test "infix to postfix conversion: invalid and empty" {
    const empty = try infixToPostfix(testing.allocator, "");
    defer testing.allocator.free(empty);
    try testing.expectEqualStrings("", empty);

    try testing.expectError(error.MismatchedParentheses, infixToPostfix(testing.allocator, "(1*(2+3)+4))"));
    try testing.expectError(error.InvalidCharacter, infixToPostfix(testing.allocator, "a+b@c"));
}

test "infix to postfix conversion: extreme long expression" {
    const n: usize = 30_000;
    var expr = std.ArrayListUnmanaged(u8){};
    defer expr.deinit(testing.allocator);
    const w = expr.writer(testing.allocator);

    try w.writeByte('a');
    for (1..n) |_| {
        try w.writeAll("+a");
    }

    const infix = try expr.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(infix);

    const postfix = try infixToPostfix(testing.allocator, infix);
    defer testing.allocator.free(postfix);

    try testing.expect(postfix.len > infix.len / 2);
    try testing.expect(std.mem.startsWith(u8, postfix, "a a +"));
}
