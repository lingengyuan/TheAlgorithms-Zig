//! Infix To Prefix Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/infix_to_prefix_conversion.py

const std = @import("std");
const testing = std.testing;
const postfix_mod = @import("infix_to_postfix_conversion.zig");

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

fn reverseAndSwapParentheses(allocator: std.mem.Allocator, infix: []const u8) ![]u8 {
    const reversed = try allocator.alloc(u8, infix.len);

    var out_idx: usize = 0;
    var i = infix.len;
    while (i > 0) {
        i -= 1;
        const ch = infix[i];
        reversed[out_idx] = switch (ch) {
            '(' => ')',
            ')' => '(',
            else => ch,
        };
        out_idx += 1;
    }

    return reversed;
}

fn postfixToPrefixNoSpaces(allocator: std.mem.Allocator, postfix: []const u8) ![]u8 {
    var tokens = std.ArrayListUnmanaged([]const u8){};
    defer tokens.deinit(allocator);

    var it = std.mem.tokenizeAny(u8, postfix, " \t\n\r");
    while (it.next()) |token| {
        try tokens.append(allocator, token);
    }

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var i = tokens.items.len;
    while (i > 0) {
        i -= 1;
        try out.appendSlice(allocator, tokens.items[i]);
    }

    return try out.toOwnedSlice(allocator);
}

fn infixToPostfixFlippedAssociativity(allocator: std.mem.Allocator, expression: []const u8) ![]u8 {
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
            const flipped_right_assoc = !isRightAssociative(ch);

            if (char_prec > top_prec) break;
            if (char_prec == top_prec and flipped_right_assoc) break;

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

/// Converts infix to prefix notation.
/// Time complexity: O(n), Space complexity: O(n)
pub fn infixToPrefix(allocator: std.mem.Allocator, infix: []const u8) ![]u8 {
    if (infix.len == 0) return allocator.alloc(u8, 0);

    const reversed = try reverseAndSwapParentheses(allocator, infix);
    defer allocator.free(reversed);

    const postfix = try infixToPostfixFlippedAssociativity(allocator, reversed);
    defer allocator.free(postfix);

    return postfixToPrefixNoSpaces(allocator, postfix);
}

test "infix to prefix conversion: python examples" {
    const p1 = try infixToPrefix(testing.allocator, "a+b^c");
    defer testing.allocator.free(p1);
    try testing.expectEqualStrings("+a^bc", p1);

    const p2 = try infixToPrefix(testing.allocator, "1*((-a)*2+b)");
    defer testing.allocator.free(p2);
    try testing.expectEqualStrings("*1+*-a2b", p2);

    const p3 = try infixToPrefix(testing.allocator, "");
    defer testing.allocator.free(p3);
    try testing.expectEqualStrings("", p3);
}

test "infix to prefix conversion: invalid expression" {
    try testing.expectError(error.MismatchedParentheses, infixToPrefix(testing.allocator, "(()"));
    try testing.expectError(error.MismatchedParentheses, infixToPrefix(testing.allocator, "())"));
}

test "infix to prefix conversion: extreme long expression" {
    const n: usize = 20_000;
    var expr = std.ArrayListUnmanaged(u8){};
    defer expr.deinit(testing.allocator);
    const w = expr.writer(testing.allocator);

    try w.writeByte('a');
    for (1..n) |_| {
        try w.writeAll("+a");
    }

    const infix = try expr.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(infix);

    const prefix = try infixToPrefix(testing.allocator, infix);
    defer testing.allocator.free(prefix);

    try testing.expect(prefix.len > n);
    try testing.expect(prefix[0] == '+');
}

test "infix to prefix conversion: repeated left associative operators" {
    const prefix = try infixToPrefix(testing.allocator, "a-b-c");
    defer testing.allocator.free(prefix);
    try testing.expectEqualStrings("--abc", prefix);
}
