//! Project Euler Problem 55: Lychrel Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_055/sol1.py

const std = @import("std");
const testing = std.testing;

fn isPalindromeText(text: []const u8) bool {
    var left: usize = 0;
    var right: usize = text.len;
    while (left < right) {
        right -= 1;
        if (text[left] != text[right]) return false;
        left += 1;
    }
    return true;
}

fn addDecimalStrings(allocator: std.mem.Allocator, left: []const u8, right: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var i: usize = left.len;
    var j: usize = right.len;
    var carry: u8 = 0;

    while (i > 0 or j > 0 or carry != 0) {
        const digit_left: u8 = if (i > 0) left[i - 1] - '0' else 0;
        const digit_right: u8 = if (j > 0) right[j - 1] - '0' else 0;
        const sum = digit_left + digit_right + carry;
        try out.append(allocator, '0' + @as(u8, sum % 10));
        carry = sum / 10;
        if (i > 0) i -= 1;
        if (j > 0) j -= 1;
    }

    std.mem.reverse(u8, out.items);
    return try out.toOwnedSlice(allocator);
}

fn sumReverseText(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const reversed = try allocator.alloc(u8, text.len);
    defer allocator.free(reversed);
    for (text, 0..) |char, idx| reversed[text.len - 1 - idx] = char;
    return try addDecimalStrings(allocator, text, reversed);
}

pub fn isPalindrome(n: u64) bool {
    var buffer: [32]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{}", .{n}) catch unreachable;
    return isPalindromeText(text);
}

pub fn sumReverse(n: u64) u64 {
    var text_buffer: [32]u8 = undefined;
    const text = std.fmt.bufPrint(&text_buffer, "{}", .{n}) catch unreachable;

    var reversed_buffer: [32]u8 = undefined;
    for (text, 0..) |char, idx| reversed_buffer[text.len - 1 - idx] = char;
    const reversed = std.fmt.parseInt(u64, reversed_buffer[0..text.len], 10) catch unreachable;
    return n + reversed;
}

/// Returns the count of all Lychrel numbers below `limit`.
/// Time complexity: O(limit * iterations * digits), Space complexity: O(digits)
pub fn solution(allocator: std.mem.Allocator, limit: usize) !usize {
    var count: usize = 0;
    for (1..limit) |start| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var current = try std.fmt.allocPrint(a, "{}", .{start});
        var iterations: usize = 0;
        while (iterations < 50) : (iterations += 1) {
            current = try sumReverseText(a, current);
            if (isPalindromeText(current)) break;
        } else {
            count += 1;
        }
    }
    return count;
}

test "problem 055: helper examples" {
    try testing.expect(!isPalindrome(12_567_321));
    try testing.expect(isPalindrome(1221));
    try testing.expect(isPalindrome(9_876_789));

    try testing.expectEqual(@as(u64, 444), sumReverse(123));
    try testing.expectEqual(@as(u64, 12_221), sumReverse(3478));
    try testing.expectEqual(@as(u64, 33), sumReverse(12));
}

test "problem 055: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 13), try solution(alloc, 1000));
    try testing.expectEqual(@as(usize, 76), try solution(alloc, 5000));
    try testing.expectEqual(@as(usize, 249), try solution(alloc, 10_000));
}

test "problem 055: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try solution(alloc, 1));
    try testing.expectEqual(@as(usize, 0), try solution(alloc, 10));

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const sum = try sumReverseText(arena.allocator(), "9");
    try testing.expectEqualStrings("18", sum);
}
