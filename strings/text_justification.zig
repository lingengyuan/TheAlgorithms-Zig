//! Text Justification - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/text_justification.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const LineList = []const []u8;

/// Fully justifies text to `max_width`, matching the Python reference behavior.
/// Time complexity: O(m * n), Space complexity: O(m * n)
pub fn textJustification(allocator: Allocator, text: []const u8, max_width: usize) !LineList {
    var tokens = std.ArrayListUnmanaged([]const u8){};
    defer tokens.deinit(allocator);

    var tokenizer = std.mem.tokenizeScalar(u8, text, ' ');
    while (tokenizer.next()) |word| {
        try tokens.append(allocator, word);
    }
    if (tokens.items.len == 0) return allocator.alloc([]u8, 0);

    var answer = std.ArrayListUnmanaged([]u8){};
    errdefer freeLines(allocator, answer.items);

    var line = std.ArrayListUnmanaged([]const u8){};
    defer line.deinit(allocator);
    var width: usize = 0;

    for (tokens.items) |word| {
        if (width + word.len + line.items.len <= max_width) {
            try line.append(allocator, word);
            width += word.len;
        } else {
            try answer.append(allocator, try justifyLine(allocator, line.items, width, max_width));
            line.clearRetainingCapacity();
            try line.append(allocator, word);
            width = word.len;
        }
    }

    const gaps = line.items.len - 1;
    const remaining_spaces = max_width - width - gaps;
    var last = std.ArrayListUnmanaged(u8){};
    defer last.deinit(allocator);
    for (line.items, 0..) |word, index| {
        if (index > 0) try last.append(allocator, ' ');
        try last.appendSlice(allocator, word);
    }
    for (0..remaining_spaces) |_| try last.append(allocator, ' ');
    try answer.append(allocator, try last.toOwnedSlice(allocator));

    return answer.toOwnedSlice(allocator);
}

fn justifyLine(allocator: Allocator, line: []const []const u8, width: usize, max_width: usize) ![]u8 {
    const overall_spaces_count = max_width - width;
    if (line.len == 1) {
        var out = std.ArrayListUnmanaged(u8){};
        errdefer out.deinit(allocator);
        try out.appendSlice(allocator, line[0]);
        for (0..overall_spaces_count) |_| try out.append(allocator, ' ');
        return out.toOwnedSlice(allocator);
    }

    const gaps = line.len - 1;
    const base_spaces = overall_spaces_count / gaps;
    const extra_spaces = overall_spaces_count % gaps;

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);
    for (line, 0..) |word, index| {
        try out.appendSlice(allocator, word);
        if (index < gaps) {
            const spaces_here = base_spaces + @as(usize, if (index < extra_spaces) 1 else 0);
            for (0..spaces_here) |_| try out.append(allocator, ' ');
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn freeLines(allocator: Allocator, lines: LineList) void {
    for (lines) |line| allocator.free(line);
    allocator.free(lines);
}

test "text justification: python samples" {
    const one = try textJustification(testing.allocator, "This is an example of text justification.", 16);
    defer freeLines(testing.allocator, one);
    try testing.expectEqual(@as(usize, 3), one.len);
    try testing.expectEqualStrings("This    is    an", one[0]);
    try testing.expectEqualStrings("example  of text", one[1]);
    try testing.expectEqualStrings("justification.  ", one[2]);

    const two = try textJustification(testing.allocator, "Two roads diverged in a yellow wood", 16);
    defer freeLines(testing.allocator, two);
    try testing.expectEqualStrings("Two        roads", two[0]);
    try testing.expectEqualStrings("diverged   in  a", two[1]);
    try testing.expectEqualStrings("yellow wood     ", two[2]);
}

test "text justification: exact fit and empty input" {
    const exact = try textJustification(testing.allocator, "ab cd", 5);
    defer freeLines(testing.allocator, exact);
    try testing.expectEqual(@as(usize, 1), exact.len);
    try testing.expectEqualStrings("ab cd", exact[0]);

    const empty = try textJustification(testing.allocator, "", 5);
    defer freeLines(testing.allocator, empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}
