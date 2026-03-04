//! Split String by Separator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/split.py

const std = @import("std");
const testing = std.testing;

/// Splits string by single-byte separator.
/// Returned slices reference `string`; caller owns only outer slice.
/// Time complexity: O(n), Space complexity: O(k)
pub fn split(allocator: std.mem.Allocator, string: []const u8, separator: u8) ![][]const u8 {
    var out = std.ArrayListUnmanaged([]const u8){};
    errdefer out.deinit(allocator);

    var last_index: usize = 0;
    for (string, 0..) |char, index| {
        if (char == separator) {
            try out.append(allocator, string[last_index..index]);
            last_index = index + 1;
        }
        if (index + 1 == string.len) {
            try out.append(allocator, string[last_index .. index + 1]);
        }
    }

    return try out.toOwnedSlice(allocator);
}

/// Splits string by space separator.
pub fn splitDefault(allocator: std.mem.Allocator, string: []const u8) ![][]const u8 {
    return split(allocator, string, ' ');
}

test "split: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try split(alloc, "apple#banana#cherry#orange", '#');
    defer alloc.free(s1);
    try testing.expectEqual(@as(usize, 4), s1.len);
    try testing.expectEqualStrings("apple", s1[0]);
    try testing.expectEqualStrings("banana", s1[1]);
    try testing.expectEqualStrings("cherry", s1[2]);
    try testing.expectEqualStrings("orange", s1[3]);

    const s2 = try splitDefault(alloc, "Hello there");
    defer alloc.free(s2);
    try testing.expectEqual(@as(usize, 2), s2.len);
    try testing.expectEqualStrings("Hello", s2[0]);
    try testing.expectEqualStrings("there", s2[1]);

    const s3 = try split(alloc, "11/22/63", '/');
    defer alloc.free(s3);
    try testing.expectEqualStrings("11", s3[0]);
    try testing.expectEqualStrings("22", s3[1]);
    try testing.expectEqualStrings("63", s3[2]);

    const s4 = try split(alloc, ";abbb;;c;", ';');
    defer alloc.free(s4);
    try testing.expectEqual(@as(usize, 5), s4.len);
    try testing.expectEqualStrings("", s4[0]);
    try testing.expectEqualStrings("abbb", s4[1]);
    try testing.expectEqualStrings("", s4[2]);
    try testing.expectEqualStrings("c", s4[3]);
    try testing.expectEqualStrings("", s4[4]);
}

test "split: edge and extreme cases" {
    const empty = try split(testing.allocator, "", ',');
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var long_input: [100_001]u8 = undefined;
    @memset(&long_input, 'a');
    long_input[50_000] = ',';
    const res = try split(testing.allocator, &long_input, ',');
    defer testing.allocator.free(res);
    try testing.expectEqual(@as(usize, 2), res.len);
}
