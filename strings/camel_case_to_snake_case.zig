//! Camel Case to Snake Case - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/camel_case_to_snake_case.py

const std = @import("std");
const testing = std.testing;

/// Transforms camelCase or PascalCase string to snake_case.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn camelToSnakeCase(allocator: std.mem.Allocator, input_str: []const u8) ![]u8 {
    if (input_str.len == 0) return try allocator.alloc(u8, 0);

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (input_str, 0..) |char, index| {
        const has_prev = index > 0;
        const prev = if (has_prev) input_str[index - 1] else 0;

        if (std.ascii.isUpper(char)) {
            try out.append(allocator, '_');
            try out.append(allocator, std.ascii.toLower(char));
        } else if (has_prev and std.ascii.isDigit(prev) and std.ascii.isLower(char)) {
            try out.append(allocator, '_');
            try out.append(allocator, char);
        } else if (has_prev and std.ascii.isAlphabetic(prev) and std.ascii.isDigit(char)) {
            try out.append(allocator, '_');
            try out.append(allocator, char);
        } else if (!std.ascii.isAlphanumeric(char)) {
            try out.append(allocator, '_');
        } else {
            try out.append(allocator, char);
        }
    }

    if (out.items.len > 0 and out.items[0] == '_') {
        _ = out.orderedRemove(0);
    }
    return try out.toOwnedSlice(allocator);
}

test "camel to snake case: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try camelToSnakeCase(alloc, "someRandomString");
    defer alloc.free(s1);
    try testing.expectEqualStrings("some_random_string", s1);

    const s2 = try camelToSnakeCase(alloc, "SomeRandomStr#ng");
    defer alloc.free(s2);
    try testing.expectEqualStrings("some_random_str_ng", s2);

    const s3 = try camelToSnakeCase(alloc, "123someRandom123String123");
    defer alloc.free(s3);
    try testing.expectEqualStrings("123_some_random_123_string_123", s3);

    const s4 = try camelToSnakeCase(alloc, "123SomeRandom123String123");
    defer alloc.free(s4);
    try testing.expectEqualStrings("123_some_random_123_string_123", s4);
}

test "camel to snake case: edge and extreme cases" {
    const empty = try camelToSnakeCase(testing.allocator, "");
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const single = try camelToSnakeCase(testing.allocator, "A");
    defer testing.allocator.free(single);
    try testing.expectEqualStrings("a", single);

    var long_input = [_]u8{'A'} ** 20_000;
    const long_out = try camelToSnakeCase(testing.allocator, &long_input);
    defer testing.allocator.free(long_out);
    try testing.expect(long_out.len > long_input.len); // inserted underscores
}
