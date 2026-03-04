//! Snake Case to Camel/Pascal Case - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/snake_case_to_camel_pascal_case.py

const std = @import("std");
const testing = std.testing;

/// Converts snake_case to camelCase or PascalCase.
/// Empty segments (e.g. consecutive underscores) are skipped in capitalization.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn snakeToCamelCase(
    allocator: std.mem.Allocator,
    inputStr: []const u8,
    usePascal: bool,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var segment_start: usize = 0;
    var segment_index: usize = 0;
    var i: usize = 0;

    while (i <= inputStr.len) : (i += 1) {
        if (i == inputStr.len or inputStr[i] == '_') {
            const segment = inputStr[segment_start..i];
            const should_capitalize = usePascal or segment_index > 0;

            if (should_capitalize) {
                if (segment.len > 0) {
                    try out.append(allocator, std.ascii.toUpper(segment[0]));
                    if (segment.len > 1) try out.appendSlice(allocator, segment[1..]);
                }
            } else {
                try out.appendSlice(allocator, segment);
            }

            segment_index += 1;
            segment_start = i + 1;
        }
    }

    return try out.toOwnedSlice(allocator);
}

test "snake to camel case: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try snakeToCamelCase(alloc, "some_random_string", false);
    defer alloc.free(s1);
    try testing.expectEqualStrings("someRandomString", s1);

    const s2 = try snakeToCamelCase(alloc, "some_random_string", true);
    defer alloc.free(s2);
    try testing.expectEqualStrings("SomeRandomString", s2);

    const s3 = try snakeToCamelCase(alloc, "some_random_string_with_numbers_123", false);
    defer alloc.free(s3);
    try testing.expectEqualStrings("someRandomStringWithNumbers123", s3);

    const s4 = try snakeToCamelCase(alloc, "some_random_string_with_numbers_123", true);
    defer alloc.free(s4);
    try testing.expectEqualStrings("SomeRandomStringWithNumbers123", s4);
}

test "snake to camel case: boundary behavior" {
    const alloc = testing.allocator;

    const empty = try snakeToCamelCase(alloc, "", false);
    defer alloc.free(empty);
    try testing.expectEqualStrings("", empty);

    const leading = try snakeToCamelCase(alloc, "_some_string", false);
    defer alloc.free(leading);
    try testing.expectEqualStrings("SomeString", leading);

    const trailing = try snakeToCamelCase(alloc, "some_string_", false);
    defer alloc.free(trailing);
    try testing.expectEqualStrings("someString", trailing);

    const doubled = try snakeToCamelCase(alloc, "some__string", false);
    defer alloc.free(doubled);
    try testing.expectEqualStrings("someString", doubled);
}

test "snake to camel case: extreme long input" {
    const alloc = testing.allocator;

    var builder = std.ArrayListUnmanaged(u8){};
    defer builder.deinit(alloc);

    try builder.appendSlice(alloc, "part");
    for (0..60_000) |_| {
        try builder.appendSlice(alloc, "_x");
    }

    const out = try snakeToCamelCase(alloc, builder.items, false);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 60_004), out.len);
    try testing.expect(std.mem.startsWith(u8, out, "partX"));
    try testing.expect(out[out.len - 1] == 'X');
}
