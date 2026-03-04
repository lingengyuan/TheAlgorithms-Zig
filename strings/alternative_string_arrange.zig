//! Alternative String Arrange - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/alternative_string_arrange.py

const std = @import("std");
const testing = std.testing;

/// Alternately merges two strings, appending any remaining suffix.
/// Caller owns returned slice.
/// Time complexity: O(n + m), Space complexity: O(n + m)
pub fn alternativeStringArrange(
    allocator: std.mem.Allocator,
    firstStr: []const u8,
    secondStr: []const u8,
) ![]u8 {
    const out = try allocator.alloc(u8, firstStr.len + secondStr.len);
    const max_len = @max(firstStr.len, secondStr.len);

    var write_index: usize = 0;
    for (0..max_len) |i| {
        if (i < firstStr.len) {
            out[write_index] = firstStr[i];
            write_index += 1;
        }
        if (i < secondStr.len) {
            out[write_index] = secondStr[i];
            write_index += 1;
        }
    }
    return out;
}

test "alternative string arrange: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try alternativeStringArrange(alloc, "ABCD", "XY");
    defer alloc.free(s1);
    try testing.expectEqualStrings("AXBYCD", s1);

    const s2 = try alternativeStringArrange(alloc, "XY", "ABCD");
    defer alloc.free(s2);
    try testing.expectEqualStrings("XAYBCD", s2);

    const s3 = try alternativeStringArrange(alloc, "AB", "XYZ");
    defer alloc.free(s3);
    try testing.expectEqualStrings("AXBYZ", s3);

    const s4 = try alternativeStringArrange(alloc, "ABC", "");
    defer alloc.free(s4);
    try testing.expectEqualStrings("ABC", s4);
}

test "alternative string arrange: edge and extreme cases" {
    const alloc = testing.allocator;

    const empty = try alternativeStringArrange(alloc, "", "");
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var first = [_]u8{'A'} ** 120_000;
    var second = [_]u8{'b'} ** 120_000;
    const merged = try alternativeStringArrange(alloc, &first, &second);
    defer alloc.free(merged);

    try testing.expectEqual(@as(usize, 240_000), merged.len);
    try testing.expect(merged[0] == 'A');
    try testing.expect(merged[1] == 'b');
    try testing.expect(merged[merged.len - 2] == 'A');
    try testing.expect(merged[merged.len - 1] == 'b');
}
