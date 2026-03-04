//! Prefix Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/prefix_function.py

const std = @import("std");
const testing = std.testing;

/// Computes KMP prefix-function values for each prefix of input string.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn prefixFunction(allocator: std.mem.Allocator, inputString: []const u8) ![]usize {
    const prefix_result = try allocator.alloc(usize, inputString.len);
    if (inputString.len == 0) return prefix_result;

    prefix_result[0] = 0;
    for (1..inputString.len) |i| {
        var j = prefix_result[i - 1];
        while (j > 0 and inputString[i] != inputString[j]) {
            j = prefix_result[j - 1];
        }
        if (inputString[i] == inputString[j]) j += 1;
        prefix_result[i] = j;
    }

    return prefix_result;
}

/// Returns the maximum prefix-function value.
/// Time complexity: O(n), Space complexity: O(n)
pub fn longestPrefix(allocator: std.mem.Allocator, inputString: []const u8) !usize {
    if (inputString.len == 0) return 0;

    const prefix = try prefixFunction(allocator, inputString);
    defer allocator.free(prefix);

    var max_value: usize = 0;
    for (prefix) |v| {
        if (v > max_value) max_value = v;
    }
    return max_value;
}

test "prefix function: python reference examples" {
    const alloc = testing.allocator;

    const p1 = try prefixFunction(alloc, "aabcdaabc");
    defer alloc.free(p1);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 0, 0, 0, 1, 2, 3, 4 }, p1);

    const p2 = try prefixFunction(alloc, "asdasdad");
    defer alloc.free(p2);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 0, 0, 1, 2, 3, 4, 0 }, p2);

    try testing.expectEqual(@as(usize, 4), try longestPrefix(alloc, "aabcdaabc"));
    try testing.expectEqual(@as(usize, 4), try longestPrefix(alloc, "asdasdad"));
    try testing.expectEqual(@as(usize, 2), try longestPrefix(alloc, "abcab"));
}

test "prefix function: edge and extreme cases" {
    const alloc = testing.allocator;

    const empty = try prefixFunction(alloc, "");
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
    try testing.expectEqual(@as(usize, 0), try longestPrefix(alloc, ""));

    var repeated = [_]u8{'a'} ** 200_000;
    const pref = try prefixFunction(alloc, &repeated);
    defer alloc.free(pref);
    try testing.expectEqual(@as(usize, 199_999), pref[pref.len - 1]);
    try testing.expectEqual(@as(usize, 199_999), try longestPrefix(alloc, &repeated));
}
