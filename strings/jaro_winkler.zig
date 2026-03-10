//! Jaro-Winkler Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/jaro_winkler.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the Jaro-Winkler similarity between two strings.
/// Time complexity: O(n²), Space complexity: O(n)
pub fn jaroWinkler(allocator: Allocator, str1: []const u8, str2: []const u8) !f64 {
    const matching_1 = try getMatchedCharacters(allocator, str1, str2);
    defer allocator.free(matching_1);
    const matching_2 = try getMatchedCharacters(allocator, str2, str1);
    defer allocator.free(matching_2);

    const match_count = matching_1.len;
    var transpositions: usize = 0;
    for (matching_1, 0..) |char, index| {
        if (char != matching_2[index]) transpositions += 1;
    }
    transpositions /= 2;

    var jaro: f64 = 0.0;
    if (match_count != 0) {
        jaro = (1.0 / 3.0) * (@as(f64, @floatFromInt(match_count)) / @as(f64, @floatFromInt(str1.len)) +
            @as(f64, @floatFromInt(match_count)) / @as(f64, @floatFromInt(str2.len)) +
            @as(f64, @floatFromInt(match_count - transpositions)) / @as(f64, @floatFromInt(match_count)));
    }

    var prefix_len: usize = 0;
    const max_prefix = @min(@min(str1.len, str2.len), @as(usize, 4));
    while (prefix_len < max_prefix and str1[prefix_len] == str2[prefix_len]) : (prefix_len += 1) {}

    return jaro + 0.1 * @as(f64, @floatFromInt(prefix_len)) * (1.0 - jaro);
}

fn getMatchedCharacters(allocator: Allocator, str1: []const u8, str2: []const u8) ![]u8 {
    var working = try allocator.dupe(u8, str2);
    defer allocator.free(working);

    var matched = std.ArrayListUnmanaged(u8){};
    defer matched.deinit(allocator);

    const limit = @min(str1.len, str2.len) / 2;
    for (str1, 0..) |char, index| {
        const left = if (index > limit) index - limit else 0;
        const right = @min(index + limit + 1, working.len);
        if (left >= right) continue;
        var found_index: ?usize = null;
        for (left..right) |j| {
            if (working[j] == char) {
                found_index = j;
                break;
            }
        }
        if (found_index) |j| {
            try matched.append(allocator, char);
            working[j] = ' ';
        }
    }

    return matched.toOwnedSlice(allocator);
}

test "jaro winkler: python samples" {
    try testing.expectApproxEqRel(0.9611111111111111, try jaroWinkler(testing.allocator, "martha", "marhta"), 1e-12);
    try testing.expectApproxEqRel(0.7333333333333334, try jaroWinkler(testing.allocator, "CRATE", "TRACE"), 1e-12);
    try testing.expectApproxEqRel(0.0, try jaroWinkler(testing.allocator, "test", "dbdbdbdb"), 1e-12);
    try testing.expectApproxEqRel(1.0, try jaroWinkler(testing.allocator, "test", "test"), 1e-12);
}

test "jaro winkler: additional samples" {
    try testing.expectApproxEqRel(0.6363636363636364, try jaroWinkler(testing.allocator, "hello world", "HeLLo W0rlD"), 1e-12);
    try testing.expectApproxEqRel(0.0, try jaroWinkler(testing.allocator, "test", ""), 1e-12);
    try testing.expectApproxEqRel(0.4666666666666666, try jaroWinkler(testing.allocator, "hello", "world"), 1e-12);
    try testing.expectApproxEqRel(0.4365079365079365, try jaroWinkler(testing.allocator, "hell**o", "*world"), 1e-12);
}
