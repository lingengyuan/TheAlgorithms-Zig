//! Frequency Finder - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/frequency_finder.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const ETAOIN = "ETAOINSHRDLCUMWFGYPBVKJXQZ";
pub const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub fn getLetterCount(message: []const u8) [26]u32 {
    var counts = [_]u32{0} ** 26;
    for (message) |char| {
        const upper = std.ascii.toUpper(char);
        if (upper >= 'A' and upper <= 'Z') counts[upper - 'A'] += 1;
    }
    return counts;
}

/// Returns the letters sorted by observed frequency, breaking ties with the Python ETAOIN rule.
/// Caller owns the returned slice.
/// Time complexity: O(n + 26 log 26), Space complexity: O(1)
pub fn getFrequencyOrder(allocator: Allocator, message: []const u8) ![]u8 {
    const counts = getLetterCount(message);
    var letters = [_]u8{ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };

    const Context = struct { counts: [26]u32 };
    std.sort.heap(u8, &letters, Context{ .counts = counts }, struct {
        fn lessThan(ctx: Context, a: u8, b: u8) bool {
            const count_a = ctx.counts[a - 'A'];
            const count_b = ctx.counts[b - 'A'];
            if (count_a != count_b) return count_a > count_b;
            return etaoinIndex(a) < etaoinIndex(b);
        }

        fn etaoinIndex(char: u8) usize {
            return std.mem.indexOfScalar(u8, ETAOIN, char).?;
        }
    }.lessThan);

    return allocator.dupe(u8, &letters);
}

pub fn englishFreqMatchScore(allocator: Allocator, message: []const u8) !u32 {
    const freq_order = try getFrequencyOrder(allocator, message);
    defer allocator.free(freq_order);

    var score: u32 = 0;
    for (ETAOIN[0..6]) |char| {
        if (std.mem.indexOfScalar(u8, freq_order[0..6], char) != null) score += 1;
    }
    for (ETAOIN[ETAOIN.len - 6 ..]) |char| {
        if (std.mem.indexOfScalar(u8, freq_order[freq_order.len - 6 ..], char) != null) score += 1;
    }
    return score;
}

test "frequency finder: python samples" {
    const one = try getFrequencyOrder(testing.allocator, "Hello World");
    defer testing.allocator.free(one);
    try testing.expectEqualStrings("LOEHRDWTAINSCUMFGYPBVKJXQZ", one);

    const two = try getFrequencyOrder(testing.allocator, "Hello@");
    defer testing.allocator.free(two);
    try testing.expectEqualStrings("LEOHTAINSRDCUMWFGYPBVKJXQZ", two);

    const three = try getFrequencyOrder(testing.allocator, "h");
    defer testing.allocator.free(three);
    try testing.expectEqualStrings("HETAOINSRDLCUMWFGYPBVKJXQZ", three);
}

test "frequency finder: match score and extreme" {
    try testing.expectEqual(@as(u32, 8), try englishFreqMatchScore(testing.allocator, "Hello World"));

    const counts = getLetterCount("AaBbCc123!!!");
    try testing.expectEqual(@as(u32, 2), counts['A' - 'A']);
    try testing.expectEqual(@as(u32, 2), counts['B' - 'A']);
    try testing.expectEqual(@as(u32, 2), counts['C' - 'A']);

    const empty = try getFrequencyOrder(testing.allocator, "");
    defer testing.allocator.free(empty);
    try testing.expectEqualStrings(ETAOIN, empty);
}
