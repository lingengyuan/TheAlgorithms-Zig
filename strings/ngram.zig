//! N-Gram - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/ngram.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const NgramResult = []const []u8;

/// Creates fixed-width n-grams from a sentence.
/// Time complexity: O(n * k), Space complexity: O(n * k)
pub fn createNgram(allocator: Allocator, sentence: []const u8, ngram_size: usize) !NgramResult {
    if (ngram_size == 0 or ngram_size > sentence.len) return allocator.alloc([]u8, 0);

    var out = std.ArrayListUnmanaged([]u8){};
    errdefer {
        for (out.items) |gram| allocator.free(gram);
        out.deinit(allocator);
    }

    var index: usize = 0;
    while (index + ngram_size <= sentence.len) : (index += 1) {
        const gram = try allocator.dupe(u8, sentence[index .. index + ngram_size]);
        errdefer allocator.free(gram);
        try out.append(allocator, gram);
    }

    return out.toOwnedSlice(allocator);
}

pub fn freeNgrams(allocator: Allocator, ngrams: NgramResult) void {
    for (ngrams) |gram| allocator.free(gram);
    allocator.free(ngrams);
}

test "ngram: python samples" {
    const one = try createNgram(testing.allocator, "I am a sentence", 2);
    defer freeNgrams(testing.allocator, one);
    try testing.expectEqual(@as(usize, 14), one.len);
    try testing.expectEqualStrings("I ", one[0]);
    try testing.expectEqualStrings("ce", one[13]);

    const two = try createNgram(testing.allocator, "I am an NLPer", 2);
    defer freeNgrams(testing.allocator, two);
    try testing.expectEqualStrings("NL", two[8]);
    try testing.expectEqualStrings("er", two[11]);
}

test "ngram: edge and extreme" {
    const none = try createNgram(testing.allocator, "This is short", 50);
    defer freeNgrams(testing.allocator, none);
    try testing.expectEqual(@as(usize, 0), none.len);

    const empty = try createNgram(testing.allocator, "", 1);
    defer freeNgrams(testing.allocator, empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}

fn ngramAllocFailImpl(allocator: Allocator) !void {
    const grams = try createNgram(allocator, "I am a sentence", 2);
    defer freeNgrams(allocator, grams);
}

test "ngram: allocation failures free partial output" {
    try testing.checkAllAllocationFailures(testing.allocator, ngramAllocFailImpl, .{});
}
