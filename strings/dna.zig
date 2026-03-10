//! DNA Complement - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/dna.py

const std = @import("std");
const testing = std.testing;

pub const DnaError = error{InvalidStrand} || std.mem.Allocator.Error;

/// Returns the complementary DNA strand.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn dna(allocator: std.mem.Allocator, strand: []const u8) DnaError![]u8 {
    const out = try allocator.alloc(u8, strand.len);
    errdefer allocator.free(out);

    for (strand, 0..) |char, index| {
        out[index] = switch (char) {
            'A' => 'T',
            'T' => 'A',
            'C' => 'G',
            'G' => 'C',
            else => return DnaError.InvalidStrand,
        };
    }
    return out;
}

test "dna: python samples" {
    const one = try dna(testing.allocator, "GCTA");
    defer testing.allocator.free(one);
    try testing.expectEqualStrings("CGAT", one);

    const two = try dna(testing.allocator, "ATGC");
    defer testing.allocator.free(two);
    try testing.expectEqualStrings("TACG", two);

    const three = try dna(testing.allocator, "CTGA");
    defer testing.allocator.free(three);
    try testing.expectEqualStrings("GACT", three);
}

test "dna: invalid and extreme" {
    try testing.expectError(DnaError.InvalidStrand, dna(testing.allocator, "GFGG"));

    var long_strand = [_]u8{'A'} ** 100_000;
    const out = try dna(testing.allocator, &long_strand);
    defer testing.allocator.free(out);
    try testing.expect(out[0] == 'T');
    try testing.expect(out[out.len - 1] == 'T');
}
