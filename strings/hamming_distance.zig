//! Hamming Distance - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/hamming_distance.py

const std = @import("std");
const testing = std.testing;

pub const HammingError = error{LengthMismatch};

/// Counts positions at which two equal-length strings differ.
/// Returns error.LengthMismatch if lengths differ.
/// Time complexity: O(n)
pub fn hammingDistance(a: []const u8, b: []const u8) HammingError!usize {
    if (a.len != b.len) return HammingError.LengthMismatch;
    var count: usize = 0;
    for (a, b) |ca, cb| {
        if (ca != cb) count += 1;
    }
    return count;
}

test "hamming distance: known values" {
    try testing.expectEqual(@as(usize, 0), try hammingDistance("python", "python"));
    try testing.expectEqual(@as(usize, 3), try hammingDistance("karolin", "kathrin"));
    try testing.expectEqual(@as(usize, 5), try hammingDistance("00000", "11111"));
}

test "hamming distance: length mismatch" {
    try testing.expectError(HammingError.LengthMismatch, hammingDistance("karolin", "kath"));
}

test "hamming distance: empty strings" {
    try testing.expectEqual(@as(usize, 0), try hammingDistance("", ""));
}
