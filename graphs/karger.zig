//! Karger Minimum Cut - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/karger.py

const testing = @import("std").testing;
const impl = @import("karger_min_cut.zig");

test "karger: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
