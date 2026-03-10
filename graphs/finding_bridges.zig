//! Bridges in Undirected Graph - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/finding_bridges.py

const testing = @import("std").testing;
const impl = @import("bridges.zig");

test "finding bridges: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
