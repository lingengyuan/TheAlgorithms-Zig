//! Floyd-Warshall All-Pairs Shortest Paths - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/graphs_floyd_warshall.py

const testing = @import("std").testing;
const impl = @import("floyd_warshall.zig");

test "graphs floyd warshall: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
