//! Bi-directional Dijkstra - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/bi_directional_dijkstra.py

const testing = @import("std").testing;
const impl = @import("bidirectional_dijkstra.zig");

test "bi-directional dijkstra: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
