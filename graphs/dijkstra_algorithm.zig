//! Dijkstra Shortest Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dijkstra_algorithm.py

const testing = @import("std").testing;
const impl = @import("dijkstra.zig");

test "dijkstra algorithm: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
