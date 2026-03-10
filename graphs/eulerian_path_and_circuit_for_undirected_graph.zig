//! Eulerian Path/Circuit (Undirected) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/eulerian_path_and_circuit_for_undirected_graph.py

const testing = @import("std").testing;
const impl = @import("eulerian_path_circuit_undirected.zig");

test "eulerian path and circuit for undirected graph: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
