//! Kahn Topological Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/kahns_algorithm_topo.py

const testing = @import("std").testing;
const impl = @import("kahn_topological_sort.zig");

test "kahns algorithm topo: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
