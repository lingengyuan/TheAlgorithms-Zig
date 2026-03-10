//! Kruskal Minimum Spanning Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/minimum_spanning_tree_kruskal.py

const testing = @import("std").testing;
const impl = @import("kruskal.zig");

test "minimum spanning tree kruskal: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
