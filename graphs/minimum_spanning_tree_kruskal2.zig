//! Kruskal Minimum Spanning Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/minimum_spanning_tree_kruskal2.py

const testing = @import("std").testing;
const impl = @import("kruskal.zig");

test "minimum spanning tree kruskal 2: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
