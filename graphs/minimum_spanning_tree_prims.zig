//! Prim Minimum Spanning Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/minimum_spanning_tree_prims.py

const testing = @import("std").testing;
const impl = @import("prim.zig");

test "minimum spanning tree prims: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
