//! Graph Adjacency List Data Structure - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/graph_list.py

const testing = @import("std").testing;
const impl = @import("graph_adjacency_list.zig");

test "graph list: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
