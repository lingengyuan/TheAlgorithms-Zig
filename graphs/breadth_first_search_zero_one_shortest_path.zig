//! 0-1 Breadth-First Search Shortest Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search_zero_one_shortest_path.py

const testing = @import("std").testing;
const impl = @import("zero_one_bfs_shortest_path.zig");

test "breadth first search zero one shortest path: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
