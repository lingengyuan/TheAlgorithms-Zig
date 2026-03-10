//! Breadth-First Search Shortest Path - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search_shortest_path_2.py

const testing = @import("std").testing;
const impl = @import("breadth_first_search_shortest_path.zig");

test "breadth first search shortest path 2: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
