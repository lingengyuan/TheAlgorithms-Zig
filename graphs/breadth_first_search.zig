//! Breadth-First Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/breadth_first_search.py

const testing = @import("std").testing;
const impl = @import("bfs.zig");

test "breadth first search: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
