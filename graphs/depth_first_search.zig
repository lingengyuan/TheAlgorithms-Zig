//! Depth-First Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/depth_first_search.py

const testing = @import("std").testing;
const impl = @import("dfs.zig");

test "depth first search: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
