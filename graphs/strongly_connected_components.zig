//! Strongly Connected Components - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/strongly_connected_components.py

const testing = @import("std").testing;
const impl = @import("kosaraju_scc.zig");

test "strongly connected components: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
