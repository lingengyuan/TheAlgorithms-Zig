//! Topological Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/g_topological_sort.py

const testing = @import("std").testing;
const impl = @import("kahn_topological_sort.zig");

test "g topological sort: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
