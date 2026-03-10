//! Kosaraju Strongly Connected Components - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/scc_kosaraju.py

const testing = @import("std").testing;
const impl = @import("kosaraju_scc.zig");

test "scc kosaraju: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
