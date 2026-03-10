//! Bipartite Graph Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/check_bipatrite.py

const testing = @import("std").testing;
const impl = @import("bipartite_check_bfs.zig");

test "check bipatrite: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
