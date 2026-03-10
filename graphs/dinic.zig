//! Dinic Maximum Flow - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/dinic.py

const testing = @import("std").testing;
const impl = @import("dinic_max_flow.zig");

test "dinic: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
