//! Tarjan Strongly Connected Components - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/tarjans_scc.py

const testing = @import("std").testing;
const impl = @import("tarjan_scc.zig");

test "tarjans scc: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
