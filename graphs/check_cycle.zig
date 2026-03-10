//! Cycle Detection in Directed Graph - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/check_cycle.py

const testing = @import("std").testing;
const impl = @import("detect_cycle.zig");

test "check cycle: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
