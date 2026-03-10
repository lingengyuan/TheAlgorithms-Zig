//! Kahn Longest Distance in DAG - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/kahns_algorithm_long.py

const testing = @import("std").testing;
const impl = @import("kahn_longest_distance.zig");

test "kahns algorithm long: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
