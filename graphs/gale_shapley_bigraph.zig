//! Gale-Shapley Stable Matching - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/gale_shapley_bigraph.py

const testing = @import("std").testing;
const impl = @import("gale_shapley_stable_matching.zig");

test "gale shapley bigraph: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
