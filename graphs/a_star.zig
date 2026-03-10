//! A* Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/a_star.py

const testing = @import("std").testing;
const impl = @import("a_star_search.zig");

test "a star: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
