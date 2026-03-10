//! Boruvka Minimum Spanning Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/boruvka.py

const testing = @import("std").testing;
const impl = @import("boruvka_mst.zig");

test "boruvka: compatibility wrapper imports implementation" {
    _ = impl;
    try testing.expect(true);
}
