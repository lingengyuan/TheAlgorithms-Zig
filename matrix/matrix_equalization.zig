//! Matrix Equalization - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/matrix_equalization.py

const std = @import("std");
const testing = std.testing;

pub const MatrixEqualizationError = error{InvalidStepSize};

/// Returns the minimal updates needed to equalize the vector under the given step size.
/// Time complexity: O(u * n), Space complexity: O(u)
pub fn arrayEqualization(
    allocator: std.mem.Allocator,
    vector: []const i64,
    step_size: usize,
) (MatrixEqualizationError || std.mem.Allocator.Error)!usize {
    if (step_size == 0) return error.InvalidStepSize;
    if (vector.len == 0) return 0;

    var unique_elements = std.AutoHashMap(i64, void).init(allocator);
    defer unique_elements.deinit();
    for (vector) |value| {
        try unique_elements.put(value, {});
    }

    var min_updates: usize = std.math.maxInt(usize);
    var it = unique_elements.keyIterator();
    while (it.next()) |element_ptr| {
        const element = element_ptr.*;
        var idx: usize = 0;
        var updates: usize = 0;
        while (idx < vector.len) {
            if (vector[idx] != element) {
                updates += 1;
                const next = @addWithOverflow(idx, step_size);
                idx = if (next[1] != 0) vector.len else next[0];
            } else {
                idx += 1;
            }
        }
        min_updates = @min(min_updates, updates);
    }
    return min_updates;
}

test "matrix equalization: python reference examples" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        @as(usize, 4),
        try arrayEqualization(alloc, &[_]i64{ 1, 1, 6, 2, 4, 6, 5, 1, 7, 2, 2, 1, 7, 2, 2 }, 4),
    );
    try testing.expectEqual(
        @as(usize, 5),
        try arrayEqualization(alloc, &[_]i64{ 22, 81, 88, 71, 22, 81, 632, 81, 81, 22, 92 }, 2),
    );
    try testing.expectEqual(
        @as(usize, 0),
        try arrayEqualization(alloc, &[_]i64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 5),
    );
}

test "matrix equalization: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(
        @as(usize, 2),
        try arrayEqualization(alloc, &[_]i64{ 22, 22, 22, 33, 33, 33 }, 2),
    );
    try testing.expectError(error.InvalidStepSize, arrayEqualization(alloc, &[_]i64{ 1, 2, 3 }, 0));
    try testing.expectEqual(@as(usize, 1), try arrayEqualization(alloc, &[_]i64{ 1, 2, 3 }, std.math.maxInt(usize)));
    try testing.expectEqual(@as(usize, 0), try arrayEqualization(alloc, &[_]i64{}, 3));
}
