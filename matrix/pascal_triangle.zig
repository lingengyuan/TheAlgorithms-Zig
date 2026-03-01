//! Pascal's Triangle - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/pascal_triangle.py

const std = @import("std");
const testing = std.testing;

pub const PascalError = error{Overflow};

/// Generates Pascal's triangle with `num_rows` rows.
/// Returns a slice of slices. Caller must free each inner slice and the outer slice.
pub fn pascalTriangle(
    allocator: std.mem.Allocator,
    num_rows: usize,
) (PascalError || std.mem.Allocator.Error)![][]u64 {
    const triangle = try allocator.alloc([]u64, num_rows);
    errdefer allocator.free(triangle);

    var built_rows: usize = 0;
    errdefer {
        for (triangle[0..built_rows]) |row| allocator.free(row);
    }

    for (0..num_rows) |r| {
        const row_len = @addWithOverflow(r, @as(usize, 1));
        if (row_len[1] != 0) return PascalError.Overflow;
        const row = try allocator.alloc(u64, row_len[0]);
        errdefer allocator.free(row);

        row[0] = 1;
        row[r] = 1;
        if (r >= 2) {
            const prev = triangle[r - 1];
            for (1..r) |c| {
                const sum = @addWithOverflow(prev[c - 1], prev[c]);
                if (sum[1] != 0) return PascalError.Overflow;
                row[c] = sum[0];
            }
        }
        triangle[r] = row;
        built_rows += 1;
    }
    return triangle;
}

fn freePascal(allocator: std.mem.Allocator, triangle: [][]u64) void {
    for (triangle) |row| allocator.free(row);
    allocator.free(triangle);
}

test "pascal triangle: 5 rows" {
    const alloc = testing.allocator;
    const t = try pascalTriangle(alloc, 5);
    defer freePascal(alloc, t);

    try testing.expectEqualSlices(u64, &[_]u64{1}, t[0]);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 1 }, t[1]);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 2, 1 }, t[2]);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 3, 3, 1 }, t[3]);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 4, 6, 4, 1 }, t[4]);
}

test "pascal triangle: 0 rows" {
    const alloc = testing.allocator;
    const t = try pascalTriangle(alloc, 0);
    defer alloc.free(t);
    try testing.expectEqual(@as(usize, 0), t.len);
}

test "pascal triangle: 1 row" {
    const alloc = testing.allocator;
    const t = try pascalTriangle(alloc, 1);
    defer freePascal(alloc, t);
    try testing.expectEqualSlices(u64, &[_]u64{1}, t[0]);
}

test "pascal triangle: overflow is reported for large rows" {
    try testing.expectError(PascalError.Overflow, pascalTriangle(testing.allocator, 69));
}
