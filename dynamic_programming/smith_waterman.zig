//! Smith-Waterman - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/smith_waterman.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const SmithWatermanError = Allocator.Error;
pub const ScoreMatrix = []const []i32;

/// Returns the local-alignment score matrix for `query` and `subject`.
/// Time complexity: O(m * n), Space complexity: O(m * n)
pub fn smithWaterman(
    allocator: Allocator,
    query: []const u8,
    subject: []const u8,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) SmithWatermanError!ScoreMatrix {
    const rows = query.len + 1;
    const cols = subject.len + 1;

    const matrix = try allocator.alloc([]i32, rows);
    errdefer allocator.free(matrix);

    var allocated_rows: usize = 0;
    errdefer {
        for (matrix[0..allocated_rows]) |row| allocator.free(row);
        allocator.free(matrix);
    }

    for (0..rows) |i| {
        matrix[i] = try allocator.alloc(i32, cols);
        allocated_rows += 1;
        @memset(matrix[i], 0);
    }

    for (1..rows) |i| {
        for (1..cols) |j| {
            const diagonal = matrix[i - 1][j - 1] + scoreFunction(
                query[i - 1],
                subject[j - 1],
                match_score,
                mismatch_score,
                gap_score,
            );
            const up = matrix[i - 1][j] + gap_score;
            const left = matrix[i][j - 1] + gap_score;
            matrix[i][j] = @max(0, @max(diagonal, @max(up, left)));
        }
    }

    return matrix;
}

pub fn scoreFunction(source_char: u8, target_char: u8, match_score: i32, mismatch_score: i32, gap_score: i32) i32 {
    if (source_char == '-' or target_char == '-') return gap_score;
    return if (std.ascii.toUpper(source_char) == std.ascii.toUpper(target_char)) match_score else mismatch_score;
}

pub fn traceback(
    allocator: Allocator,
    score: ScoreMatrix,
    query: []const u8,
    subject: []const u8,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) SmithWatermanError![]u8 {
    var max_value: i32 = std.math.minInt(i32);
    var i_max: usize = 0;
    var j_max: usize = 0;

    for (score, 0..) |row, i| {
        for (row, 0..) |value, j| {
            if (value > max_value) {
                max_value = value;
                i_max = i;
                j_max = j;
            }
        }
    }

    if (i_max == 0 or j_max == 0) return allocator.dupe(u8, "");

    var align1 = std.ArrayListUnmanaged(u8){};
    defer align1.deinit(allocator);
    var align2 = std.ArrayListUnmanaged(u8){};
    defer align2.deinit(allocator);

    var i = i_max;
    var j = j_max;
    while (i > 0 and j > 0 and score[i][j] > 0) {
        const diagonal_score = score[i - 1][j - 1] + scoreFunction(
            query[i - 1],
            subject[j - 1],
            match_score,
            mismatch_score,
            gap_score,
        );
        if (score[i][j] == diagonal_score) {
            try align1.append(allocator, std.ascii.toUpper(query[i - 1]));
            try align2.append(allocator, std.ascii.toUpper(subject[j - 1]));
            i -= 1;
            j -= 1;
        } else if (score[i][j] == score[i - 1][j] + gap_score) {
            try align1.append(allocator, std.ascii.toUpper(query[i - 1]));
            try align2.append(allocator, '-');
            i -= 1;
        } else {
            try align1.append(allocator, '-');
            try align2.append(allocator, std.ascii.toUpper(subject[j - 1]));
            j -= 1;
        }
    }

    std.mem.reverse(u8, align1.items);
    std.mem.reverse(u8, align2.items);
    return std.fmt.allocPrint(allocator, "{s}\n{s}", .{ align1.items, align2.items });
}

pub fn freeScoreMatrix(allocator: Allocator, matrix: ScoreMatrix) void {
    for (matrix) |row| allocator.free(row);
    allocator.free(matrix);
}

test "smith waterman: python sample matrix and traceback" {
    const matrix = try smithWaterman(testing.allocator, "ACAC", "CA", 1, -1, -2);
    defer freeScoreMatrix(testing.allocator, matrix);

    const expected = [_][3]i32{
        .{ 0, 0, 0 },
        .{ 0, 0, 1 },
        .{ 0, 1, 0 },
        .{ 0, 0, 2 },
        .{ 0, 1, 0 },
    };
    for (expected, 0..) |row, i| try testing.expectEqualSlices(i32, &row, matrix[i]);

    const alignment = try traceback(testing.allocator, matrix, "ACAC", "CA", 1, -1, -2);
    defer testing.allocator.free(alignment);
    try testing.expectEqualStrings("CA\nCA", alignment);
}

test "smith waterman: case insensitive behavior" {
    const matrix = try smithWaterman(testing.allocator, "acac", "CA", 1, -1, -2);
    defer freeScoreMatrix(testing.allocator, matrix);

    try testing.expectEqual(@as(i32, 2), matrix[3][2]);
}

test "smith waterman: empty subject" {
    const matrix = try smithWaterman(testing.allocator, "ACAC", "", 1, -1, -2);
    defer freeScoreMatrix(testing.allocator, matrix);

    try testing.expectEqual(@as(usize, 5), matrix.len);
    for (matrix) |row| try testing.expectEqual(@as(usize, 1), row.len);

    const alignment = try traceback(testing.allocator, matrix, "ACAC", "", 1, -1, -2);
    defer testing.allocator.free(alignment);
    try testing.expectEqualStrings("", alignment);
}

test "smith waterman: extreme identical sequences" {
    const matrix = try smithWaterman(testing.allocator, "AGT", "AGT", 1, -1, -2);
    defer freeScoreMatrix(testing.allocator, matrix);

    try testing.expectEqual(@as(i32, 3), matrix[3][3]);
}
