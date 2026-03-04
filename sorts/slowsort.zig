//! SlowSort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/slowsort.py

const std = @import("std");
const testing = std.testing;

fn slowsortRange(comptime T: type, sequence: []T, start: usize, end: usize) void {
    if (start >= end) return;

    const mid = (start + end) / 2;
    slowsortRange(T, sequence, start, mid);
    slowsortRange(T, sequence, mid + 1, end);

    if (sequence[end] < sequence[mid]) {
        std.mem.swap(T, &sequence[end], &sequence[mid]);
    }

    slowsortRange(T, sequence, start, end - 1);
}

/// In-place slowsort on [start..end] inclusive.
/// If start/end are null, defaults to full array.
/// Time complexity: super-polynomially bad (humorous algorithm)
/// Space complexity: O(log n) recursion depth
pub fn slowSort(comptime T: type, sequence: []T, start_opt: ?usize, end_opt: ?usize) void {
    if (sequence.len == 0) return;

    const start = start_opt orelse 0;
    const default_end = sequence.len - 1;
    const end = end_opt orelse default_end;
    if (start >= sequence.len or end >= sequence.len or start > end) return;

    slowsortRange(T, sequence, start, end);
}

test "slowsort: python reference examples" {
    var seq1 = [_]i32{ 1, 6, 2, 5, 3, 4, 4, 5 };
    slowSort(i32, &seq1, null, null);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 4, 5, 5, 6 }, &seq1);

    var seq2 = [_]i32{};
    slowSort(i32, &seq2, null, null);
    try testing.expectEqual(@as(usize, 0), seq2.len);

    var seq3 = [_]i32{2};
    slowSort(i32, &seq3, null, null);
    try testing.expectEqualSlices(i32, &[_]i32{2}, &seq3);

    var seq4 = [_]i32{ 4, 3, 2, 1 };
    slowSort(i32, &seq4, null, null);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, &seq4);
}

test "slowsort: range behavior examples" {
    var seq1 = [_]i32{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    slowSort(i32, &seq1, 2, 7);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 2, 3, 4, 5, 6, 7, 1, 0 }, &seq1);

    var seq2 = [_]i32{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    slowSort(i32, &seq2, null, 4);
    try testing.expectEqualSlices(i32, &[_]i32{ 5, 6, 7, 8, 9, 4, 3, 2, 1, 0 }, &seq2);

    var seq3 = [_]i32{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    slowSort(i32, &seq3, 5, null);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 7, 6, 5, 0, 1, 2, 3, 4 }, &seq3);
}

test "slowsort: extreme small-size reverse input" {
    // Keep size small due intentionally very slow complexity.
    var seq = [_]i32{ 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    slowSort(i32, &seq, null, null);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 }, &seq);
}
