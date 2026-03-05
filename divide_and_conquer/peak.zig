//! Peak of Unimodal Array - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/peak.py

const std = @import("std");
const testing = std.testing;

pub const PeakError = error{EmptyInput};

/// Returns a peak value from a unimodal array using divide-and-conquer.
///
/// Time complexity: O(log n)
/// Space complexity: O(1)
pub fn peak(values: []const i64) PeakError!i64 {
    if (values.len == 0) return PeakError.EmptyInput;
    if (values.len == 1) return values[0];

    var left: usize = 0;
    var right: usize = values.len - 1;

    while (left < right) {
        const mid = left + (right - left) / 2;
        if (values[mid] < values[mid + 1]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return values[left];
}

test "peak: python examples" {
    try testing.expectEqual(@as(i64, 5), try peak(&[_]i64{ 1, 2, 3, 4, 5, 4, 3, 2, 1 }));
    try testing.expectEqual(@as(i64, 10), try peak(&[_]i64{ 1, 10, 9, 8, 7, 6, 5, 4 }));
    try testing.expectEqual(@as(i64, 9), try peak(&[_]i64{ 1, 9, 8, 7 }));
    try testing.expectEqual(@as(i64, 7), try peak(&[_]i64{ 1, 2, 3, 4, 5, 6, 7, 0 }));
    try testing.expectEqual(@as(i64, 4), try peak(&[_]i64{ 1, 2, 3, 4, 3, 2, 1, 0, -1, -2 }));
}

test "peak: edge and extreme cases" {
    try testing.expectError(PeakError.EmptyInput, peak(&[_]i64{}));
    try testing.expectEqual(@as(i64, 42), try peak(&[_]i64{42}));
    try testing.expectEqual(@as(i64, std.math.maxInt(i64)), try peak(&[_]i64{ std.math.minInt(i64), 0, std.math.maxInt(i64), -1 }));
}
