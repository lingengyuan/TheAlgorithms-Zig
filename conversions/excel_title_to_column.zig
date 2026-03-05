//! Excel Title to Column Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/excel_title_to_column.py

const std = @import("std");
const testing = std.testing;

pub const ConversionError = error{ InvalidTitle, Overflow };

/// Converts Excel column title (e.g. "AB") to 1-based index.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn excelTitleToColumn(title: []const u8) ConversionError!u64 {
    if (title.len == 0) return ConversionError.InvalidTitle;

    var answer: u64 = 0;
    for (title) |ch| {
        if (ch < 'A' or ch > 'Z') return ConversionError.InvalidTitle;

        const mul = @mulWithOverflow(answer, @as(u64, 26));
        if (mul[1] != 0) return ConversionError.Overflow;
        const add = @addWithOverflow(mul[0], @as(u64, ch - 'A' + 1));
        if (add[1] != 0) return ConversionError.Overflow;
        answer = add[0];
    }

    return answer;
}

test "excel title to column: python examples" {
    try testing.expectEqual(@as(u64, 1), try excelTitleToColumn("A"));
    try testing.expectEqual(@as(u64, 2), try excelTitleToColumn("B"));
    try testing.expectEqual(@as(u64, 28), try excelTitleToColumn("AB"));
    try testing.expectEqual(@as(u64, 26), try excelTitleToColumn("Z"));
}

test "excel title to column: edge and validation" {
    try testing.expectEqual(@as(u64, 2147483647), try excelTitleToColumn("FXSHRXW"));

    try testing.expectError(ConversionError.InvalidTitle, excelTitleToColumn(""));
    try testing.expectError(ConversionError.InvalidTitle, excelTitleToColumn("a"));
    try testing.expectError(ConversionError.InvalidTitle, excelTitleToColumn("A1"));
}
