//! Zeller's Congruence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/zellers_congruence.py

const std = @import("std");
const testing = std.testing;

pub const ZellerError = error{
    InvalidLength,
    InvalidMonth,
    InvalidDay,
    InvalidSeparator,
    InvalidYear,
};

fn isLeapYear(year: i32) bool {
    return (@rem(year, 400) == 0) or (@rem(year, 4) == 0 and @rem(year, 100) != 0);
}

fn daysInMonth(month: i32, year: i32) i32 {
    return switch (month) {
        1, 3, 5, 7, 8, 10, 12 => 31,
        4, 6, 9, 11 => 30,
        2 => if (isLeapYear(year)) 29 else 28,
        else => 0,
    };
}

/// Returns the formatted weekday message for a `mm-dd-yyyy` or `mm/dd/yyyy` date.
/// Caller owns the returned slice.
pub fn zeller(allocator: std.mem.Allocator, date_input: []const u8) ZellerError![]u8 {
    if (date_input.len != 10) return error.InvalidLength;

    const month = std.fmt.parseInt(i32, date_input[0..2], 10) catch return error.InvalidMonth;
    if (!(month > 0 and month < 13)) return error.InvalidMonth;

    const sep_1 = date_input[2];
    if (sep_1 != '-' and sep_1 != '/') return error.InvalidSeparator;

    const day = std.fmt.parseInt(i32, date_input[3..5], 10) catch return error.InvalidDay;
    const sep_2 = date_input[5];
    if (sep_2 != '-' and sep_2 != '/') return error.InvalidSeparator;

    const year = std.fmt.parseInt(i32, date_input[6..10], 10) catch return error.InvalidYear;
    if (!(year > 45 and year < 8500)) return error.InvalidYear;
    if (!(day > 0 and day <= daysInMonth(month, year))) return error.InvalidDay;

    var m = month;
    var y = year;
    if (m <= 2) {
        y -= 1;
        m += 12;
    }

    const c = @divTrunc(y, 100);
    const k = @mod(y, 100);
    const t: i32 = @intFromFloat(2.6 * @as(f64, @floatFromInt(m)) - 5.39);
    const u = @divTrunc(c, 4);
    const v = @divTrunc(k, 4);
    const x = day + k;
    const z = t + u + v + x;
    const w = z - (2 * c);
    const f: usize = @intCast(@mod(w, 7));

    const days = [_][]const u8{ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
    return std.fmt.allocPrint(allocator, "Your date {s}, is a {s}!", .{ date_input, days[f] }) catch unreachable;
}

test "zellers congruence: python reference examples" {
    const alloc = testing.allocator;
    const msg = try zeller(alloc, "01-31-2010");
    defer alloc.free(msg);
    try testing.expectEqualStrings("Your date 01-31-2010, is a Sunday!", msg);
}

test "zellers congruence: edge cases" {
    const alloc = testing.allocator;
    try testing.expectError(error.InvalidMonth, zeller(alloc, "13-31-2010"));
    try testing.expectError(error.InvalidDay, zeller(alloc, "01-33-2010"));
    try testing.expectError(error.InvalidSeparator, zeller(alloc, "01-31*2010"));
    try testing.expectError(error.InvalidYear, zeller(alloc, "01-31-8999"));
    try testing.expectError(error.InvalidLength, zeller(alloc, ""));
}
