//! Project Euler Problem 19: Counting Sundays - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_019/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem019Error = error{
    InvalidMonth,
    InvalidDay,
    InvalidYearRange,
};

/// Leap year rule used by Gregorian calendar.
pub fn isLeapYear(year: u32) bool {
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0);
}

fn daysInMonth(year: u32, month: u8) Problem019Error!u8 {
    return switch (month) {
        1, 3, 5, 7, 8, 10, 12 => 31,
        4, 6, 9, 11 => 30,
        2 => if (isLeapYear(year)) 29 else 28,
        else => Problem019Error.InvalidMonth,
    };
}

/// Returns weekday index for a valid date using 1900-01-01 (Monday) as base.
/// Monday=0 ... Sunday=6.
pub fn weekday(year: u32, month: u8, day: u8) Problem019Error!u8 {
    const dim = try daysInMonth(year, month);
    if (day == 0 or day > dim) return Problem019Error.InvalidDay;

    var days_since_base: u64 = 0;

    var y: u32 = 1900;
    while (y < year) : (y += 1) {
        days_since_base += if (isLeapYear(y)) 366 else 365;
    }

    var m: u8 = 1;
    while (m < month) : (m += 1) {
        days_since_base += try daysInMonth(year, m);
    }

    days_since_base += day - 1;
    return @intCast(days_since_base % 7);
}

/// Counts Sundays that fall on day 1 from `start_year`..`end_year` inclusive.
///
/// Time complexity: O((end_year - start_year + 1) * 12)
/// Space complexity: O(1)
pub fn countSundaysOnFirst(start_year: u32, end_year: u32) Problem019Error!u32 {
    if (start_year > end_year) return Problem019Error.InvalidYearRange;

    var sundays: u32 = 0;

    var year = start_year;
    while (year <= end_year) : (year += 1) {
        var month: u8 = 1;
        while (month <= 12) : (month += 1) {
            if (try weekday(year, month, 1) == 6) {
                sundays += 1;
            }
        }
    }

    return sundays;
}

/// Euler problem default solution (1901-2000).
pub fn solution() !u32 {
    return countSundaysOnFirst(1901, 2000);
}

test "problem 019: python reference" {
    try testing.expectEqual(@as(u32, 171), try solution());
}

test "problem 019: boundaries and calendar checks" {
    try testing.expectEqual(@as(u32, 2), try countSundaysOnFirst(1901, 1901));
    try testing.expectEqual(@as(u32, 1), try countSundaysOnFirst(2000, 2000));
    try testing.expectEqual(@as(u32, 2), try countSundaysOnFirst(1999, 2000));

    // 1900-01-01 is Monday, by statement.
    try testing.expectEqual(@as(u8, 0), try weekday(1900, 1, 1));
    // 1901-01-01 is Tuesday.
    try testing.expectEqual(@as(u8, 1), try weekday(1901, 1, 1));

    try testing.expect(isLeapYear(2000));
    try testing.expect(!isLeapYear(1900));
    try testing.expect(isLeapYear(1996));

    try testing.expectError(Problem019Error.InvalidMonth, weekday(1900, 13, 1));
    try testing.expectError(Problem019Error.InvalidDay, weekday(1900, 2, 29));
    try testing.expectError(Problem019Error.InvalidYearRange, countSundaysOnFirst(2001, 2000));
}
