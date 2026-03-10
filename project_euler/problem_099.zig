//! Project Euler Problem 99: Largest Exponential - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_099/sol1.py

const std = @import("std");
const testing = std.testing;

const data_file = @embedFile("problem_099_base_exp.txt");

pub const Problem099Error = error{ InvalidLine, InvalidNumber };

/// Returns the 1-based line number of the largest base/exponent pair in the dataset.
/// Time complexity: O(lines)
/// Space complexity: O(1)
pub fn solution(data: []const u8) Problem099Error!u32 {
    var largest: f64 = -std.math.inf(f64);
    var result: u32 = 0;
    var line_no: u32 = 0;
    var lines = std.mem.tokenizeAny(u8, data, "\r\n");
    while (lines.next()) |line| {
        line_no += 1;
        var items = std.mem.tokenizeScalar(u8, line, ',');
        const a_text = items.next() orelse return error.InvalidLine;
        const x_text = items.next() orelse return error.InvalidLine;
        if (items.next() != null) return error.InvalidLine;
        const a = std.fmt.parseInt(u32, a_text, 10) catch return error.InvalidNumber;
        const x = std.fmt.parseInt(u32, x_text, 10) catch return error.InvalidNumber;
        const score = @as(f64, @floatFromInt(x)) * std.math.log10(@as(f64, @floatFromInt(a)));
        if (score > largest) {
            largest = score;
            result = line_no;
        }
    }
    return result;
}

test "problem 099: python reference" {
    try testing.expectEqual(@as(u32, 709), try solution(data_file));
}

test "problem 099: small sample and invalid data" {
    try testing.expectEqual(@as(u32, 2), try solution("519432,525806\n632382,518061\n"));
    try testing.expectError(error.InvalidLine, solution("2\n"));
}
