//! Minimum Cost For Tickets - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_tickets_cost.py

const std = @import("std");
const testing = std.testing;

pub const MinimumTicketsCostError = error{
    InvalidDays,
    InvalidCosts,
    Overflow,
};

/// Returns the minimum ticket cost to cover all travel days.
/// `days` must be in [1, 365]. `costs` must contain exactly three non-negative values:
/// [1-day, 7-day, 30-day].
/// Time complexity: O(365), Space complexity: O(365)
pub fn minimumTicketsCost(
    days: []const i32,
    costs: []const i32,
) MinimumTicketsCostError!u32 {
    if (costs.len != 3) return MinimumTicketsCostError.InvalidCosts;
    for (costs) |cost| {
        if (cost < 0) return MinimumTicketsCostError.InvalidCosts;
    }

    if (days.len == 0) return 0;

    var travel = [_]bool{false} ** 366;
    for (days) |day| {
        if (day <= 0 or day >= 366) return MinimumTicketsCostError.InvalidDays;
        const idx: usize = @intCast(day);
        travel[idx] = true;
    }

    var dp = [_]u32{0} ** 367;
    const c1: u32 = @intCast(costs[0]);
    const c7: u32 = @intCast(costs[1]);
    const c30: u32 = @intCast(costs[2]);

    var day: usize = 365;
    while (true) : (day -= 1) {
        if (!travel[day]) {
            dp[day] = dp[day + 1];
        } else {
            const one = @addWithOverflow(c1, dp[day + 1]);
            const d7 = @min(day + 7, @as(usize, 366));
            const seven = @addWithOverflow(c7, dp[d7]);
            const d30 = @min(day + 30, @as(usize, 366));
            const thirty = @addWithOverflow(c30, dp[d30]);
            if (one[1] != 0 or seven[1] != 0 or thirty[1] != 0) {
                return MinimumTicketsCostError.Overflow;
            }
            dp[day] = @min(one[0], @min(seven[0], thirty[0]));
        }

        if (day == 1) break;
    }

    return dp[1];
}

test "minimum tickets cost: python samples" {
    try testing.expectEqual(@as(u32, 11), try minimumTicketsCost(
        &[_]i32{ 1, 4, 6, 7, 8, 20 },
        &[_]i32{ 2, 7, 15 },
    ));

    try testing.expectEqual(@as(u32, 17), try minimumTicketsCost(
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 31 },
        &[_]i32{ 2, 7, 15 },
    ));

    try testing.expectEqual(@as(u32, 24), try minimumTicketsCost(
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 31 },
        &[_]i32{ 2, 90, 150 },
    ));
}

test "minimum tickets cost: boundary behavior" {
    try testing.expectEqual(@as(u32, 2), try minimumTicketsCost(&[_]i32{2}, &[_]i32{ 2, 90, 150 }));
    try testing.expectEqual(@as(u32, 0), try minimumTicketsCost(&[_]i32{}, &[_]i32{ 2, 90, 150 }));
}

test "minimum tickets cost: invalid inputs" {
    try testing.expectError(MinimumTicketsCostError.InvalidDays, minimumTicketsCost(&[_]i32{ 0, 2 }, &[_]i32{ 2, 7, 15 }));
    try testing.expectError(MinimumTicketsCostError.InvalidDays, minimumTicketsCost(&[_]i32{ 2, 367 }, &[_]i32{ 2, 7, 15 }));
    try testing.expectError(MinimumTicketsCostError.InvalidCosts, minimumTicketsCost(&[_]i32{ 2, 3 }, &[_]i32{ 2, 7 }));
    try testing.expectError(MinimumTicketsCostError.InvalidCosts, minimumTicketsCost(&[_]i32{ 2, 3 }, &[_]i32{ 2, -7, 15 }));
}

test "minimum tickets cost: extreme full-year case" {
    var days = [_]i32{0} ** 365;
    for (&days, 1..) |*slot, day| slot.* = @intCast(day);

    // Full year best is 13 monthly passes when 30-day pass is cheapest.
    try testing.expectEqual(@as(u32, 13), try minimumTicketsCost(&days, &[_]i32{ 10, 20, 1 }));
}
