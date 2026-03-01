//! Egg Drop Problem - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/egg_drop.py

const std = @import("std");
const testing = std.testing;

pub const EggDropError = error{NoEggs};

/// Returns the minimum number of trials needed in the worst case.
/// Uses the classic moves-based DP: dp[e] = max floors testable with `moves` and `e` eggs.
/// Time complexity: O(eggs * answer), space complexity: O(eggs)
pub fn eggDropMinTrials(
    allocator: std.mem.Allocator,
    eggs: usize,
    floors: usize,
) (EggDropError || std.mem.Allocator.Error)!usize {
    if (floors == 0) return 0;
    if (eggs == 0) return EggDropError.NoEggs;
    if (eggs == 1) return floors;

    const dp = try allocator.alloc(usize, eggs + 1);
    defer allocator.free(dp);
    @memset(dp, 0);

    var moves: usize = 0;
    while (dp[eggs] < floors) {
        moves += 1;

        var e = eggs;
        while (e > 0) {
            const prev_same = dp[e];
            const prev_less = dp[e - 1];

            // Cap at `floors` to avoid needless growth and overflow risk.
            const extra = prev_less + 1;
            if (prev_same >= floors or floors - prev_same <= extra) {
                dp[e] = floors;
            } else {
                dp[e] = prev_same + extra;
            }
            e -= 1;
        }
    }

    return moves;
}

test "egg drop: trivial cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try eggDropMinTrials(alloc, 5, 0));
    try testing.expectEqual(@as(usize, 1), try eggDropMinTrials(alloc, 1, 1));
    try testing.expectEqual(@as(usize, 10), try eggDropMinTrials(alloc, 1, 10));
}

test "egg drop: known values" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 4), try eggDropMinTrials(alloc, 2, 10));
    try testing.expectEqual(@as(usize, 8), try eggDropMinTrials(alloc, 2, 36));
    try testing.expectEqual(@as(usize, 4), try eggDropMinTrials(alloc, 3, 14));
}

test "egg drop: no eggs error" {
    const alloc = testing.allocator;
    try testing.expectError(EggDropError.NoEggs, eggDropMinTrials(alloc, 0, 100));
}

test "egg drop: more eggs than floors" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 1), try eggDropMinTrials(alloc, 100, 1));
    try testing.expectEqual(@as(usize, 2), try eggDropMinTrials(alloc, 100, 3));
}

test "egg drop: extreme floors" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 10), try eggDropMinTrials(alloc, 10, 1000));
}
