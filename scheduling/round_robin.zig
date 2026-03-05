//! Round Robin Scheduling - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/round_robin.py

const std = @import("std");
const testing = std.testing;

/// Computes waiting times for round-robin scheduling with fixed quantum=2
/// (matching the Python reference).
///
/// Time complexity: O(total_quanta * n)
/// Space complexity: O(n)
pub fn calculateWaitingTimes(
    allocator: std.mem.Allocator,
    burst_times: []const i64,
) std.mem.Allocator.Error![]i64 {
    const quantum: i64 = 2;
    var rem_burst_times = try allocator.alloc(i64, burst_times.len);
    defer allocator.free(rem_burst_times);
    @memcpy(rem_burst_times, burst_times);

    const waiting_times = try allocator.alloc(i64, burst_times.len);
    @memset(waiting_times, 0);

    var current_time: i64 = 0;
    while (true) {
        var done = true;
        for (burst_times, 0..) |burst_time, i| {
            if (rem_burst_times[i] > 0) {
                done = false;
                if (rem_burst_times[i] > quantum) {
                    current_time += quantum;
                    rem_burst_times[i] -= quantum;
                } else {
                    current_time += rem_burst_times[i];
                    waiting_times[i] = current_time - burst_time;
                    rem_burst_times[i] = 0;
                }
            }
        }

        if (done) return waiting_times;
    }
}

/// Computes turnaround times by zipping burst_times and waiting_times.
/// Output length equals min(burst_times.len, waiting_times.len), matching Python `zip`.
///
/// Time complexity: O(min(n, m))
/// Space complexity: O(min(n, m))
pub fn calculateTurnAroundTimes(
    allocator: std.mem.Allocator,
    burst_times: []const i64,
    waiting_times: []const i64,
) std.mem.Allocator.Error![]i64 {
    const result_len = @min(burst_times.len, waiting_times.len);
    const turn_around_times = try allocator.alloc(i64, result_len);
    for (0..result_len) |i| {
        turn_around_times[i] = burst_times[i] + waiting_times[i];
    }
    return turn_around_times;
}

test "round robin: waiting times python examples" {
    const alloc = testing.allocator;

    const w1 = try calculateWaitingTimes(alloc, &[_]i64{ 10, 5, 8 });
    defer alloc.free(w1);
    try testing.expectEqualSlices(i64, &[_]i64{ 13, 10, 13 }, w1);

    const w2 = try calculateWaitingTimes(alloc, &[_]i64{ 4, 6, 3, 1 });
    defer alloc.free(w2);
    try testing.expectEqualSlices(i64, &[_]i64{ 5, 8, 9, 6 }, w2);

    const w3 = try calculateWaitingTimes(alloc, &[_]i64{ 12, 2, 10 });
    defer alloc.free(w3);
    try testing.expectEqualSlices(i64, &[_]i64{ 12, 2, 12 }, w3);
}

test "round robin: turn around times python examples" {
    const alloc = testing.allocator;

    const t1 = try calculateTurnAroundTimes(alloc, &[_]i64{ 1, 2, 3, 4 }, &[_]i64{ 0, 1, 3 });
    defer alloc.free(t1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 3, 6 }, t1);

    const t2 = try calculateTurnAroundTimes(alloc, &[_]i64{ 10, 3, 7 }, &[_]i64{ 10, 6, 11 });
    defer alloc.free(t2);
    try testing.expectEqualSlices(i64, &[_]i64{ 20, 9, 18 }, t2);
}

test "round robin: boundary and extreme cases" {
    const alloc = testing.allocator;

    const empty_wait = try calculateWaitingTimes(alloc, &[_]i64{});
    defer alloc.free(empty_wait);
    try testing.expectEqual(@as(usize, 0), empty_wait.len);

    const zeros_wait = try calculateWaitingTimes(alloc, &[_]i64{ 0, 0, 0 });
    defer alloc.free(zeros_wait);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0 }, zeros_wait);

    const mixed_wait = try calculateWaitingTimes(alloc, &[_]i64{ -3, 5 });
    defer alloc.free(mixed_wait);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0 }, mixed_wait);

    const huge_wait = try calculateWaitingTimes(alloc, &[_]i64{1_000_000});
    defer alloc.free(huge_wait);
    try testing.expectEqualSlices(i64, &[_]i64{0}, huge_wait);
}
