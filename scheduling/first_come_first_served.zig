//! First Come First Served Scheduling - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/first_come_first_served.py

const std = @import("std");
const testing = std.testing;

pub const FcfsError = error{
    WaitingTimesTooShort,
    EmptyInput,
};

/// Computes waiting time per process in FCFS order.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn calculateWaitingTimes(
    allocator: std.mem.Allocator,
    duration_times: []const i64,
) std.mem.Allocator.Error![]i64 {
    const waiting_times = try allocator.alloc(i64, duration_times.len);
    @memset(waiting_times, 0);
    if (duration_times.len == 0) return waiting_times;

    for (1..duration_times.len) |i| {
        waiting_times[i] = duration_times[i - 1] + waiting_times[i - 1];
    }
    return waiting_times;
}

/// Computes turnaround time per process: duration + waiting.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn calculateTurnaroundTimes(
    allocator: std.mem.Allocator,
    duration_times: []const i64,
    waiting_times: []const i64,
) (std.mem.Allocator.Error || FcfsError)![]i64 {
    if (waiting_times.len < duration_times.len) return FcfsError.WaitingTimesTooShort;

    const turnaround_times = try allocator.alloc(i64, duration_times.len);
    for (duration_times, 0..) |duration_time, i| {
        turnaround_times[i] = duration_time + waiting_times[i];
    }
    return turnaround_times;
}

/// Computes average turnaround time.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn calculateAverageTurnaroundTime(turnaround_times: []const i64) FcfsError!f64 {
    if (turnaround_times.len == 0) return FcfsError.EmptyInput;

    var sum: i64 = 0;
    for (turnaround_times) |v| sum += v;
    return @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(turnaround_times.len));
}

/// Computes average waiting time.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn calculateAverageWaitingTime(waiting_times: []const i64) FcfsError!f64 {
    if (waiting_times.len == 0) return FcfsError.EmptyInput;

    var sum: i64 = 0;
    for (waiting_times) |v| sum += v;
    return @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(waiting_times.len));
}

test "first come first served: python waiting and turnaround examples" {
    const alloc = testing.allocator;

    const w1 = try calculateWaitingTimes(alloc, &[_]i64{ 5, 10, 15 });
    defer alloc.free(w1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 5, 15 }, w1);
    const t1 = try calculateTurnaroundTimes(alloc, &[_]i64{ 5, 10, 15 }, w1);
    defer alloc.free(t1);
    try testing.expectEqualSlices(i64, &[_]i64{ 5, 15, 30 }, t1);

    const w2 = try calculateWaitingTimes(alloc, &[_]i64{ 1, 2, 3, 4, 5 });
    defer alloc.free(w2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 3, 6, 10 }, w2);
    const t2 = try calculateTurnaroundTimes(alloc, &[_]i64{ 1, 2, 3, 4, 5 }, w2);
    defer alloc.free(t2);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 3, 6, 10, 15 }, t2);

    const w3 = try calculateWaitingTimes(alloc, &[_]i64{ 10, 3 });
    defer alloc.free(w3);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 10 }, w3);
    const t3 = try calculateTurnaroundTimes(alloc, &[_]i64{ 10, 3 }, w3);
    defer alloc.free(t3);
    try testing.expectEqualSlices(i64, &[_]i64{ 10, 13 }, t3);
}

test "first come first served: python average examples" {
    try testing.expectApproxEqAbs(@as(f64, 7.0), try calculateAverageTurnaroundTime(&[_]i64{ 0, 5, 16 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 6.5), try calculateAverageTurnaroundTime(&[_]i64{ 1, 5, 8, 12 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 17.0), try calculateAverageTurnaroundTime(&[_]i64{ 10, 24 }), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 7.0), try calculateAverageWaitingTime(&[_]i64{ 0, 5, 16 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 6.5), try calculateAverageWaitingTime(&[_]i64{ 1, 5, 8, 12 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 17.0), try calculateAverageWaitingTime(&[_]i64{ 10, 24 }), 1e-12);
}

test "first come first served: boundary and extreme cases" {
    const alloc = testing.allocator;

    const empty_wait = try calculateWaitingTimes(alloc, &[_]i64{});
    defer alloc.free(empty_wait);
    try testing.expectEqual(@as(usize, 0), empty_wait.len);

    const empty_turn = try calculateTurnaroundTimes(alloc, &[_]i64{}, empty_wait);
    defer alloc.free(empty_turn);
    try testing.expectEqual(@as(usize, 0), empty_turn.len);

    try testing.expectError(FcfsError.EmptyInput, calculateAverageWaitingTime(&[_]i64{}));
    try testing.expectError(FcfsError.EmptyInput, calculateAverageTurnaroundTime(&[_]i64{}));

    try testing.expectError(FcfsError.WaitingTimesTooShort, calculateTurnaroundTimes(alloc, &[_]i64{ 1, 2, 3 }, &[_]i64{ 0, 1 }));

    const huge_wait = try calculateWaitingTimes(alloc, &[_]i64{ 1_000_000_000, 1_000_000_000, 1_000_000_000 });
    defer alloc.free(huge_wait);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1_000_000_000, 2_000_000_000 }, huge_wait);
}
