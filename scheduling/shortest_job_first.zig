//! Shortest Job First (Preemptive/SRTF-style) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/shortest_job_first.py

const std = @import("std");
const testing = std.testing;

pub const SjfError = error{
    InputLengthMismatch,
    InvalidProcessCount,
    InvalidBurstTime,
    InvalidArrivalTime,
};

pub const AverageTimes = struct {
    waiting: f64,
    turnaround: f64,
};

/// Computes waiting times following the Python reference algorithm.
///
/// Time complexity: O(T * n), where T is simulated timeline length.
/// Space complexity: O(n)
pub fn calculateWaitingTime(
    allocator: std.mem.Allocator,
    arrival_time: []const i64,
    burst_time: []const i64,
    no_of_processes: usize,
) (std.mem.Allocator.Error || SjfError)![]i64 {
    if (arrival_time.len < no_of_processes or burst_time.len < no_of_processes) {
        return SjfError.InputLengthMismatch;
    }
    for (0..no_of_processes) |i| {
        if (arrival_time[i] < 0) return SjfError.InvalidArrivalTime;
        if (burst_time[i] <= 0) return SjfError.InvalidBurstTime;
    }

    const waiting_time = try allocator.alloc(i64, no_of_processes);
    @memset(waiting_time, 0);

    if (no_of_processes == 0) return waiting_time;

    var remaining_time = try allocator.alloc(i64, no_of_processes);
    defer allocator.free(remaining_time);

    for (0..no_of_processes) |i| {
        remaining_time[i] = burst_time[i];
    }

    var complete: usize = 0;
    var increment_time: i64 = 0;
    var minm: i64 = 999_999_999;
    var short: usize = 0;
    var check = false;

    while (complete != no_of_processes) {
        for (0..no_of_processes) |j| {
            if (arrival_time[j] <= increment_time and
                remaining_time[j] > 0 and
                remaining_time[j] < minm)
            {
                minm = remaining_time[j];
                short = j;
                check = true;
            }
        }

        if (!check) {
            increment_time += 1;
            continue;
        }

        remaining_time[short] -= 1;

        minm = remaining_time[short];
        if (minm == 0) {
            minm = 999_999_999;
        }

        if (remaining_time[short] == 0) {
            complete += 1;
            check = false;

            const finish_time = increment_time + 1;
            const finar = finish_time - arrival_time[short];
            var wait = finar - burst_time[short];
            if (wait < 0) wait = 0;
            waiting_time[short] = wait;
        }

        increment_time += 1;
    }

    return waiting_time;
}

/// Computes turnaround time for each process: burst + waiting.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn calculateTurnAroundTime(
    allocator: std.mem.Allocator,
    burst_time: []const i64,
    no_of_processes: usize,
    waiting_time: []const i64,
) (std.mem.Allocator.Error || SjfError)![]i64 {
    if (no_of_processes == 0) return allocator.alloc(i64, 0);
    if (burst_time.len < no_of_processes or waiting_time.len < no_of_processes) {
        return SjfError.InputLengthMismatch;
    }

    const turn_around_time = try allocator.alloc(i64, no_of_processes);
    for (0..no_of_processes) |i| {
        turn_around_time[i] = burst_time[i] + waiting_time[i];
    }
    return turn_around_time;
}

/// Computes average waiting and turnaround times.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn calculateAverageTimes(
    waiting_time: []const i64,
    turn_around_time: []const i64,
    no_of_processes: usize,
) SjfError!AverageTimes {
    if (no_of_processes == 0) return SjfError.InvalidProcessCount;
    if (waiting_time.len < no_of_processes or turn_around_time.len < no_of_processes) {
        return SjfError.InputLengthMismatch;
    }

    var total_waiting_time: i64 = 0;
    var total_turn_around_time: i64 = 0;
    for (0..no_of_processes) |i| {
        total_waiting_time += waiting_time[i];
        total_turn_around_time += turn_around_time[i];
    }

    const n: f64 = @floatFromInt(no_of_processes);
    return AverageTimes{
        .waiting = @as(f64, @floatFromInt(total_waiting_time)) / n,
        .turnaround = @as(f64, @floatFromInt(total_turn_around_time)) / n,
    };
}

test "shortest job first: waiting time python examples" {
    const alloc = testing.allocator;

    const w1 = try calculateWaitingTime(alloc, &[_]i64{ 1, 2, 3, 4 }, &[_]i64{ 3, 3, 5, 1 }, 4);
    defer alloc.free(w1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 3, 5, 0 }, w1);

    const w2 = try calculateWaitingTime(alloc, &[_]i64{ 1, 2, 3 }, &[_]i64{ 2, 5, 1 }, 3);
    defer alloc.free(w2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 2, 0 }, w2);

    const w3 = try calculateWaitingTime(alloc, &[_]i64{ 2, 3 }, &[_]i64{ 5, 1 }, 2);
    defer alloc.free(w3);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 0 }, w3);
}

test "shortest job first: turnaround time python examples" {
    const alloc = testing.allocator;

    const t1 = try calculateTurnAroundTime(alloc, &[_]i64{ 3, 3, 5, 1 }, 4, &[_]i64{ 0, 3, 5, 0 });
    defer alloc.free(t1);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 6, 10, 1 }, t1);

    const t2 = try calculateTurnAroundTime(alloc, &[_]i64{ 3, 3 }, 2, &[_]i64{ 0, 3 });
    defer alloc.free(t2);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 6 }, t2);

    const t3 = try calculateTurnAroundTime(alloc, &[_]i64{ 8, 10, 1 }, 3, &[_]i64{ 1, 0, 3 });
    defer alloc.free(t3);
    try testing.expectEqualSlices(i64, &[_]i64{ 9, 10, 4 }, t3);
}

test "shortest job first: average time python examples" {
    const avg1 = try calculateAverageTimes(&[_]i64{ 0, 3, 5, 0 }, &[_]i64{ 3, 6, 10, 1 }, 4);
    try testing.expectApproxEqAbs(@as(f64, 2.0), avg1.waiting, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0), avg1.turnaround, 1e-12);

    const avg2 = try calculateAverageTimes(&[_]i64{ 2, 3 }, &[_]i64{ 3, 6 }, 2);
    try testing.expectApproxEqAbs(@as(f64, 2.5), avg2.waiting, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.5), avg2.turnaround, 1e-12);

    const avg3 = try calculateAverageTimes(&[_]i64{ 10, 4, 3 }, &[_]i64{ 2, 7, 6 }, 3);
    try testing.expectApproxEqAbs(@as(f64, 5.666666666666667), avg3.waiting, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0), avg3.turnaround, 1e-12);
}

test "shortest job first: boundary and extreme cases" {
    const alloc = testing.allocator;

    const empty_wait = try calculateWaitingTime(alloc, &[_]i64{}, &[_]i64{}, 0);
    defer alloc.free(empty_wait);
    try testing.expectEqual(@as(usize, 0), empty_wait.len);

    const delayed = try calculateWaitingTime(alloc, &[_]i64{1000}, &[_]i64{1}, 1);
    defer alloc.free(delayed);
    try testing.expectEqualSlices(i64, &[_]i64{0}, delayed);

    try testing.expectError(SjfError.InputLengthMismatch, calculateWaitingTime(alloc, &[_]i64{1}, &[_]i64{}, 1));
    try testing.expectError(SjfError.InvalidBurstTime, calculateWaitingTime(alloc, &[_]i64{1}, &[_]i64{0}, 1));
    try testing.expectError(SjfError.InvalidArrivalTime, calculateWaitingTime(alloc, &[_]i64{-1}, &[_]i64{1}, 1));
    try testing.expectError(SjfError.InvalidProcessCount, calculateAverageTimes(&[_]i64{}, &[_]i64{}, 0));
}
