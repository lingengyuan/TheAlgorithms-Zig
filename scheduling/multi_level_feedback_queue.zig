//! Multi Level Feedback Queue - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/multi_level_feedback_queue.py

const std = @import("std");
const testing = std.testing;

pub const MlfqError = error{
    InvalidQueueCount,
    InvalidTimeSliceCount,
};

pub const Process = struct {
    process_name: []const u8,
    arrival_time: i64,
    stop_time: i64,
    burst_time: i64,
    waiting_time: i64,
    turnaround_time: i64,

    pub fn init(process_name: []const u8, arrival_time: i64, burst_time: i64) Process {
        return Process{
            .process_name = process_name,
            .arrival_time = arrival_time,
            .stop_time = arrival_time,
            .burst_time = burst_time,
            .waiting_time = 0,
            .turnaround_time = 0,
        };
    }
};

pub const MLFQ = struct {
    allocator: std.mem.Allocator,
    number_of_queues: usize,
    time_slices: []const i64,
    processes: []Process,
    current_time: i64,
    ready_queue: std.ArrayListUnmanaged(usize),
    finish_queue: std.ArrayListUnmanaged(usize),

    pub fn init(
        allocator: std.mem.Allocator,
        number_of_queues: usize,
        time_slices: []const i64,
        processes: []Process,
        initial_ready_queue: []const usize,
        current_time: i64,
    ) (std.mem.Allocator.Error || MlfqError)!MLFQ {
        if (number_of_queues == 0) return MlfqError.InvalidQueueCount;
        if (time_slices.len != number_of_queues - 1) return MlfqError.InvalidTimeSliceCount;

        var ready = std.ArrayListUnmanaged(usize){};
        errdefer ready.deinit(allocator);
        try ready.appendSlice(allocator, initial_ready_queue);

        return MLFQ{
            .allocator = allocator,
            .number_of_queues = number_of_queues,
            .time_slices = time_slices,
            .processes = processes,
            .current_time = current_time,
            .ready_queue = ready,
            .finish_queue = .{},
        };
    }

    pub fn deinit(self: *MLFQ) void {
        self.ready_queue.deinit(self.allocator);
        self.finish_queue.deinit(self.allocator);
    }

    /// Returns sequence of finished process names.
    pub fn calculateSequenceOfFinishQueue(self: *const MLFQ, allocator: std.mem.Allocator) std.mem.Allocator.Error![][]const u8 {
        const sequence = try allocator.alloc([]const u8, self.finish_queue.items.len);
        for (self.finish_queue.items, 0..) |idx, i| {
            sequence[i] = self.processes[idx].process_name;
        }
        return sequence;
    }

    pub fn calculateWaitingTime(self: *const MLFQ, allocator: std.mem.Allocator, queue: []const usize) std.mem.Allocator.Error![]i64 {
        const waiting_times = try allocator.alloc(i64, queue.len);
        for (queue, 0..) |idx, i| waiting_times[i] = self.processes[idx].waiting_time;
        return waiting_times;
    }

    pub fn calculateTurnaroundTime(self: *const MLFQ, allocator: std.mem.Allocator, queue: []const usize) std.mem.Allocator.Error![]i64 {
        const turnaround_times = try allocator.alloc(i64, queue.len);
        for (queue, 0..) |idx, i| turnaround_times[i] = self.processes[idx].turnaround_time;
        return turnaround_times;
    }

    pub fn calculateCompletionTime(self: *const MLFQ, allocator: std.mem.Allocator, queue: []const usize) std.mem.Allocator.Error![]i64 {
        const completion_times = try allocator.alloc(i64, queue.len);
        for (queue, 0..) |idx, i| completion_times[i] = self.processes[idx].stop_time;
        return completion_times;
    }

    pub fn calculateRemainingBurstTimeOfProcesses(
        self: *const MLFQ,
        allocator: std.mem.Allocator,
        queue: []const usize,
    ) std.mem.Allocator.Error![]i64 {
        const remaining = try allocator.alloc(i64, queue.len);
        for (queue, 0..) |idx, i| remaining[i] = self.processes[idx].burst_time;
        return remaining;
    }

    /// Updates waiting time for a process:
    /// waiting_time += current_time - stop_time.
    pub fn updateWaitingTime(self: *MLFQ, process_index: usize) i64 {
        var p = &self.processes[process_index];
        p.waiting_time += self.current_time - p.stop_time;
        return p.waiting_time;
    }

    /// FCFS for the last queue.
    pub fn firstComeFirstServed(self: *MLFQ, queue: []const usize) std.mem.Allocator.Error!void {
        for (queue) |idx| {
            var p = &self.processes[idx];

            if (self.current_time < p.arrival_time) {
                self.current_time = p.arrival_time;
            }

            _ = self.updateWaitingTime(idx);
            self.current_time += p.burst_time;
            p.burst_time = 0;
            p.turnaround_time = self.current_time - p.arrival_time;
            p.stop_time = self.current_time;
            try self.finish_queue.append(self.allocator, idx);
        }
    }

    /// One RR cycle for the queue and given time slice.
    /// Returns remaining process indices (unfinished).
    pub fn roundRobin(self: *MLFQ, queue: []const usize, time_slice: i64) std.mem.Allocator.Error![]usize {
        var remaining = std.ArrayListUnmanaged(usize){};
        errdefer remaining.deinit(self.allocator);

        for (queue) |idx| {
            var p = &self.processes[idx];

            if (self.current_time < p.arrival_time) {
                self.current_time = p.arrival_time;
            }

            _ = self.updateWaitingTime(idx);

            if (p.burst_time > time_slice) {
                self.current_time += time_slice;
                p.burst_time -= time_slice;
                p.stop_time = self.current_time;
                try remaining.append(self.allocator, idx);
            } else {
                self.current_time += p.burst_time;
                p.burst_time = 0;
                p.stop_time = self.current_time;
                p.turnaround_time = self.current_time - p.arrival_time;
                try self.finish_queue.append(self.allocator, idx);
            }
        }

        return remaining.toOwnedSlice(self.allocator);
    }

    /// Executes full MLFQ scheduling:
    /// - RR on queues [0..N-2]
    /// - FCFS on queue [N-1]
    pub fn multiLevelFeedbackQueue(self: *MLFQ) std.mem.Allocator.Error![]const usize {
        var ready = try self.allocator.dupe(usize, self.ready_queue.items);
        var i: usize = 0;
        while (i < self.number_of_queues - 1) : (i += 1) {
            const next_ready = try self.roundRobin(ready, self.time_slices[i]);
            self.allocator.free(ready);
            ready = next_ready;
        }
        defer self.allocator.free(ready);

        try self.firstComeFirstServed(ready);
        self.ready_queue.clearRetainingCapacity();
        return self.finish_queue.items;
    }
};

test "mlfq: python full run examples" {
    const alloc = testing.allocator;
    var processes = [_]Process{
        Process.init("P1", 0, 53),
        Process.init("P2", 0, 17),
        Process.init("P3", 0, 68),
        Process.init("P4", 0, 24),
    };

    var mlfq = try MLFQ.init(
        alloc,
        3,
        &[_]i64{ 17, 25 },
        processes[0..],
        &[_]usize{ 0, 1, 2, 3 },
        0,
    );
    defer mlfq.deinit();

    _ = try mlfq.multiLevelFeedbackQueue();

    const sequence = try mlfq.calculateSequenceOfFinishQueue(alloc);
    defer alloc.free(sequence);
    try testing.expectEqual(@as(usize, 4), sequence.len);
    try testing.expectEqualStrings("P2", sequence[0]);
    try testing.expectEqualStrings("P4", sequence[1]);
    try testing.expectEqualStrings("P1", sequence[2]);
    try testing.expectEqualStrings("P3", sequence[3]);

    const wt = try mlfq.calculateWaitingTime(alloc, &[_]usize{ 0, 1, 2, 3 });
    defer alloc.free(wt);
    try testing.expectEqualSlices(i64, &[_]i64{ 83, 17, 94, 101 }, wt);

    const tat = try mlfq.calculateTurnaroundTime(alloc, &[_]usize{ 0, 1, 2, 3 });
    defer alloc.free(tat);
    try testing.expectEqualSlices(i64, &[_]i64{ 136, 34, 162, 125 }, tat);

    const ct = try mlfq.calculateCompletionTime(alloc, &[_]usize{ 0, 1, 2, 3 });
    defer alloc.free(ct);
    try testing.expectEqualSlices(i64, &[_]i64{ 136, 34, 162, 125 }, ct);
}

test "mlfq: python round robin intermediate examples" {
    const alloc = testing.allocator;
    var processes = [_]Process{
        Process.init("P1", 0, 53),
        Process.init("P2", 0, 17),
        Process.init("P3", 0, 68),
        Process.init("P4", 0, 24),
    };

    var mlfq = try MLFQ.init(
        alloc,
        3,
        &[_]i64{ 17, 25 },
        processes[0..],
        &[_]usize{ 0, 1, 2, 3 },
        0,
    );
    defer mlfq.deinit();

    const ready_after_rr1 = try mlfq.roundRobin(mlfq.ready_queue.items, 17);
    defer alloc.free(ready_after_rr1);

    const seq_after_rr1 = try mlfq.calculateSequenceOfFinishQueue(alloc);
    defer alloc.free(seq_after_rr1);
    try testing.expectEqual(@as(usize, 1), seq_after_rr1.len);
    try testing.expectEqualStrings("P2", seq_after_rr1[0]);

    const finish_burst_rr1 = try mlfq.calculateRemainingBurstTimeOfProcesses(alloc, mlfq.finish_queue.items);
    defer alloc.free(finish_burst_rr1);
    try testing.expectEqualSlices(i64, &[_]i64{0}, finish_burst_rr1);

    const ready_burst_rr1 = try mlfq.calculateRemainingBurstTimeOfProcesses(alloc, ready_after_rr1);
    defer alloc.free(ready_burst_rr1);
    try testing.expectEqualSlices(i64, &[_]i64{ 36, 51, 7 }, ready_burst_rr1);

    const ready_after_rr2 = try mlfq.roundRobin(ready_after_rr1, 25);
    defer alloc.free(ready_after_rr2);

    const seq_after_rr2 = try mlfq.calculateSequenceOfFinishQueue(alloc);
    defer alloc.free(seq_after_rr2);
    try testing.expectEqual(@as(usize, 2), seq_after_rr2.len);
    try testing.expectEqualStrings("P2", seq_after_rr2[0]);
    try testing.expectEqualStrings("P4", seq_after_rr2[1]);

    const finish_burst_rr2 = try mlfq.calculateRemainingBurstTimeOfProcesses(alloc, mlfq.finish_queue.items);
    defer alloc.free(finish_burst_rr2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0 }, finish_burst_rr2);

    const ready_burst_rr2 = try mlfq.calculateRemainingBurstTimeOfProcesses(alloc, ready_after_rr2);
    defer alloc.free(ready_burst_rr2);
    try testing.expectEqualSlices(i64, &[_]i64{ 11, 26 }, ready_burst_rr2);
}

test "mlfq: boundary and validation cases" {
    const alloc = testing.allocator;
    var processes = [_]Process{
        Process.init("A", 0, 2),
        Process.init("B", 0, 1),
    };

    try testing.expectError(
        MlfqError.InvalidQueueCount,
        MLFQ.init(alloc, 0, &[_]i64{}, processes[0..], &[_]usize{ 0, 1 }, 0),
    );
    try testing.expectError(
        MlfqError.InvalidTimeSliceCount,
        MLFQ.init(alloc, 3, &[_]i64{1}, processes[0..], &[_]usize{ 0, 1 }, 0),
    );

    var only_fcfs = try MLFQ.init(
        alloc,
        1,
        &[_]i64{},
        processes[0..],
        &[_]usize{ 0, 1 },
        0,
    );
    defer only_fcfs.deinit();
    _ = try only_fcfs.multiLevelFeedbackQueue();
    const seq = try only_fcfs.calculateSequenceOfFinishQueue(alloc);
    defer alloc.free(seq);
    try testing.expectEqual(@as(usize, 2), seq.len);
    try testing.expectEqualStrings("A", seq[0]);
    try testing.expectEqualStrings("B", seq[1]);

    var delayed_processes = [_]Process{
        Process.init("A", 5, 2),
        Process.init("B", 10, 1),
    };
    var delayed = try MLFQ.init(
        alloc,
        1,
        &[_]i64{},
        delayed_processes[0..],
        &[_]usize{ 0, 1 },
        0,
    );
    defer delayed.deinit();
    _ = try delayed.multiLevelFeedbackQueue();
    const delayed_ct = try delayed.calculateCompletionTime(alloc, &[_]usize{ 0, 1 });
    defer alloc.free(delayed_ct);
    try testing.expectEqualSlices(i64, &[_]i64{ 7, 11 }, delayed_ct);
}
