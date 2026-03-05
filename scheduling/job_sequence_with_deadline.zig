//! Job Sequence With Deadline - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/job_sequence_with_deadline.py

const std = @import("std");
const testing = std.testing;

pub const TaskInfo = struct {
    deadline: i64,
    reward: i64,
};

const Task = struct {
    task_id: usize,
    deadline: i64,
    reward: i64,
};

/// Returns task ids selected by the Python reference greedy filter:
/// sort by reward descending, then keep tasks where deadline >= 1-based position.
///
/// Time complexity: O(n^2) with insertion sort
/// Space complexity: O(n)
pub fn maxTasks(allocator: std.mem.Allocator, tasks_info: []const TaskInfo) std.mem.Allocator.Error![]usize {
    if (tasks_info.len == 0) return allocator.alloc(usize, 0);

    const tasks = try allocator.alloc(Task, tasks_info.len);
    defer allocator.free(tasks);
    for (tasks_info, 0..) |t, i| {
        tasks[i] = Task{
            .task_id = i,
            .deadline = t.deadline,
            .reward = t.reward,
        };
    }

    std.sort.insertion(Task, tasks, {}, struct {
        fn lessThan(_: void, a: Task, b: Task) bool {
            return a.reward > b.reward;
        }
    }.lessThan);

    var selected = std.ArrayListUnmanaged(usize){};
    errdefer selected.deinit(allocator);

    for (tasks, 0..) |task, i| {
        const position: i64 = @intCast(i + 1);
        if (task.deadline >= position) {
            try selected.append(allocator, task.task_id);
        }
    }

    return selected.toOwnedSlice(allocator);
}

test "job sequence with deadline: python examples" {
    const alloc = testing.allocator;

    const r1 = try maxTasks(alloc, &[_]TaskInfo{
        .{ .deadline = 4, .reward = 20 },
        .{ .deadline = 1, .reward = 10 },
        .{ .deadline = 1, .reward = 40 },
        .{ .deadline = 1, .reward = 30 },
    });
    defer alloc.free(r1);
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 0 }, r1);

    const r2 = try maxTasks(alloc, &[_]TaskInfo{
        .{ .deadline = 1, .reward = 10 },
        .{ .deadline = 2, .reward = 20 },
        .{ .deadline = 3, .reward = 30 },
        .{ .deadline = 2, .reward = 40 },
    });
    defer alloc.free(r2);
    try testing.expectEqualSlices(usize, &[_]usize{ 3, 2 }, r2);

    const r3 = try maxTasks(alloc, &[_]TaskInfo{.{ .deadline = 9, .reward = 10 }});
    defer alloc.free(r3);
    try testing.expectEqualSlices(usize, &[_]usize{0}, r3);
}

test "job sequence with deadline: python edge examples" {
    const alloc = testing.allocator;

    const r1 = try maxTasks(alloc, &[_]TaskInfo{.{ .deadline = -9, .reward = 10 }});
    defer alloc.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    const r2 = try maxTasks(alloc, &[_]TaskInfo{});
    defer alloc.free(r2);
    try testing.expectEqual(@as(usize, 0), r2.len);

    const r3 = try maxTasks(alloc, &[_]TaskInfo{
        .{ .deadline = 0, .reward = 10 },
        .{ .deadline = 0, .reward = 20 },
        .{ .deadline = 0, .reward = 30 },
        .{ .deadline = 0, .reward = 40 },
    });
    defer alloc.free(r3);
    try testing.expectEqual(@as(usize, 0), r3.len);

    const r4 = try maxTasks(alloc, &[_]TaskInfo{
        .{ .deadline = -1, .reward = 10 },
        .{ .deadline = -2, .reward = 20 },
        .{ .deadline = -3, .reward = 30 },
        .{ .deadline = -4, .reward = 40 },
    });
    defer alloc.free(r4);
    try testing.expectEqual(@as(usize, 0), r4.len);
}

test "job sequence with deadline: boundary and extreme values" {
    const alloc = testing.allocator;

    const stress = try maxTasks(alloc, &[_]TaskInfo{
        .{ .deadline = 1_000_000, .reward = 5 },
        .{ .deadline = 1_000_001, .reward = 4 },
        .{ .deadline = 1_000_002, .reward = 3 },
        .{ .deadline = 1_000_003, .reward = 2 },
        .{ .deadline = 1_000_004, .reward = 1 },
    });
    defer alloc.free(stress);
    try testing.expectEqual(@as(usize, 5), stress.len);
}
