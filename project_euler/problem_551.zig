//! Project Euler Problem 551: Sum of Digits Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_551/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

const ks_min: usize = 2;
const max_k: usize = 20;

const Jump = struct {
    diff: u64,
    dn: u64,
    k: usize,
};

const JumpState = struct {
    diff: u64,
    terms_jumped: u64,
};

const MemoKey = struct {
    ds_b: u16,
    c: u64,
};

const Memo = std.AutoHashMap(MemoKey, std.ArrayList(Jump));

const powers_of_ten = blk: {
    var values: [max_k + 1]u128 = undefined;
    values[0] = 1;
    var index: usize = 1;
    while (index <= max_k) : (index += 1) values[index] = values[index - 1] * 10;
    break :blk values;
};

fn add(allocator: Allocator, digits: *std.ArrayList(u8), k: usize, addend_start: u64) !void {
    while (digits.items.len < k) try digits.append(allocator, 0);

    var addend = addend_start;
    var index = k;
    while (index < digits.items.len) : (index += 1) {
        const sum = digits.items[index] + addend;
        if (sum >= 10) {
            const quotient = sum / 10;
            digits.items[index] = @intCast(sum % 10);
            addend = addend / 10 + quotient;
        } else {
            digits.items[index] = @intCast(sum);
            addend /= 10;
        }
        if (addend == 0) break;
    }

    while (addend > 0) {
        try digits.append(allocator, @intCast(addend % 10));
        addend /= 10;
    }
}

fn compute(allocator: Allocator, digits: *std.ArrayList(u8), k: usize, i: u64, n: u64) !JumpState {
    if (i >= n) return .{ .diff = 0, .terms_jumped = 0 };

    while (digits.items.len < k) try digits.append(allocator, 0);

    const start_i = i;
    var index = i;
    var ds_b: u64 = 0;
    var ds_c: u64 = 0;
    var diff: u64 = 0;

    for (digits.items, 0..) |digit, j| {
        if (j >= k) {
            ds_b += digit;
        } else {
            ds_c += digit;
        }
    }

    while (index < n) {
        index += 1;
        var addend_value = ds_c + ds_b;
        diff += addend_value;
        ds_c = 0;

        var j: usize = 0;
        while (j < k) : (j += 1) {
            const sum = digits.items[j] + addend_value;
            digits.items[j] = @intCast(sum % 10);
            addend_value = sum / 10;
            ds_c += digits.items[j];
        }

        if (addend_value > 0) {
            try add(allocator, digits, k, addend_value);
            break;
        }
    }

    return .{ .diff = diff, .terms_jumped = index - start_i };
}

fn nextTerm(allocator: Allocator, digits: *std.ArrayList(u8), k: usize, i: u64, n: u64, memo: *Memo) !JumpState {
    var ds_b: u16 = 0;
    for (digits.items, 0..) |digit, j| {
        if (j >= k) ds_b += digit;
    }

    var c: u64 = 0;
    const limit = @min(digits.items.len, k);
    for (0..limit) |j| c += @as(u64, digits.items[j]) * @as(u64, @intCast(powers_of_ten[j]));

    var diff: u64 = 0;
    var dn: u64 = 0;
    const max_dn = n - i;

    const key = MemoKey{ .ds_b = ds_b, .c = c };
    {
        const entry = try memo.getOrPut(key);
        if (!entry.found_existing) entry.value_ptr.* = .empty;
    }

    if (memo.getPtr(key).?.items.len > 0) {
        const jumps = memo.getPtr(key).?.items;
        var max_jump_index: ?usize = null;
        var idx: usize = jumps.len;
        while (idx > 0) {
            idx -= 1;
            const jump = jumps[idx];
            if (jump.k <= k and jump.dn <= max_dn) {
                max_jump_index = idx;
                break;
            }
        }

        if (max_jump_index) |jump_index| {
            const jump = jumps[jump_index];
            diff = jump.diff;
            dn = jump.dn;

            var new_c = diff + c;
            const low_limit = @min(k, digits.items.len);
            for (0..low_limit) |j| {
                digits.items[j] = @intCast(new_c % 10);
                new_c /= 10;
            }
            if (new_c > 0) try add(allocator, digits, k, new_c);
        }
    }

    if (dn >= max_dn or @as(u128, c) + diff >= powers_of_ten[k]) {
        return .{ .diff = diff, .terms_jumped = dn };
    }

    if (k > ks_min) {
        while (true) {
            const result = try nextTerm(allocator, digits, k - 1, i + dn, n, memo);
            diff += result.diff;
            dn += result.terms_jumped;

            if (dn >= max_dn or @as(u128, c) + diff >= powers_of_ten[k]) break;
        }
    } else {
        const result = try compute(allocator, digits, k, i + dn, n);
        diff += result.diff;
        dn += result.terms_jumped;
    }

    const jumps_ptr = memo.getPtr(key).?;
    var insert_index: usize = 0;
    while (insert_index < jumps_ptr.items.len) : (insert_index += 1) {
        if (jumps_ptr.items[insert_index].dn > dn) break;
    }
    try jumps_ptr.insert(allocator, insert_index, .{ .diff = diff, .dn = dn, .k = k });

    return .{ .diff = diff, .terms_jumped = dn };
}

fn digitsToInt(digits: []const u8) u64 {
    var result: u64 = 0;
    for (digits, 0..) |digit, index| result += @as(u64, digit) * @as(u64, @intCast(powers_of_ten[index]));
    return result;
}

/// Returns the `n`-th term of the Project Euler 551 sum-of-digits sequence.
/// Time complexity: amortized sublinear via cached jump compression
/// Space complexity: depends on memoized jump states
pub fn solution(allocator: Allocator, n: u64) !u64 {
    if (n == 0) return 1;

    var memo = Memo.init(allocator);
    defer {
        var iterator = memo.iterator();
        while (iterator.next()) |entry| entry.value_ptr.deinit(allocator);
        memo.deinit();
    }

    var digits = try std.ArrayList(u8).initCapacity(allocator, 1);
    defer digits.deinit(allocator);
    try digits.append(allocator, 1);

    const i: u64 = 1;
    var dn: u64 = 0;
    while (true) {
        const result = try nextTerm(allocator, &digits, max_k, i + dn, n, &memo);
        dn += result.terms_jumped;
        if (dn == n - i) break;
    }

    return digitsToInt(digits.items);
}

test "problem 551: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 62), try solution(alloc, 10));
    try testing.expectEqual(@as(u64, 31054319), try solution(alloc, 1_000_000));
    try testing.expectEqual(@as(u64, 73597483551591773), try solution(alloc, 1_000_000_000_000_000));
}

test "problem 551: helper edge cases" {
    const alloc = testing.allocator;

    var digits = try std.ArrayList(u8).initCapacity(alloc, 4);
    defer digits.deinit(alloc);
    try digits.appendSlice(alloc, &[_]u8{ 1, 2, 3 });
    try add(alloc, &digits, 1, 99);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 2, 3 }, digits.items);

    try testing.expectEqual(@as(u64, 3211), digitsToInt(&[_]u8{ 1, 1, 2, 3 }));
    try testing.expectEqual(@as(u64, 1), try solution(alloc, 0));
}
