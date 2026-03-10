//! Pi Generator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/pi_generator.py

const std = @import("std");
const testing = std.testing;
const BigInt = std.math.big.int.Managed;

const Allocator = std.mem.Allocator;

pub const PiGeneratorError = error{OutOfMemory};

/// Generates pi as a decimal string with `limit` digits after the decimal point.
/// Caller owns the returned slice.
/// Time complexity: superlinear in `limit` due to arbitrary-precision arithmetic.
/// Space complexity: O(limit)
pub fn calculatePi(allocator: Allocator, limit: usize) PiGeneratorError![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var q = try BigInt.initSet(a, 1);
    var r = try BigInt.initSet(a, 0);
    var t = try BigInt.initSet(a, 1);
    var k = try BigInt.initSet(a, 1);
    var n = try BigInt.initSet(a, 3);
    var m = try BigInt.initSet(a, 3);

    var counter: usize = 0;
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    while (counter != limit + 1) {
        if (try shouldEmitDigit(a, &q, &r, &t, &n)) {
            const digit = n.toInt(u8) catch unreachable;
            try result.append(allocator, '0' + digit);
            if (counter == 0) try result.append(allocator, '.');
            if (limit == counter) break;

            counter += 1;

            const nt = try mulBig(a, &n, &t);
            const r_minus_nt = try subBig(a, &r, &nt);
            const nr = try mulScalarBig(a, &r_minus_nt, 10);

            const three_q = try mulScalarBig(a, &q, 3);
            const three_q_plus_r = try addBig(a, &three_q, &r);
            const scaled_numerator = try mulScalarBig(a, &three_q_plus_r, 10);
            const quotient = try divTruncBig(a, &scaled_numerator, &t);
            const ten_n = try mulScalarBig(a, &n, 10);

            q = try mulScalarBig(a, &q, 10);
            r = nr;
            n = try subBig(a, &quotient, &ten_n);
        } else {
            const two_q = try mulScalarBig(a, &q, 2);
            const two_q_plus_r = try addBig(a, &two_q, &r);
            const nr = try mulBig(a, &two_q_plus_r, &m);

            const seven_k = try mulScalarBig(a, &k, 7);
            const q_times_seven_k = try mulBig(a, &q, &seven_k);
            const q_times_seven_k_plus_two = try addScalarBig(a, &q_times_seven_k, 2);
            const r_times_m = try mulBig(a, &r, &m);
            const numerator = try addBig(a, &q_times_seven_k_plus_two, &r_times_m);
            const denominator = try mulBig(a, &t, &m);
            const quotient = try divTruncBig(a, &numerator, &denominator);

            q = try mulBig(a, &q, &k);
            t = try mulBig(a, &t, &m);
            m = try addScalarBig(a, &m, 2);
            k = try addScalarBig(a, &k, 1);
            n = quotient;
            r = nr;
        }
    }

    return try result.toOwnedSlice(allocator);
}

fn shouldEmitDigit(a: Allocator, q: *const BigInt, r: *const BigInt, t: *const BigInt, n: *const BigInt) PiGeneratorError!bool {
    const four_q = try mulScalarBig(a, q, 4);
    const four_q_plus_r = try addBig(a, &four_q, r);
    const lhs = try subBig(a, &four_q_plus_r, t);
    const rhs = try mulBig(a, n, t);
    return lhs.order(rhs) == .lt;
}

fn addBig(a: Allocator, left: *const BigInt, right: *const BigInt) PiGeneratorError!BigInt {
    var result = try BigInt.init(a);
    try result.add(left, right);
    return result;
}

fn addScalarBig(a: Allocator, value: *const BigInt, scalar: anytype) PiGeneratorError!BigInt {
    var result = try BigInt.init(a);
    try result.addScalar(value, scalar);
    return result;
}

fn subBig(a: Allocator, left: *const BigInt, right: *const BigInt) PiGeneratorError!BigInt {
    var result = try BigInt.init(a);
    try result.sub(left, right);
    return result;
}

fn mulBig(a: Allocator, left: *const BigInt, right: *const BigInt) PiGeneratorError!BigInt {
    var result = try BigInt.init(a);
    try result.mul(left, right);
    return result;
}

fn mulScalarBig(a: Allocator, value: *const BigInt, scalar: anytype) PiGeneratorError!BigInt {
    var scalar_big = try BigInt.initSet(a, scalar);
    return mulBig(a, value, &scalar_big);
}

fn divTruncBig(a: Allocator, numerator: *const BigInt, denominator: *const BigInt) PiGeneratorError!BigInt {
    var quotient = try BigInt.init(a);
    var remainder = try BigInt.init(a);
    try quotient.divTrunc(&remainder, numerator, denominator);
    return quotient;
}

test "pi generator: python reference exact strings" {
    const alloc = testing.allocator;

    const p0 = try calculatePi(alloc, 0);
    defer alloc.free(p0);
    try testing.expectEqualStrings("3.", p0);

    const p15 = try calculatePi(alloc, 15);
    defer alloc.free(p15);
    try testing.expectEqualStrings("3.141592653589793", p15);

    const p50 = try calculatePi(alloc, 50);
    defer alloc.free(p50);
    try testing.expectEqualStrings("3.14159265358979323846264338327950288419716939937510", p50);

    const p80 = try calculatePi(alloc, 80);
    defer alloc.free(p80);
    try testing.expectEqualStrings("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899", p80);
}

test "pi generator: prefix and length invariants" {
    const alloc = testing.allocator;
    const p100 = try calculatePi(alloc, 100);
    defer alloc.free(p100);
    try testing.expectEqual(@as(usize, 102), p100.len);
    try testing.expectEqualStrings("3.14159265358979323846264338327950288419716939937510", p100[0..52]);
}
