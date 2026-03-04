//! Decimal to Fraction - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/decimal_to_fraction.py

const std = @import("std");
const testing = std.testing;

pub const DecimalToFractionError = error{ InvalidNumber, Overflow };

pub const Fraction = struct {
    numerator: i128,
    denominator: u128,
};

/// Converts a decimal floating-point value into a reduced fraction.
/// Time complexity: O(d), Space complexity: O(1), where d is decimal digits.
pub fn decimalToFraction(decimal: f64) DecimalToFractionError!Fraction {
    if (!std.math.isFinite(decimal)) return DecimalToFractionError.InvalidNumber;

    var buffer: [128]u8 = undefined;
    const repr = std.fmt.bufPrint(&buffer, "{d}", .{decimal}) catch return DecimalToFractionError.Overflow;
    return fractionFromDecimalRepr(repr);
}

/// Parses a number-like string, then converts to a reduced fraction.
/// Behavior follows Python reference intent: parse float first, then convert.
/// Time complexity: O(d), Space complexity: O(1)
pub fn decimalStringToFraction(input: []const u8) DecimalToFractionError!Fraction {
    const trimmed = std.mem.trim(u8, input, " \t\r\n");
    const parsed = std.fmt.parseFloat(f64, trimmed) catch return DecimalToFractionError.InvalidNumber;
    return decimalToFraction(parsed);
}

fn fractionFromDecimalRepr(repr: []const u8) DecimalToFractionError!Fraction {
    if (repr.len == 0) return DecimalToFractionError.InvalidNumber;

    var idx: usize = 0;
    var negative = false;
    if (repr[idx] == '-') {
        negative = true;
        idx += 1;
        if (idx >= repr.len) return DecimalToFractionError.InvalidNumber;
    } else if (repr[idx] == '+') {
        idx += 1;
        if (idx >= repr.len) return DecimalToFractionError.InvalidNumber;
    }

    const unsigned_repr = repr[idx..];
    const dot_pos_opt = std.mem.indexOfScalar(u8, unsigned_repr, '.');

    if (dot_pos_opt == null) {
        const integer_abs = try parseUnsignedDigits(unsigned_repr);
        return makeFraction(integer_abs, 1, negative);
    }

    const dot_pos = dot_pos_opt.?;
    const integer_part = unsigned_repr[0..dot_pos];
    const fractional_part = unsigned_repr[dot_pos + 1 ..];
    if (fractional_part.len == 0) return DecimalToFractionError.InvalidNumber;

    const integer_abs = if (integer_part.len == 0) 0 else try parseUnsignedDigits(integer_part);
    const fractional_abs = try parseUnsignedDigits(fractional_part);
    const scale = try pow10Checked(@intCast(fractional_part.len));

    const mul = @mulWithOverflow(integer_abs, scale);
    if (mul[1] != 0) return DecimalToFractionError.Overflow;
    const sum = @addWithOverflow(mul[0], fractional_abs);
    if (sum[1] != 0) return DecimalToFractionError.Overflow;

    return makeFraction(sum[0], scale, negative);
}

fn makeFraction(numerator_abs: u128, denominator: u128, negative: bool) DecimalToFractionError!Fraction {
    if (denominator == 0) return DecimalToFractionError.InvalidNumber;

    const divisor = gcd(numerator_abs, denominator);
    const reduced_num = numerator_abs / divisor;
    const reduced_den = denominator / divisor;

    if (reduced_num > std.math.maxInt(i128)) return DecimalToFractionError.Overflow;
    var signed_num: i128 = @intCast(reduced_num);
    if (negative and signed_num != 0) signed_num = -signed_num;

    return .{ .numerator = signed_num, .denominator = reduced_den };
}

fn parseUnsignedDigits(digits: []const u8) DecimalToFractionError!u128 {
    if (digits.len == 0) return DecimalToFractionError.InvalidNumber;

    var value: u128 = 0;
    for (digits) |ch| {
        if (ch < '0' or ch > '9') return DecimalToFractionError.InvalidNumber;

        const mul = @mulWithOverflow(value, @as(u128, 10));
        if (mul[1] != 0) return DecimalToFractionError.Overflow;
        const add = @addWithOverflow(mul[0], @as(u128, ch - '0'));
        if (add[1] != 0) return DecimalToFractionError.Overflow;
        value = add[0];
    }

    return value;
}

fn pow10Checked(exp: u32) DecimalToFractionError!u128 {
    var value: u128 = 1;
    var i: u32 = 0;
    while (i < exp) : (i += 1) {
        const mul = @mulWithOverflow(value, @as(u128, 10));
        if (mul[1] != 0) return DecimalToFractionError.Overflow;
        value = mul[0];
    }
    return value;
}

fn gcd(a: u128, b: u128) u128 {
    var x = a;
    var y = b;
    while (y != 0) {
        const r = x % y;
        x = y;
        y = r;
    }
    return x;
}

fn expectFraction(expected_num: i128, expected_den: u128, actual: Fraction) !void {
    try testing.expectEqual(expected_num, actual.numerator);
    try testing.expectEqual(expected_den, actual.denominator);
}

test "decimal to fraction: python reference examples with numeric input" {
    try expectFraction(2, 1, try decimalToFraction(2));
    try expectFraction(89, 1, try decimalToFraction(89.0));
    try expectFraction(3, 2, try decimalToFraction(1.5));
    try expectFraction(0, 1, try decimalToFraction(0));
    try expectFraction(-5, 2, try decimalToFraction(-2.5));
    try expectFraction(1, 8, try decimalToFraction(0.125));
    try expectFraction(4_000_001, 4, try decimalToFraction(1_000_000.25));
    try expectFraction(13_333, 10_000, try decimalToFraction(1.3333));
}

test "decimal to fraction: python reference examples with string input" {
    try expectFraction(67, 1, try decimalStringToFraction("67"));
    try expectFraction(45, 1, try decimalStringToFraction("45.0"));
    try expectFraction(25, 4, try decimalStringToFraction("6.25"));
    try expectFraction(123, 1, try decimalStringToFraction("1.23e2"));
    try expectFraction(1, 2, try decimalStringToFraction("0.500"));
}

test "decimal to fraction: invalid and extreme cases" {
    try testing.expectError(DecimalToFractionError.InvalidNumber, decimalStringToFraction("78td"));
    try testing.expectError(DecimalToFractionError.InvalidNumber, decimalStringToFraction("NaN"));
    try testing.expectError(DecimalToFractionError.InvalidNumber, decimalToFraction(std.math.inf(f64)));

    try expectFraction(1, 10, try decimalToFraction(0.1));
    try expectFraction(1, 1_000_000, try decimalToFraction(0.000001));
}
