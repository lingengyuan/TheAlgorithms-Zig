//! Deterministic Miller-Rabin - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/deterministic_miller_rabin.py

const std = @import("std");
const testing = std.testing;

pub const MillerRabinError = error{UpperBoundExceeded};

const DETERMINISTIC_UPPER_BOUND: u128 = 3_317_044_064_679_887_385_961_981;

fn mulMod(a_init: u128, b_init: u128, modulus: u128) u128 {
    var a = a_init % modulus;
    var b = b_init;
    var result: u128 = 0;

    while (b > 0) {
        if ((b & 1) == 1) result = (result + a) % modulus;
        a = (a << 1) % modulus;
        b >>= 1;
    }

    return result;
}

fn powMod(base_init: u128, exp_init: u128, modulus: u128) u128 {
    if (modulus == 1) return 0;

    var base = base_init % modulus;
    var exp = exp_init;
    var result: u128 = 1;

    while (exp > 0) {
        if ((exp & 1) == 1) result = mulMod(result, base, modulus);
        base = mulMod(base, base, modulus);
        exp >>= 1;
    }

    return result;
}

/// Deterministic Miller-Rabin for n <= 3_317_044_064_679_887_385_961_981.
/// If allow_probable is true, larger n is tested probabilistically using same bases.
/// Time complexity: O(k * log^3 n) using modular multiplication, Space complexity: O(1)
pub fn millerRabin(n: u128, allow_probable: bool) !bool {
    if (n == 2) return true;
    if ((n & 1) == 0 or n < 2) return false;

    if (n > 5) {
        const last_digit = n % 10;
        if (last_digit != 1 and last_digit != 3 and last_digit != 7 and last_digit != 9) return false;
    }

    if (n > DETERMINISTIC_UPPER_BOUND and !allow_probable) {
        return MillerRabinError.UpperBoundExceeded;
    }

    const bounds = [_]u128{
        2_047,
        1_373_653,
        25_326_001,
        3_215_031_751,
        2_152_302_898_747,
        3_474_749_660_383,
        341_550_071_728_321,
        1,
        3_825_123_056_546_413_051,
        1,
        1,
        318_665_857_834_031_151_167_461,
        DETERMINISTIC_UPPER_BOUND,
    };

    const primes = [_]u128{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41 };

    var stop_idx: usize = primes.len;
    for (bounds, 1..) |bound, idx| {
        if (n < bound) {
            stop_idx = idx;
            break;
        }
    }

    var d = n - 1;
    var s: u32 = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s += 1;
    }

    for (primes[0..stop_idx]) |prime| {
        var probable = false;

        var r: u32 = 0;
        while (r < s) : (r += 1) {
            const exp = d << @as(u7, @intCast(r));
            const m = powMod(prime, exp, n);
            if ((r == 0 and m == 1) or ((m + 1) % n == 0)) {
                probable = true;
                break;
            }
        }

        if (!probable) return false;
    }

    return true;
}

test "deterministic miller rabin: basic cases" {
    try testing.expect(!try millerRabin(1, false));
    try testing.expect(try millerRabin(2, false));
    try testing.expect(!try millerRabin(4, false));
    try testing.expect(try millerRabin(563, false));
    try testing.expect(!try millerRabin(561, false));
}

test "deterministic miller rabin: range samples from python" {
    try testing.expect(!try millerRabin(838_201, false));
    try testing.expect(try millerRabin(838_207, false));

    try testing.expect(!try millerRabin(3_078_386_641, false));
    try testing.expect(try millerRabin(3_078_386_653, false));

    try testing.expect(!try millerRabin(2_779_799_728_307, false));
    try testing.expect(try millerRabin(2_779_799_728_327, false));

    try testing.expect(!try millerRabin(552_840_677_446_647_897_660_333, false));
    try testing.expect(try millerRabin(552_840_677_446_647_897_660_359, false));
}

test "deterministic miller rabin: upper bound behavior" {
    const above = DETERMINISTIC_UPPER_BOUND + 2;
    try testing.expectError(MillerRabinError.UpperBoundExceeded, millerRabin(above, false));

    // With allow_probable=true, function should return a boolean instead of error.
    _ = try millerRabin(above, true);
}
