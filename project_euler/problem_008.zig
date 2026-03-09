//! Project Euler Problem 8: Largest Product in a Series - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_008/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem008Error = error{
    InvalidDigit,
};

pub const N =
    "73167176531330624919225119674426574742355349194934" ++
    "96983520312774506326239578318016984801869478851843" ++
    "85861560789112949495459501737958331952853208805511" ++
    "12540698747158523863050715693290963295227443043557" ++
    "66896648950445244523161731856403098711121722383113" ++
    "62229893423380308135336276614282806444486645238749" ++
    "30358907296290491560440772390713810515859307960866" ++
    "70172427121883998797908792274921901699720888093776" ++
    "65727333001053367881220235421809751254540594752243" ++
    "52584907711670556013604839586446706324415722155397" ++
    "53697817977846174064955149290862569321978468622482" ++
    "83972241375657056057490261407972968652414535100474" ++
    "82166370484403199890008895243450658541227588666881" ++
    "16427171479924442928230863465674813919123162824586" ++
    "17866458359124566529476545682848912883142607690042" ++
    "24219022671055626321111109370544217506941658960408" ++
    "07198403850962455444362981230987879927244284909188" ++
    "84580156166097919133875499200524063689912560717606" ++
    "05886116467109405077541002256983155200055935729725" ++
    "71636269561882670428252483600823257530420752963450";

/// Returns the greatest product of thirteen adjacent digits.
/// Python-reference edge behavior: if input length < 13, returns min i64.
///
/// Time complexity: O(n * 13)
/// Space complexity: O(1)
pub fn solution(input: []const u8) Problem008Error!i64 {
    if (input.len < 13) {
        return std.math.minInt(i64);
    }

    var largest: i64 = std.math.minInt(i64);
    var i: usize = 0;
    while (i <= input.len - 13) : (i += 1) {
        var product: i64 = 1;
        for (0..13) |j| {
            const ch = input[i + j];
            if (ch < '0' or ch > '9') return Problem008Error.InvalidDigit;
            product *= @as(i64, ch - '0');
        }
        if (product > largest) largest = product;
    }

    return largest;
}

test "problem 008: python examples" {
    try testing.expectEqual(@as(i64, 609638400), try solution("13978431290823798458352374"));
    try testing.expectEqual(@as(i64, 2612736000), try solution("13978431295823798458352374"));
    try testing.expectEqual(@as(i64, 209018880), try solution("1397843129582379841238352374"));
}

test "problem 008: boundaries and official case" {
    try testing.expectEqual(std.math.minInt(i64), try solution("1234"));
    try testing.expectError(Problem008Error.InvalidDigit, solution("123456789012x4567890"));

    try testing.expectEqual(@as(i64, 23514624000), try solution(N));

    // Extreme: long repeated digits
    const repeated = "9" ** 20_000;
    try testing.expectEqual(@as(i64, 2_541_865_828_329), try solution(repeated));
}
