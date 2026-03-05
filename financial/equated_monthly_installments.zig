//! Equated Monthly Installments - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/financial/equated_monthly_installments.py

const std = @import("std");
const testing = std.testing;

pub const EmiError = error{
    PrincipalMustBePositive,
    RateMustBeNonNegative,
    YearsMustBePositive,
    UndefinedAtZeroRate,
};

/// Computes EMI using:
/// A = p * r * (1 + r)^n / ((1 + r)^n - 1),
/// where r is monthly interest rate and n is payment count.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn equatedMonthlyInstallments(
    principal: f64,
    rate_per_annum: f64,
    years_to_repay: i64,
) EmiError!f64 {
    if (principal <= 0) return EmiError.PrincipalMustBePositive;
    if (rate_per_annum < 0) return EmiError.RateMustBeNonNegative;
    if (years_to_repay <= 0) return EmiError.YearsMustBePositive;

    const rate_per_month = rate_per_annum / 12.0;
    const number_of_payments = @as(f64, @floatFromInt(years_to_repay * 12));
    const growth = std.math.pow(f64, 1.0 + rate_per_month, number_of_payments);
    const denominator = growth - 1.0;
    if (denominator == 0.0) return EmiError.UndefinedAtZeroRate;

    return principal * rate_per_month * growth / denominator;
}

test "equated monthly installments: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 830.3577453212793), try equatedMonthlyInstallments(25000, 0.12, 3), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 358.67737100646826), try equatedMonthlyInstallments(25000, 0.12, 10), 1e-9);

    try testing.expectError(EmiError.PrincipalMustBePositive, equatedMonthlyInstallments(0, 0.12, 3));
    try testing.expectError(EmiError.RateMustBeNonNegative, equatedMonthlyInstallments(25000, -1, 3));
    try testing.expectError(EmiError.YearsMustBePositive, equatedMonthlyInstallments(25000, 0.12, 0));
}

test "equated monthly installments: boundary and extreme values" {
    try testing.expectError(EmiError.UndefinedAtZeroRate, equatedMonthlyInstallments(25000, 0.0, 3));

    const long_term = try equatedMonthlyInstallments(1e8, 0.2, 40);
    try testing.expect(std.math.isFinite(long_term));
    try testing.expect(long_term > 0);
}
