//! Interest Calculations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/financial/interest.py

const std = @import("std");
const testing = std.testing;

pub const InterestError = error{
    DaysBetweenPaymentsNonPositive,
    DailyInterestRateNegative,
    PrincipalNonPositive,
    NumberOfCompoundingPeriodsNonPositive,
    NominalAnnualInterestRateNegative,
    NumberOfYearsNonPositive,
    NominalAnnualPercentageRateNegative,
};

/// Computes simple interest: principal * daily_rate * days.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn simpleInterest(
    principal: f64,
    daily_interest_rate: f64,
    days_between_payments: f64,
) InterestError!f64 {
    if (days_between_payments <= 0) return InterestError.DaysBetweenPaymentsNonPositive;
    if (daily_interest_rate < 0) return InterestError.DailyInterestRateNegative;
    if (principal <= 0) return InterestError.PrincipalNonPositive;
    return principal * daily_interest_rate * days_between_payments;
}

/// Computes compound interest gain:
/// principal * ((1 + nominal_rate) ^ periods - 1).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn compoundInterest(
    principal: f64,
    nominal_annual_interest_rate_percentage: f64,
    number_of_compounding_periods: f64,
) InterestError!f64 {
    if (number_of_compounding_periods <= 0) return InterestError.NumberOfCompoundingPeriodsNonPositive;
    if (nominal_annual_interest_rate_percentage < 0) return InterestError.NominalAnnualInterestRateNegative;
    if (principal <= 0) return InterestError.PrincipalNonPositive;

    return principal * (std.math.pow(f64, 1 + nominal_annual_interest_rate_percentage, number_of_compounding_periods) - 1);
}

/// Computes APR-based interest by converting APR to daily rate and compounding daily.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn aprInterest(
    principal: f64,
    nominal_annual_percentage_rate: f64,
    number_of_years: f64,
) InterestError!f64 {
    if (number_of_years <= 0) return InterestError.NumberOfYearsNonPositive;
    if (nominal_annual_percentage_rate < 0) return InterestError.NominalAnnualPercentageRateNegative;
    if (principal <= 0) return InterestError.PrincipalNonPositive;

    return compoundInterest(
        principal,
        nominal_annual_percentage_rate / 365.0,
        number_of_years * 365.0,
    );
}

test "interest: simple interest python examples" {
    try testing.expectApproxEqAbs(@as(f64, 3240.0), try simpleInterest(18000.0, 0.06, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.09), try simpleInterest(0.5, 0.06, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1800.0), try simpleInterest(18000.0, 0.01, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try simpleInterest(18000.0, 0.0, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5500.0), try simpleInterest(5500.0, 0.01, 100), 1e-12);
}

test "interest: simple interest invalid inputs" {
    try testing.expectError(InterestError.DailyInterestRateNegative, simpleInterest(10000.0, -0.06, 3));
    try testing.expectError(InterestError.PrincipalNonPositive, simpleInterest(-10000.0, 0.06, 3));
    try testing.expectError(InterestError.DaysBetweenPaymentsNonPositive, simpleInterest(5500.0, 0.01, -5));
}

test "interest: compound interest python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1576.2500000000014), try compoundInterest(10000.0, 0.05, 3), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 500.00000000000045), try compoundInterest(10000.0, 0.05, 1), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.07881250000000006), try compoundInterest(0.5, 0.05, 3), 1e-12);
}

test "interest: compound interest invalid inputs" {
    try testing.expectError(InterestError.NumberOfCompoundingPeriodsNonPositive, compoundInterest(10000.0, 0.06, -4));
    try testing.expectError(InterestError.NominalAnnualInterestRateNegative, compoundInterest(10000.0, -3.5, 3.0));
    try testing.expectError(InterestError.PrincipalNonPositive, compoundInterest(-5500.0, 0.01, 5));
}

test "interest: apr interest python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1618.223072263547), try aprInterest(10000.0, 0.05, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 512.6749646744732), try aprInterest(10000.0, 0.05, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.08091115361317736), try aprInterest(0.5, 0.05, 3), 1e-12);
}

test "interest: apr interest invalid inputs" {
    try testing.expectError(InterestError.NumberOfYearsNonPositive, aprInterest(10000.0, 0.06, -4));
    try testing.expectError(InterestError.NominalAnnualPercentageRateNegative, aprInterest(10000.0, -3.5, 3.0));
    try testing.expectError(InterestError.PrincipalNonPositive, aprInterest(-5500.0, 0.01, 5));
}

test "interest: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 1e8), try simpleInterest(1e8, 1e-6, 1e6), 1e-3);

    const long_term = try compoundInterest(1e5, 1e-4, 100000);
    try testing.expect(std.math.isFinite(long_term));
    try testing.expect(long_term > 0);

    const apr_long = try aprInterest(1e4, 0.15, 50);
    try testing.expect(std.math.isFinite(apr_long));
    try testing.expect(apr_long > 0);
}
