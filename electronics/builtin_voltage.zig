//! Builtin Voltage - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/builtin_voltage.py

const std = @import("std");
const testing = std.testing;

pub const BuiltinVoltageError = error{
    DonorConcentrationNotPositive,
    AcceptorConcentrationNotPositive,
    IntrinsicConcentrationNotPositive,
    DonorNotGreaterThanIntrinsic,
    AcceptorNotGreaterThanIntrinsic,
};

const boltzmann: f64 = 1.380649e-23;
const electron_volt: f64 = 1.602176634e-19;
const default_temperature_kelvin: f64 = 300.0;

/// Computes the built-in voltage of a pn junction:
/// V_bi = (kT/q) * ln((N_d * N_a) / n_i^2).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn builtinVoltage(
    donor_conc: f64,
    acceptor_conc: f64,
    intrinsic_conc: f64,
) BuiltinVoltageError!f64 {
    if (donor_conc <= 0) {
        return BuiltinVoltageError.DonorConcentrationNotPositive;
    }
    if (acceptor_conc <= 0) {
        return BuiltinVoltageError.AcceptorConcentrationNotPositive;
    }
    if (intrinsic_conc <= 0) {
        return BuiltinVoltageError.IntrinsicConcentrationNotPositive;
    }
    if (donor_conc <= intrinsic_conc) {
        return BuiltinVoltageError.DonorNotGreaterThanIntrinsic;
    }
    if (acceptor_conc <= intrinsic_conc) {
        return BuiltinVoltageError.AcceptorNotGreaterThanIntrinsic;
    }

    // Use logarithm properties for better numerical stability on extreme magnitudes.
    const ln_term = @log(donor_conc) + @log(acceptor_conc) - 2.0 * @log(intrinsic_conc);
    return (boltzmann * default_temperature_kelvin * ln_term) / electron_volt;
}

test "builtin voltage: python examples and validation" {
    try testing.expectApproxEqAbs(@as(f64, 0.833370010652644), try builtinVoltage(1e17, 1e17, 1e10), 1e-12);

    try testing.expectError(BuiltinVoltageError.DonorConcentrationNotPositive, builtinVoltage(0, 1600, 200));
    try testing.expectError(BuiltinVoltageError.AcceptorConcentrationNotPositive, builtinVoltage(1000, 0, 1200));
    try testing.expectError(BuiltinVoltageError.IntrinsicConcentrationNotPositive, builtinVoltage(1000, 1000, 0));
    try testing.expectError(BuiltinVoltageError.DonorNotGreaterThanIntrinsic, builtinVoltage(1000, 3000, 2000));
    try testing.expectError(BuiltinVoltageError.AcceptorNotGreaterThanIntrinsic, builtinVoltage(3000, 1000, 2000));
}

test "builtin voltage: boundary and extreme values" {
    const near_threshold = try builtinVoltage(2000.0001, 3000.0, 2000.0);
    try testing.expect(near_threshold > 0);

    const extreme = try builtinVoltage(1e308, 1e307, 1e-300);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
