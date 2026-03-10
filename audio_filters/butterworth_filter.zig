//! Butterworth Filters - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/audio_filters/butterworth_filter.py

const std = @import("std");
const testing = std.testing;
const iir = @import("iir_filter.zig");

pub const ButterworthError = iir.IIRFilterError || std.mem.Allocator.Error || error{
    InvalidFrequency,
    InvalidSamplerate,
    InvalidQFactor,
    InvalidGain,
};

fn defaultQFactor() f64 {
    return 1.0 / std.math.sqrt(2.0);
}

fn validateCommon(frequency: f64, samplerate: f64, q_factor: f64) ButterworthError!void {
    if (!std.math.isFinite(frequency) or frequency <= 0.0) return error.InvalidFrequency;
    if (!std.math.isFinite(samplerate) or samplerate <= 0.0) return error.InvalidSamplerate;
    if (!std.math.isFinite(q_factor) or q_factor <= 0.0) return error.InvalidQFactor;
}

fn validateGain(gain_db: f64) ButterworthError!void {
    if (!std.math.isFinite(gain_db)) return error.InvalidGain;
}

fn makeBiquad(
    allocator: std.mem.Allocator,
    a_coeffs: [3]f64,
    b_coeffs: [3]f64,
) ButterworthError!iir.IIRFilter {
    var filt = try iir.IIRFilter.init(allocator, 2);
    errdefer filt.deinit();
    try filt.setCoefficients(&a_coeffs, &b_coeffs);
    return filt;
}

/// Creates a Butterworth low-pass filter.
/// Time complexity: O(1), Space complexity: O(1)
pub fn makeLowpass(
    allocator: std.mem.Allocator,
    frequency: f64,
    samplerate: f64,
    q_factor: f64,
) ButterworthError!iir.IIRFilter {
    try validateCommon(frequency, samplerate, q_factor);
    const w0 = std.math.tau * frequency / samplerate;
    const s = @sin(w0);
    const c = @cos(w0);
    const alpha = s / (2.0 * q_factor);

    const b0 = (1.0 - c) / 2.0;
    const b1 = 1.0 - c;
    const a0 = 1.0 + alpha;
    const a1 = -2.0 * c;
    const a2 = 1.0 - alpha;

    return makeBiquad(allocator, .{ a0, a1, a2 }, .{ b0, b1, b0 });
}

/// Creates a Butterworth high-pass filter.
pub fn makeHighpass(
    allocator: std.mem.Allocator,
    frequency: f64,
    samplerate: f64,
    q_factor: f64,
) ButterworthError!iir.IIRFilter {
    try validateCommon(frequency, samplerate, q_factor);
    const w0 = std.math.tau * frequency / samplerate;
    const s = @sin(w0);
    const c = @cos(w0);
    const alpha = s / (2.0 * q_factor);

    const b0 = (1.0 + c) / 2.0;
    const b1 = -1.0 - c;
    const a0 = 1.0 + alpha;
    const a1 = -2.0 * c;
    const a2 = 1.0 - alpha;

    return makeBiquad(allocator, .{ a0, a1, a2 }, .{ b0, b1, b0 });
}

/// Creates a Butterworth band-pass filter.
pub fn makeBandpass(
    allocator: std.mem.Allocator,
    frequency: f64,
    samplerate: f64,
    q_factor: f64,
) ButterworthError!iir.IIRFilter {
    try validateCommon(frequency, samplerate, q_factor);
    const w0 = std.math.tau * frequency / samplerate;
    const s = @sin(w0);
    const c = @cos(w0);
    const alpha = s / (2.0 * q_factor);

    const b0 = s / 2.0;
    const a0 = 1.0 + alpha;
    const a1 = -2.0 * c;
    const a2 = 1.0 - alpha;

    return makeBiquad(allocator, .{ a0, a1, a2 }, .{ b0, 0.0, -b0 });
}

/// Creates an all-pass filter.
pub fn makeAllpass(
    allocator: std.mem.Allocator,
    frequency: f64,
    samplerate: f64,
    q_factor: f64,
) ButterworthError!iir.IIRFilter {
    try validateCommon(frequency, samplerate, q_factor);
    const w0 = std.math.tau * frequency / samplerate;
    const s = @sin(w0);
    const c = @cos(w0);
    const alpha = s / (2.0 * q_factor);

    const b0 = 1.0 - alpha;
    const b1 = -2.0 * c;
    const b2 = 1.0 + alpha;

    return makeBiquad(allocator, .{ b2, b1, b0 }, .{ b0, b1, b2 });
}

/// Creates a peak filter.
pub fn makePeak(
    allocator: std.mem.Allocator,
    frequency: f64,
    samplerate: f64,
    gain_db: f64,
    q_factor: f64,
) ButterworthError!iir.IIRFilter {
    try validateCommon(frequency, samplerate, q_factor);
    try validateGain(gain_db);
    const w0 = std.math.tau * frequency / samplerate;
    const s = @sin(w0);
    const c = @cos(w0);
    const alpha = s / (2.0 * q_factor);
    const big_a = std.math.pow(f64, 10.0, gain_db / 40.0);

    const b0 = 1.0 + alpha * big_a;
    const b1 = -2.0 * c;
    const b2 = 1.0 - alpha * big_a;
    const a0 = 1.0 + alpha / big_a;
    const a1 = -2.0 * c;
    const a2 = 1.0 - alpha / big_a;

    return makeBiquad(allocator, .{ a0, a1, a2 }, .{ b0, b1, b2 });
}

/// Creates a low-shelf filter.
pub fn makeLowshelf(
    allocator: std.mem.Allocator,
    frequency: f64,
    samplerate: f64,
    gain_db: f64,
    q_factor: f64,
) ButterworthError!iir.IIRFilter {
    try validateCommon(frequency, samplerate, q_factor);
    try validateGain(gain_db);
    const w0 = std.math.tau * frequency / samplerate;
    const s = @sin(w0);
    const c = @cos(w0);
    const alpha = s / (2.0 * q_factor);
    const big_a = std.math.pow(f64, 10.0, gain_db / 40.0);
    const pmc = (big_a + 1.0) - (big_a - 1.0) * c;
    const ppmc = (big_a + 1.0) + (big_a - 1.0) * c;
    const mpc = (big_a - 1.0) - (big_a + 1.0) * c;
    const pmpc = (big_a - 1.0) + (big_a + 1.0) * c;
    const aa2 = 2.0 * std.math.sqrt(big_a) * alpha;

    const b0 = big_a * (pmc + aa2);
    const b1 = 2.0 * big_a * mpc;
    const b2 = big_a * (pmc - aa2);
    const a0 = ppmc + aa2;
    const a1 = -2.0 * pmpc;
    const a2 = ppmc - aa2;

    return makeBiquad(allocator, .{ a0, a1, a2 }, .{ b0, b1, b2 });
}

/// Creates a high-shelf filter.
pub fn makeHighshelf(
    allocator: std.mem.Allocator,
    frequency: f64,
    samplerate: f64,
    gain_db: f64,
    q_factor: f64,
) ButterworthError!iir.IIRFilter {
    try validateCommon(frequency, samplerate, q_factor);
    try validateGain(gain_db);
    const w0 = std.math.tau * frequency / samplerate;
    const s = @sin(w0);
    const c = @cos(w0);
    const alpha = s / (2.0 * q_factor);
    const big_a = std.math.pow(f64, 10.0, gain_db / 40.0);
    const pmc = (big_a + 1.0) - (big_a - 1.0) * c;
    const ppmc = (big_a + 1.0) + (big_a - 1.0) * c;
    const mpc = (big_a - 1.0) - (big_a + 1.0) * c;
    const pmpc = (big_a - 1.0) + (big_a + 1.0) * c;
    const aa2 = 2.0 * std.math.sqrt(big_a) * alpha;

    const b0 = big_a * (ppmc + aa2);
    const b1 = -2.0 * big_a * pmpc;
    const b2 = big_a * (ppmc - aa2);
    const a0 = pmc + aa2;
    const a1 = 2.0 * mpc;
    const a2 = pmc - aa2;

    return makeBiquad(allocator, .{ a0, a1, a2 }, .{ b0, b1, b2 });
}

fn expectApproxEqualSlices(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "butterworth: lowpass coefficients match python" {
    var filt = try makeLowpass(testing.allocator, 1000.0, 48000.0, defaultQFactor());
    defer filt.deinit();
    try expectApproxEqualSlices(&[_]f64{ 1.0922959556412573, -1.9828897227476208, 0.9077040443587427 }, filt.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 0.004277569313094809, 0.008555138626189618, 0.004277569313094809 }, filt.b_coeffs, 1e-12);
}

test "butterworth: highpass coefficients match python" {
    var filt = try makeHighpass(testing.allocator, 1000.0, 48000.0, defaultQFactor());
    defer filt.deinit();
    try expectApproxEqualSlices(&[_]f64{ 1.0922959556412573, -1.9828897227476208, 0.9077040443587427 }, filt.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 0.9957224306869052, -1.9914448613738105, 0.9957224306869052 }, filt.b_coeffs, 1e-12);
}

test "butterworth: bandpass coefficients match python" {
    var filt = try makeBandpass(testing.allocator, 1000.0, 48000.0, defaultQFactor());
    defer filt.deinit();
    try expectApproxEqualSlices(&[_]f64{ 1.0922959556412573, -1.9828897227476208, 0.9077040443587427 }, filt.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 0.06526309611002579, 0.0, -0.06526309611002579 }, filt.b_coeffs, 1e-12);
}

test "butterworth: allpass coefficients match python" {
    var filt = try makeAllpass(testing.allocator, 1000.0, 48000.0, defaultQFactor());
    defer filt.deinit();
    try expectApproxEqualSlices(&[_]f64{ 1.0922959556412573, -1.9828897227476208, 0.9077040443587427 }, filt.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 0.9077040443587427, -1.9828897227476208, 1.0922959556412573 }, filt.b_coeffs, 1e-12);
}

test "butterworth: peak and shelf coefficients match python" {
    var peak = try makePeak(testing.allocator, 1000.0, 48000.0, 6.0, defaultQFactor());
    defer peak.deinit();
    try expectApproxEqualSlices(&[_]f64{ 1.0653405327119334, -1.9828897227476208, 0.9346594672880666 }, peak.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 1.1303715025601122, -1.9828897227476208, 0.8696284974398878 }, peak.b_coeffs, 1e-12);

    var low = try makeLowshelf(testing.allocator, 1000.0, 48000.0, 6.0, defaultQFactor());
    defer low.deinit();
    try expectApproxEqualSlices(&[_]f64{ 3.0409336710888786, -5.608870992220748, 2.602157875636628 }, low.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 3.139954022810743, -5.591841778072785, 2.5201667380627257 }, low.b_coeffs, 1e-12);

    var high = try makeHighshelf(testing.allocator, 1000.0, 48000.0, 6.0, defaultQFactor());
    defer high.deinit();
    try expectApproxEqualSlices(&[_]f64{ 2.2229172136088806, -3.9587208137297303, 1.7841414181566304 }, high.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 4.295432981120543, -7.922740859457287, 3.6756456963725253 }, high.b_coeffs, 1e-12);
}

test "butterworth: invalid parameters" {
    try testing.expectError(error.InvalidFrequency, makeLowpass(testing.allocator, 0.0, 48000.0, defaultQFactor()));
    try testing.expectError(error.InvalidSamplerate, makeHighpass(testing.allocator, 1000.0, 0.0, defaultQFactor()));
    try testing.expectError(error.InvalidQFactor, makeBandpass(testing.allocator, 1000.0, 48000.0, 0.0));
    try testing.expectError(error.InvalidGain, makePeak(testing.allocator, 1000.0, 48000.0, std.math.inf(f64), defaultQFactor()));
}

test "butterworth: extreme stable impulse response" {
    var filt = try makeLowpass(testing.allocator, 12000.0, 96000.0, defaultQFactor());
    defer filt.deinit();
    var sample = filt.process(1.0);
    try testing.expect(std.math.isFinite(sample));
    for (0..4096) |_| {
        sample = filt.process(0.0);
        try testing.expect(std.math.isFinite(sample));
    }
}
