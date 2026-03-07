//! Peak Signal-to-Noise Ratio - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/peak_signal_to_noise_ratio.py

const std = @import("std");
const testing = std.testing;

pub const PsnrError = error{
    EmptyInput,
    LengthMismatch,
};

const pixel_max: f64 = 255.0;

/// Computes peak signal-to-noise ratio (PSNR) in dB between two equal-length signals.
/// Returns 100 when MSE is zero, matching Python reference behavior.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn peakSignalToNoiseRatio(original: []const f64, contrast: []const f64) PsnrError!f64 {
    if (original.len == 0) {
        return PsnrError.EmptyInput;
    }
    if (original.len != contrast.len) {
        return PsnrError.LengthMismatch;
    }

    var mse: f64 = 0.0;
    for (original, contrast) |a, b| {
        const d = a - b;
        mse += d * d;
    }
    mse /= @as(f64, @floatFromInt(original.len));

    if (mse == 0.0) {
        return 100.0;
    }

    return 20.0 * std.math.log10(pixel_max / @sqrt(mse));
}

test "psnr: basic and python-like behavior" {
    const a = [_]f64{ 10, 20, 30, 40 };
    const b = [_]f64{ 10, 20, 30, 40 };
    try testing.expectApproxEqAbs(@as(f64, 100.0), try peakSignalToNoiseRatio(&a, &b), 1e-12);

    const c = [_]f64{ 0, 0, 0, 0 };
    const d = [_]f64{ 0, 0, 0, 10 };
    try testing.expectApproxEqAbs(@as(f64, 34.15140352195873), try peakSignalToNoiseRatio(&c, &d), 1e-12);
}

test "psnr: validation and extreme values" {
    try testing.expectError(PsnrError.EmptyInput, peakSignalToNoiseRatio(&[_]f64{}, &[_]f64{}));
    try testing.expectError(PsnrError.LengthMismatch, peakSignalToNoiseRatio(&[_]f64{ 1, 2 }, &[_]f64{1}));

    const large_a = try testing.allocator.alloc(f64, 100_000);
    defer testing.allocator.free(large_a);
    const large_b = try testing.allocator.alloc(f64, 100_000);
    defer testing.allocator.free(large_b);
    @memset(large_a, 255.0);
    @memset(large_b, 254.0);

    const psnr = try peakSignalToNoiseRatio(large_a, large_b);
    try testing.expect(std.math.isFinite(psnr));
    try testing.expect(psnr > 40.0);
}
