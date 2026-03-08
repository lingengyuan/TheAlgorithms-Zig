//! Hamming Code Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/hamming_code.py

const std = @import("std");
const testing = std.testing;

pub const HammingCodeError = error{
    EmptyData,
    InvalidBitCharacter,
    InvalidBitLength,
    SizeParityDontMatch,
};

pub const ReceptorResult = struct {
    data: []u8,
    ack: bool,
};

fn isPowerOfTwo(value: usize) bool {
    return value != 0 and (value & (value - 1)) == 0;
}

fn validateBitString(bits: []const u8) HammingCodeError!void {
    for (bits) |bit| {
        if (bit != '0' and bit != '1') {
            return HammingCodeError.InvalidBitCharacter;
        }
    }
}

/// Converts text bytes into a contiguous bitstring.
/// Caller owns returned buffer.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn textToBits(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, text.len * 8);
    errdefer allocator.free(out);

    for (text, 0..) |ch, i| {
        var bit: usize = 0;
        while (bit < 8) : (bit += 1) {
            const shift = 7 - bit;
            out[i * 8 + bit] = if (((ch >> @intCast(shift)) & 1) == 1) '1' else '0';
        }
    }

    return out;
}

/// Converts bitstring back to bytes using Python-reference behavior.
/// Caller owns returned buffer.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn textFromBits(allocator: std.mem.Allocator, bits: []const u8) ![]u8 {
    if (bits.len == 0) {
        return allocator.dupe(u8, "\x00");
    }
    try validateBitString(bits);

    if (bits.len % 8 != 0) {
        return HammingCodeError.InvalidBitLength;
    }

    const out = try allocator.alloc(u8, bits.len / 8);
    errdefer allocator.free(out);

    for (0..out.len) |i| {
        var value: u8 = 0;
        for (0..8) |j| {
            value <<= 1;
            if (bits[i * 8 + j] == '1') value |= 1;
        }
        out[i] = value;
    }

    // Python implementation returns "\0" for zero-value decoded data.
    var all_zero = true;
    for (out) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        allocator.free(out);
        return allocator.dupe(u8, "\x00");
    }

    return out;
}

/// Hamming emitter converter equivalent to Python `emitter_converter`.
/// Caller owns returned buffer.
///
/// Time complexity: O((size_par + n) * size_par)
/// Space complexity: O(size_par + n)
pub fn emitterConverter(allocator: std.mem.Allocator, size_par: usize, data: []const u8) ![]u8 {
    if (data.len == 0) {
        return HammingCodeError.EmptyData;
    }
    try validateBitString(data);

    // Keep the same validation semantics as the Python reference.
    const lhs: i128 = @intCast(size_par + data.len);
    const rhs: i128 = (@as(i128, 1) << @intCast(size_par)) - @as(i128, @intCast(data.len)) + 1;
    if (lhs <= rhs) {
        return HammingCodeError.SizeParityDontMatch;
    }

    const total_len = size_par + data.len;
    const data_ord = try allocator.alloc(?u8, total_len);
    defer allocator.free(data_ord);

    var qtd_bp: usize = 0;
    var cont_data: usize = 0;

    for (1..total_len + 1) |x| {
        const is_parity_slot = qtd_bp < size_par and isPowerOfTwo(x);
        if (is_parity_slot) {
            qtd_bp += 1;
            data_ord[x - 1] = null;
        } else {
            data_ord[x - 1] = data[cont_data];
            cont_data += 1;
        }
    }

    const parity = try allocator.alloc(u8, size_par);
    defer allocator.free(parity);

    for (1..size_par + 1) |bp| {
        var cont_bo: usize = 0;
        for (data_ord, 0..) |bit_opt, cont_loop| {
            if (bit_opt) |bit| {
                const pos = cont_loop + 1;
                const aux = (pos >> @intCast(bp - 1)) & 1;
                if (aux == 1 and bit == '1') {
                    cont_bo += 1;
                }
            }
        }
        parity[bp - 1] = if (cont_bo % 2 == 0) '0' else '1';
    }

    const data_out = try allocator.alloc(u8, total_len);
    errdefer allocator.free(data_out);

    var cont_bp: usize = 0;
    for (0..total_len) |i| {
        if (data_ord[i] == null) {
            data_out[i] = parity[cont_bp];
            cont_bp += 1;
        } else {
            data_out[i] = data_ord[i].?;
        }
    }

    return data_out;
}

/// Hamming receptor converter equivalent to Python `receptor_converter`.
/// Caller owns `result.data`.
///
/// Time complexity: O((size_par + n) * size_par)
/// Space complexity: O(size_par + n)
pub fn receptorConverter(allocator: std.mem.Allocator, size_par: usize, data: []const u8) !ReceptorResult {
    if (data.len == 0) {
        return HammingCodeError.EmptyData;
    }
    try validateBitString(data);

    var parity_received = std.ArrayListUnmanaged(u8){};
    defer parity_received.deinit(allocator);

    var data_output = std.ArrayListUnmanaged(u8){};
    defer data_output.deinit(allocator);

    var qtd_bp: usize = 0;
    for (data, 1..) |item, i| {
        const is_parity_slot = qtd_bp < size_par and isPowerOfTwo(i);
        if (is_parity_slot) {
            qtd_bp += 1;
            try parity_received.append(allocator, item);
        } else {
            try data_output.append(allocator, item);
        }
    }

    const total_len = size_par + data_output.items.len;
    const data_ord = try allocator.alloc(?u8, total_len);
    defer allocator.free(data_ord);

    qtd_bp = 0;
    var cont_data: usize = 0;

    for (1..total_len + 1) |x| {
        const is_parity_slot = qtd_bp < size_par and isPowerOfTwo(x);
        if (is_parity_slot) {
            qtd_bp += 1;
            data_ord[x - 1] = null;
        } else {
            data_ord[x - 1] = data_output.items[cont_data];
            cont_data += 1;
        }
    }

    const parity = try allocator.alloc(u8, size_par);
    defer allocator.free(parity);

    for (1..size_par + 1) |bp| {
        var cont_bo: usize = 0;
        for (data_ord, 0..) |bit_opt, cont_loop| {
            if (bit_opt) |bit| {
                const pos = cont_loop + 1;
                const aux = (pos >> @intCast(bp - 1)) & 1;
                if (aux == 1 and bit == '1') {
                    cont_bo += 1;
                }
            }
        }
        parity[bp - 1] = if (cont_bo % 2 == 0) '0' else '1';
    }

    const ack = std.mem.eql(u8, parity_received.items, parity);
    return .{
        .data = try data_output.toOwnedSlice(allocator),
        .ack = ack,
    };
}

test "hamming code: python conversion and doctest vectors" {
    const alloc = testing.allocator;

    const bits = try textToBits(alloc, "msg");
    defer alloc.free(bits);
    try testing.expectEqualStrings("011011010111001101100111", bits);

    const text = try textFromBits(alloc, "011011010111001101100111");
    defer alloc.free(text);
    try testing.expectEqualStrings("msg", text);

    const encoded = try emitterConverter(alloc, 4, "101010111111");
    defer alloc.free(encoded);
    try testing.expectEqualStrings("1111010010111111", encoded);

    try testing.expectError(HammingCodeError.SizeParityDontMatch, emitterConverter(alloc, 5, "101010111111"));

    const decoded = try receptorConverter(alloc, 4, "1111010010111111");
    defer alloc.free(decoded.data);
    try testing.expectEqualStrings("101010111111", decoded.data);
    try testing.expectEqual(true, decoded.ack);
}

test "hamming code: additional parity cases and extreme behavior" {
    const alloc = testing.allocator;

    const encoded = try emitterConverter(alloc, 3, "1010");
    defer alloc.free(encoded);
    try testing.expectEqualStrings("1011010", encoded);

    const decoded = try receptorConverter(alloc, 3, encoded);
    defer alloc.free(decoded.data);
    try testing.expectEqualStrings("1010", decoded.data);
    try testing.expectEqual(true, decoded.ack);

    var corrupted = try alloc.dupe(u8, "1111010010111111");
    defer alloc.free(corrupted);
    corrupted[corrupted.len - 2] = if (corrupted[corrupted.len - 2] == '0') '1' else '0';

    const decoded_corrupted = try receptorConverter(alloc, 4, corrupted);
    defer alloc.free(decoded_corrupted.data);
    try testing.expectEqualStrings("101010111101", decoded_corrupted.data);
    try testing.expectEqual(false, decoded_corrupted.ack);

    const large_bits = "10100111" ** 10_000;
    const large_encoded = try emitterConverter(alloc, 4, large_bits);
    defer alloc.free(large_encoded);

    const large_decoded = try receptorConverter(alloc, 4, large_encoded);
    defer alloc.free(large_decoded.data);
    try testing.expectEqualStrings(large_bits, large_decoded.data);
    try testing.expectEqual(true, large_decoded.ack);

    try testing.expectError(HammingCodeError.InvalidBitCharacter, emitterConverter(alloc, 4, "10a1"));
    try testing.expectError(HammingCodeError.InvalidBitCharacter, receptorConverter(alloc, 4, "10b1"));
    try testing.expectError(HammingCodeError.InvalidBitLength, textFromBits(alloc, "101"));
}
