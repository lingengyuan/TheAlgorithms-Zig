//! Bloom Filter - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/hashing/bloom_filter.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const BloomError = error{InvalidSize};

pub const Bloom = struct {
    allocator: Allocator,
    size: usize,
    bits: []u64,

    pub fn init(allocator: Allocator, size: usize) !Bloom {
        if (size == 0) return BloomError.InvalidSize;
        const words = (size + 63) / 64;
        const bits = try allocator.alloc(u64, words);
        @memset(bits, 0);
        return .{
            .allocator = allocator,
            .size = size,
            .bits = bits,
        };
    }

    pub fn deinit(self: *Bloom) void {
        self.allocator.free(self.bits);
        self.* = undefined;
    }

    fn setBit(self: *Bloom, position: usize) void {
        const word = position / 64;
        const bit = position % 64;
        self.bits[word] |= (@as(u64, 1) << @as(u6, @intCast(bit)));
    }

    fn getBit(self: *const Bloom, position: usize) bool {
        const word = position / 64;
        const bit = position % 64;
        return (self.bits[word] & (@as(u64, 1) << @as(u6, @intCast(bit)))) != 0;
    }

    fn countOnes(self: *const Bloom) usize {
        var sum: usize = 0;
        for (self.bits) |word| {
            sum += @popCount(word);
        }
        return sum;
    }

    fn digestLeMod(digest: []const u8, modulus: usize) usize {
        if (modulus == 0) return 0;

        var factor: usize = 1 % modulus;
        var rem: usize = 0;
        for (digest) |b| {
            rem = (rem + (factor * @as(usize, b)) % modulus) % modulus;
            factor = (factor * 256) % modulus;
        }
        return rem;
    }

    fn hashPositions(self: *const Bloom, value: []const u8) [2]usize {
        var sha_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(value, &sha_digest, .{});

        var md5_digest: [16]u8 = undefined;
        std.crypto.hash.Md5.hash(value, &md5_digest, .{});

        return .{
            digestLeMod(&sha_digest, self.size),
            digestLeMod(&md5_digest, self.size),
        };
    }

    /// Adds value to Bloom filter.
    /// Time complexity: O(k), Space complexity: O(1), with k=2 hash functions.
    pub fn add(self: *Bloom, value: []const u8) void {
        const pos = self.hashPositions(value);
        self.setBit(pos[0]);
        self.setBit(pos[1]);
    }

    /// Checks if value may exist in Bloom filter.
    /// Time complexity: O(k), Space complexity: O(1)
    pub fn exists(self: *const Bloom, value: []const u8) bool {
        const pos = self.hashPositions(value);
        return self.getBit(pos[0]) and self.getBit(pos[1]);
    }

    pub fn contains(self: *const Bloom, value: []const u8) bool {
        return self.exists(value);
    }

    fn formatBin(self: *const Bloom, allocator: Allocator, positions_only: ?[2]usize) ![]u8 {
        const out = try allocator.alloc(u8, self.size);
        errdefer allocator.free(out);

        for (0..self.size) |i| {
            const pos = self.size - 1 - i; // highest bit first, like Python bin().zfill()
            const set = if (positions_only) |positions|
                (pos == positions[0] or pos == positions[1])
            else
                self.getBit(pos);
            out[i] = if (set) '1' else '0';
        }
        return out;
    }

    /// Returns filter bitstring in Python-compatible orientation.
    pub fn bitString(self: *const Bloom, allocator: Allocator) ![]u8 {
        return self.formatBin(allocator, null);
    }

    /// Returns hash bitstring for provided value.
    pub fn formatHash(self: *const Bloom, allocator: Allocator, value: []const u8) ![]u8 {
        return self.formatBin(allocator, self.hashPositions(value));
    }

    /// Estimated false-positive error rate.
    pub fn estimatedErrorRate(self: *const Bloom) f64 {
        const n_ones = @as(f64, @floatFromInt(self.countOnes()));
        const size_f = @as(f64, @floatFromInt(self.size));
        return std.math.pow(f64, n_ones / size_f, 2);
    }
};

test "bloom filter: python core sample" {
    var bloom = try Bloom.init(testing.allocator, 8);
    defer bloom.deinit();

    const empty_bits = try bloom.bitString(testing.allocator);
    defer testing.allocator.free(empty_bits);
    try testing.expectEqualStrings("00000000", empty_bits);

    try testing.expect(!bloom.contains("Titanic"));
    bloom.add("Titanic");

    const after_titanic = try bloom.bitString(testing.allocator);
    defer testing.allocator.free(after_titanic);
    try testing.expectEqualStrings("01100000", after_titanic);
    try testing.expect(bloom.contains("Titanic"));

    bloom.add("Avatar");
    try testing.expect(bloom.contains("Avatar"));

    const avatar_hash = try bloom.formatHash(testing.allocator, "Avatar");
    defer testing.allocator.free(avatar_hash);
    try testing.expectEqualStrings("00000100", avatar_hash);

    const after_avatar = try bloom.bitString(testing.allocator);
    defer testing.allocator.free(after_avatar);
    try testing.expectEqualStrings("01100100", after_avatar);
}

test "bloom filter: python negatives and false positive sample" {
    var bloom = try Bloom.init(testing.allocator, 8);
    defer bloom.deinit();

    bloom.add("Titanic");
    bloom.add("Avatar");

    try testing.expect(!bloom.contains("The Godfather"));
    try testing.expect(!bloom.contains("Interstellar"));
    try testing.expect(!bloom.contains("Parasite"));
    try testing.expect(!bloom.contains("Pulp Fiction"));

    // Documented false positive in reference.
    try testing.expect(bloom.contains("Ratatouille"));

    try testing.expectApproxEqRel(@as(f64, 0.140625), bloom.estimatedErrorRate(), 1e-12);

    bloom.add("The Godfather");
    try testing.expectApproxEqRel(@as(f64, 0.25), bloom.estimatedErrorRate(), 1e-12);
}

test "bloom filter: invalid and extreme" {
    try testing.expectError(BloomError.InvalidSize, Bloom.init(testing.allocator, 0));

    var bloom = try Bloom.init(testing.allocator, 4096);
    defer bloom.deinit();

    for (0..20_000) |i| {
        var buf: [32]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "item-{d}", .{i});
        bloom.add(s);
    }

    for (0..512) |i| {
        var buf: [32]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "item-{d}", .{i});
        try testing.expect(bloom.exists(s));
    }

    const bits = try bloom.bitString(testing.allocator);
    defer testing.allocator.free(bits);
    try testing.expectEqual(@as(usize, 4096), bits.len);
}
