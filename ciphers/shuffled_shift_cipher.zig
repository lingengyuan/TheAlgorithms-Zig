//! Shuffled Shift Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/shuffled_shift_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const ShuffledShiftError = error{ InvalidCharacter, EmptyPasscode };

const KEY_LIST_OPTIONS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n";
const PASSCODE_CHOICES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

pub const ShuffledShiftCipher = struct {
    passcode: []u8,
    key_list: []u8,
    shift_key: i64,

    pub fn deinit(self: ShuffledShiftCipher, allocator: Allocator) void {
        allocator.free(self.passcode);
        allocator.free(self.key_list);
    }
};

fn passcodeCreator(allocator: Allocator, random: std.Random) ![]u8 {
    const len = random.intRangeAtMost(usize, 10, 20);
    const out = try allocator.alloc(u8, len);
    errdefer allocator.free(out);

    for (out) |*ch| {
        const idx = random.intRangeLessThan(usize, 0, PASSCODE_CHOICES.len);
        ch.* = PASSCODE_CHOICES[idx];
    }

    return out;
}

fn makeKeyList(allocator: Allocator, passcode: []const u8) ![]u8 {
    var breakpoints = std.ArrayListUnmanaged(u8){};
    defer breakpoints.deinit(allocator);

    for (passcode) |ch| {
        if (std.mem.indexOfScalar(u8, breakpoints.items, ch) == null) {
            try breakpoints.append(allocator, ch);
        }
    }
    std.mem.sort(u8, breakpoints.items, {}, comptime std.sort.asc(u8));

    var keys = std.ArrayListUnmanaged(u8){};
    errdefer keys.deinit(allocator);

    var temp = std.ArrayListUnmanaged(u8){};
    defer temp.deinit(allocator);

    for (KEY_LIST_OPTIONS, 0..) |ch, i| {
        try temp.append(allocator, ch);
        const is_break = std.mem.indexOfScalar(u8, breakpoints.items, ch) != null;
        const is_last = i == KEY_LIST_OPTIONS.len - 1;

        if (is_break or is_last) {
            var j = temp.items.len;
            while (j > 0) {
                j -= 1;
                try keys.append(allocator, temp.items[j]);
            }
            temp.clearRetainingCapacity();
        }
    }

    return try keys.toOwnedSlice(allocator);
}

fn makeShiftKey(passcode: []const u8) i64 {
    var sum: i64 = 0;
    for (passcode, 0..) |ch, i| {
        const val: i64 = @intCast(ch);
        sum += if ((i & 1) == 0) val else -val;
    }
    return if (sum > 0) sum else @intCast(passcode.len);
}

pub fn init(allocator: Allocator, random_opt: ?std.Random, passcode_opt: ?[]const u8) !ShuffledShiftCipher {
    var passcode: []u8 = undefined;
    if (passcode_opt) |pc| {
        if (pc.len == 0) return ShuffledShiftError.EmptyPasscode;
        passcode = try allocator.dupe(u8, pc);
    } else {
        const random = random_opt orelse return ShuffledShiftError.EmptyPasscode;
        passcode = try passcodeCreator(allocator, random);
    }

    const key_list = try makeKeyList(allocator, passcode);
    const shift_key = makeShiftKey(passcode);

    return ShuffledShiftCipher{ .passcode = passcode, .key_list = key_list, .shift_key = shift_key };
}

/// Encrypts plaintext with shuffled-shift cipher.
/// Time complexity: O(n * m), Space complexity: O(n)
pub fn encrypt(allocator: Allocator, cipher: ShuffledShiftCipher, plaintext: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, plaintext.len);
    errdefer allocator.free(out);

    const len_i64: i64 = @intCast(cipher.key_list.len);

    for (plaintext, 0..) |ch, i| {
        const pos = std.mem.indexOfScalar(u8, cipher.key_list, ch) orelse return ShuffledShiftError.InvalidCharacter;
        const encoded_pos = @mod(@as(i64, @intCast(pos)) + cipher.shift_key, len_i64);
        out[i] = cipher.key_list[@intCast(encoded_pos)];
    }

    return out;
}

/// Decrypts shuffled-shift ciphertext.
/// Time complexity: O(n * m), Space complexity: O(n)
pub fn decrypt(allocator: Allocator, cipher: ShuffledShiftCipher, encoded_message: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, encoded_message.len);
    errdefer allocator.free(out);

    const len_i64: i64 = @intCast(cipher.key_list.len);

    for (encoded_message, 0..) |ch, i| {
        const pos = std.mem.indexOfScalar(u8, cipher.key_list, ch) orelse return ShuffledShiftError.InvalidCharacter;
        const decoded_pos = @mod(@as(i64, @intCast(pos)) - cipher.shift_key, len_i64);
        out[i] = cipher.key_list[@intCast(decoded_pos)];
    }

    return out;
}

test "shuffled shift: python sample" {
    const alloc = testing.allocator;

    const cipher = try init(alloc, null, "4PYIXyqeQZr44");
    defer cipher.deinit(alloc);

    const enc = try encrypt(alloc, cipher, "Hello, this is a modified Caesar cipher");
    defer alloc.free(enc);
    try testing.expectEqualStrings("d>**-1z6&'5z'5z:z+-='$'>=zp:>5:#z<'.&>#", enc);

    const dec = try decrypt(alloc, cipher, "d>**-1z6&'5z'5z:z+-='$'>=zp:>5:#z<'.&>#");
    defer alloc.free(dec);
    try testing.expectEqualStrings("Hello, this is a modified Caesar cipher", dec);
}

test "shuffled shift: random end-to-end" {
    const alloc = testing.allocator;

    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    const cipher = try init(alloc, rng, null);
    defer cipher.deinit(alloc);

    const msg = "Hello, this is a modified Caesar cipher";
    const enc = try encrypt(alloc, cipher, msg);
    defer alloc.free(enc);
    const out = try decrypt(alloc, cipher, enc);
    defer alloc.free(out);

    try testing.expectEqualStrings(msg, out);
}

test "shuffled shift: invalid input and extreme" {
    const alloc = testing.allocator;

    try testing.expectError(ShuffledShiftError.EmptyPasscode, init(alloc, null, ""));

    const cipher = try init(alloc, null, "A1B2C3D4");
    defer cipher.deinit(alloc);

    const n: usize = 10000;
    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);
    for (msg, 0..) |*ch, i| ch.* = KEY_LIST_OPTIONS[i % KEY_LIST_OPTIONS.len];

    const enc = try encrypt(alloc, cipher, msg);
    defer alloc.free(enc);
    const dec = try decrypt(alloc, cipher, enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, msg, dec);
}
