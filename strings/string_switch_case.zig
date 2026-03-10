//! String Switch Case - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/string_switch_case.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

fn extractWords(allocator: Allocator, text: []const u8) ![]const []u8 {
    var words = std.ArrayListUnmanaged([]u8){};
    errdefer {
        for (words.items) |word| allocator.free(word);
        words.deinit(allocator);
    }

    var current = std.ArrayListUnmanaged(u8){};
    defer current.deinit(allocator);
    for (text) |char| {
        if (std.ascii.isAlphanumeric(char)) {
            try current.append(allocator, char);
        } else {
            if (current.items.len > 0) {
                try words.append(allocator, try current.toOwnedSlice(allocator));
                current = .{};
            }
        }
    }
    if (current.items.len > 0) try words.append(allocator, try current.toOwnedSlice(allocator));
    return words.toOwnedSlice(allocator);
}

fn freeWords(allocator: Allocator, words: []const []u8) void {
    for (words) |word| allocator.free(word);
    allocator.free(words);
}

/// Caller owns returned slice.
pub fn toPascalCase(allocator: Allocator, text: []const u8) ![]u8 {
    const words = try extractWords(allocator, text);
    defer freeWords(allocator, words);
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);
    for (words) |word| {
        if (word.len == 0) continue;
        try out.append(allocator, std.ascii.toUpper(word[0]));
        for (word[1..]) |char| try out.append(allocator, std.ascii.toLower(char));
    }
    return out.toOwnedSlice(allocator);
}

pub fn toCamelCase(allocator: Allocator, text: []const u8) ![]u8 {
    const pascal = try toPascalCase(allocator, text);
    if (pascal.len == 0) {
        allocator.free(pascal);
        return allocator.dupe(u8, "not valid string");
    }
    pascal[0] = std.ascii.toLower(pascal[0]);
    return pascal;
}

fn toSeparatedCase(allocator: Allocator, text: []const u8, upper: bool, separator: u8) ![]u8 {
    const words = try extractWords(allocator, text);
    defer freeWords(allocator, words);
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);
    for (words, 0..) |word, index| {
        if (index > 0) try out.append(allocator, separator);
        for (word) |char| {
            try out.append(allocator, if (upper) std.ascii.toUpper(char) else std.ascii.toLower(char));
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn toSnakeCase(allocator: Allocator, text: []const u8, upper: bool) ![]u8 {
    return toSeparatedCase(allocator, text, upper, '_');
}

pub fn toKebabCase(allocator: Allocator, text: []const u8, upper: bool) ![]u8 {
    return toSeparatedCase(allocator, text, upper, '-');
}

test "string switch case: python samples" {
    const pascal = try toPascalCase(testing.allocator, "one two 31235three4four");
    defer testing.allocator.free(pascal);
    try testing.expectEqualStrings("OneTwo31235three4four", pascal);

    const camel = try toCamelCase(testing.allocator, "one two 31235three4four");
    defer testing.allocator.free(camel);
    try testing.expectEqualStrings("oneTwo31235three4four", camel);

    const snake_upper = try toSnakeCase(testing.allocator, "one two 31235three4four", true);
    defer testing.allocator.free(snake_upper);
    try testing.expectEqualStrings("ONE_TWO_31235THREE4FOUR", snake_upper);

    const kebab_lower = try toKebabCase(testing.allocator, "one two 31235three4four", false);
    defer testing.allocator.free(kebab_lower);
    try testing.expectEqualStrings("one-two-31235three4four", kebab_lower);
}

test "string switch case: punctuation and invalid" {
    const simple = try toPascalCase(testing.allocator, "special characters :, ', %, ^, $, are ignored");
    defer testing.allocator.free(simple);
    try testing.expectEqualStrings("SpecialCharactersAreIgnored", simple);

    const invalid = try toCamelCase(testing.allocator, "");
    defer testing.allocator.free(invalid);
    try testing.expectEqualStrings("not valid string", invalid);
}
