//! Project Euler Problem 54: Poker Hands - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_054/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem054Error = error{
    InvalidCardCount,
    InvalidCard,
    InvalidCardValue,
    InvalidSuit,
    DuplicateCard,
    InvalidLine,
};

pub const CompareResult = enum {
    Win,
    Loss,
    Tie,
};

pub const HandCategory = enum(u8) {
    HighCard = 0,
    OnePair = 1,
    TwoPairs = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
    RoyalFlush = 9,
};

const Card = struct {
    value: u8,
    suit: u8,
};

const HandRank = struct {
    category: HandCategory,
    tiebreak: [5]u8,
};

pub const PokerHand = struct {
    cards: [5]Card,
    rank: HandRank,

    pub fn parse(hand: []const u8) Problem054Error!PokerHand {
        var tokens = std.mem.tokenizeScalar(u8, hand, ' ');
        var card_tokens: [5][]const u8 = undefined;
        var count: usize = 0;
        while (tokens.next()) |token| {
            if (count >= 5) return error.InvalidCardCount;
            card_tokens[count] = token;
            count += 1;
        }
        return fromTokens(card_tokens[0..count]);
    }

    pub fn compareWith(self: PokerHand, other: PokerHand) CompareResult {
        return compareRanks(self.rank, other.rank);
    }

    pub fn category(self: PokerHand) HandCategory {
        return self.rank.category;
    }

    fn fromTokens(tokens: []const []const u8) Problem054Error!PokerHand {
        if (tokens.len != 5) return error.InvalidCardCount;

        var cards: [5]Card = undefined;
        for (tokens, 0..) |token, idx| {
            cards[idx] = try parseCard(token);
        }

        for (0..cards.len) |i| {
            for (i + 1..cards.len) |j| {
                if (cards[i].value == cards[j].value and cards[i].suit == cards[j].suit) {
                    return error.DuplicateCard;
                }
            }
        }

        return .{
            .cards = cards,
            .rank = evaluate(cards),
        };
    }
};

fn parseCard(token: []const u8) Problem054Error!Card {
    if (token.len != 2) return error.InvalidCard;
    return .{
        .value = try parseValue(token[0]),
        .suit = try parseSuit(token[1]),
    };
}

fn parseValue(char: u8) Problem054Error!u8 {
    return switch (char) {
        '2'...'9' => char - '0',
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        else => error.InvalidCardValue,
    };
}

fn parseSuit(char: u8) Problem054Error!u8 {
    return switch (char) {
        'S', 'H', 'D', 'C' => char,
        else => error.InvalidSuit,
    };
}

fn sortDesc(values: *[5]u8) void {
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const current = values[i];
        var j = i;
        while (j > 0 and values[j - 1] < current) : (j -= 1) {
            values[j] = values[j - 1];
        }
        values[j] = current;
    }
}

fn straightHigh(values_desc: [5]u8) ?u8 {
    if (std.mem.eql(u8, &values_desc, &[_]u8{ 14, 5, 4, 3, 2 })) return 5;

    var idx: usize = 0;
    while (idx < 4) : (idx += 1) {
        if (values_desc[idx] != values_desc[idx + 1] + 1) return null;
    }
    return values_desc[0];
}

fn evaluate(cards: [5]Card) HandRank {
    var values: [5]u8 = undefined;
    var counts = [_]u8{0} ** 15;
    const first_suit = cards[0].suit;
    var flush = true;

    for (cards, 0..) |card, idx| {
        values[idx] = card.value;
        counts[card.value] += 1;
        if (card.suit != first_suit) flush = false;
    }
    sortDesc(&values);

    const straight = straightHigh(values);
    if (flush and straight != null) {
        if (straight.? == 14 and values[4] == 10) {
            return .{ .category = .RoyalFlush, .tiebreak = .{ 14, 13, 12, 11, 10 } };
        }
        return .{ .category = .StraightFlush, .tiebreak = .{ straight.?, 0, 0, 0, 0 } };
    }

    var pairs: [2]u8 = .{ 0, 0 };
    var pair_count: usize = 0;
    var kickers: [5]u8 = .{ 0, 0, 0, 0, 0 };
    var kicker_count: usize = 0;
    var trip: u8 = 0;
    var quad: u8 = 0;

    var value: usize = 15;
    while (value > 2) {
        value -= 1;
        switch (counts[value]) {
            4 => quad = @intCast(value),
            3 => trip = @intCast(value),
            2 => {
                pairs[pair_count] = @intCast(value);
                pair_count += 1;
            },
            1 => {
                kickers[kicker_count] = @intCast(value);
                kicker_count += 1;
            },
            else => {},
        }
    }

    if (quad != 0) {
        return .{ .category = .FourOfAKind, .tiebreak = .{ quad, kickers[0], 0, 0, 0 } };
    }
    if (trip != 0 and pair_count == 1) {
        return .{ .category = .FullHouse, .tiebreak = .{ trip, pairs[0], 0, 0, 0 } };
    }
    if (flush) {
        return .{ .category = .Flush, .tiebreak = values };
    }
    if (straight != null) {
        return .{ .category = .Straight, .tiebreak = .{ straight.?, 0, 0, 0, 0 } };
    }
    if (trip != 0) {
        return .{ .category = .ThreeOfAKind, .tiebreak = .{ trip, kickers[0], kickers[1], 0, 0 } };
    }
    if (pair_count == 2) {
        return .{ .category = .TwoPairs, .tiebreak = .{ pairs[0], pairs[1], kickers[0], 0, 0 } };
    }
    if (pair_count == 1) {
        return .{ .category = .OnePair, .tiebreak = .{ pairs[0], kickers[0], kickers[1], kickers[2], 0 } };
    }
    return .{ .category = .HighCard, .tiebreak = values };
}

fn compareRanks(left: HandRank, right: HandRank) CompareResult {
    const left_category = @intFromEnum(left.category);
    const right_category = @intFromEnum(right.category);
    if (left_category > right_category) return .Win;
    if (left_category < right_category) return .Loss;

    for (left.tiebreak, right.tiebreak) |l, r| {
        if (l > r) return .Win;
        if (l < r) return .Loss;
    }
    return .Tie;
}

/// Counts how many lines in the dataset are wins for player 1.
///
/// Time complexity: O(lines)
/// Space complexity: O(1)
pub fn countPlayer1Wins(data: []const u8) Problem054Error!u32 {
    var wins: u32 = 0;
    var lines = std.mem.tokenizeAny(u8, data, "\r\n");
    while (lines.next()) |line| {
        var tokens = std.mem.tokenizeScalar(u8, line, ' ');
        var card_tokens: [10][]const u8 = undefined;
        var count: usize = 0;
        while (tokens.next()) |token| {
            if (count >= 10) return error.InvalidLine;
            card_tokens[count] = token;
            count += 1;
        }
        if (count != 10) return error.InvalidLine;

        const player = try PokerHand.fromTokens(card_tokens[0..5]);
        const opponent = try PokerHand.fromTokens(card_tokens[5..10]);
        if (player.compareWith(opponent) == .Win) wins += 1;
    }
    return wins;
}

/// Returns the Project Euler answer for the bundled poker-hands dataset.
pub fn solution() Problem054Error!u32 {
    const poker_hands = @embedFile("problem_054_poker_hands.txt");
    return countPlayer1Wins(poker_hands);
}

test "problem 054: python reference dataset" {
    try testing.expectEqual(@as(u32, 376), try solution());
}

test "problem 054: compare semantics" {
    const player_1 = try PokerHand.parse("2H 3H 4H 5H 6H");
    const opp_1 = try PokerHand.parse("KS AS TS QS JS");
    try testing.expectEqual(CompareResult.Loss, player_1.compareWith(opp_1));

    const player_2 = try PokerHand.parse("2S AH 2H AS AC");
    const opp_2 = try PokerHand.parse("2H 3H 5H 6H 7H");
    try testing.expectEqual(CompareResult.Win, player_2.compareWith(opp_2));

    const player_3 = try PokerHand.parse("2S AH 4H 5S 6C");
    const opp_3 = try PokerHand.parse("AD 4C 5H 6H 2C");
    try testing.expectEqual(CompareResult.Tie, player_3.compareWith(opp_3));

    const player_4 = try PokerHand.parse("2H 4D 3C AS 5S");
    const opp_4 = try PokerHand.parse("2H 4D 3C 6S 5S");
    try testing.expectEqual(CompareResult.Loss, player_4.compareWith(opp_4));

    const player_5 = try PokerHand.parse("2H 3S 3C 3H 2S");
    const opp_5 = try PokerHand.parse("3S 3C 2S 2H 2D");
    try testing.expectEqual(CompareResult.Win, player_5.compareWith(opp_5));
}

test "problem 054: categories and edge cases" {
    const royal = try PokerHand.parse("KS AS TS QS JS");
    try testing.expectEqual(HandCategory.RoyalFlush, royal.category());

    const straight_flush = try PokerHand.parse("2D 6D 3D 4D 5D");
    try testing.expectEqual(HandCategory.StraightFlush, straight_flush.category());

    const full_house = try PokerHand.parse("3D 2H 3H 2C 2D");
    try testing.expectEqual(HandCategory.FullHouse, full_house.category());

    const low_ace = try PokerHand.parse("2C 4S AS 3D 5C");
    try testing.expectEqual(HandCategory.Straight, low_ace.category());
    try testing.expectEqual(@as(u8, 5), low_ace.rank.tiebreak[0]);

    try testing.expectError(error.InvalidCardCount, PokerHand.parse("AH 2C 3D"));
    try testing.expectError(error.InvalidCardValue, PokerHand.parse("1H 2C 3D 4S 5H"));
    try testing.expectError(error.InvalidSuit, PokerHand.parse("AH 2X 3D 4S 5H"));
    try testing.expectError(error.DuplicateCard, PokerHand.parse("AH AH 3D 4S 5H"));
    try testing.expectError(error.InvalidLine, countPlayer1Wins("AH KH QH JH TH 9C 9D 9S 9H 2C extra"));
}
