//! Best Time to Buy and Sell Stock - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/greedy_methods/best_time_to_buy_and_sell_stock.py

const std = @import("std");
const testing = std.testing;

/// Returns the maximum profit achievable from a single buy + sell transaction.
/// Returns 0 if no profit is possible.
/// Time complexity: O(n), Space complexity: O(1)
pub fn maxProfit(prices: []const i64) i64 {
    if (prices.len == 0) return 0;
    var min_price: i64 = prices[0];
    var profit: i64 = 0;
    for (prices[1..]) |p| {
        if (p < min_price) min_price = p;
        const gain = p - min_price;
        if (gain > profit) profit = gain;
    }
    return profit;
}

test "buy sell: basic profit" {
    try testing.expectEqual(@as(i64, 5), maxProfit(&[_]i64{ 7, 1, 5, 3, 6, 4 }));
}

test "buy sell: no profit" {
    try testing.expectEqual(@as(i64, 0), maxProfit(&[_]i64{ 7, 6, 4, 3, 1 }));
}

test "buy sell: single price" {
    try testing.expectEqual(@as(i64, 0), maxProfit(&[_]i64{5}));
}

test "buy sell: empty" {
    try testing.expectEqual(@as(i64, 0), maxProfit(&[_]i64{}));
}

test "buy sell: all same" {
    try testing.expectEqual(@as(i64, 0), maxProfit(&[_]i64{ 3, 3, 3, 3 }));
}
