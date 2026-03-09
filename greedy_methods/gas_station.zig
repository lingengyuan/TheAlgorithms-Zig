//! Gas Station - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/greedy_methods/gas_station.py

const std = @import("std");
const testing = std.testing;

pub const GasStation = struct {
    gas_quantity: i64,
    cost: i64,
};

pub const GasStationError = error{LengthMismatch};

/// Builds the gas station list from quantity and cost arrays.
/// Caller owns the returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn getGasStations(
    allocator: std.mem.Allocator,
    gas_quantities: []const i64,
    costs: []const i64,
) (GasStationError || std.mem.Allocator.Error)![]GasStation {
    if (gas_quantities.len != costs.len) return error.LengthMismatch;

    const stations = try allocator.alloc(GasStation, gas_quantities.len);
    for (gas_quantities, costs, 0..) |quantity, cost, i| {
        stations[i] = .{
            .gas_quantity = quantity,
            .cost = cost,
        };
    }
    return stations;
}

/// Returns the starting index needed to complete the circuit, or -1 if impossible.
/// Time complexity: O(n), Space complexity: O(1)
pub fn canCompleteJourney(gas_stations: []const GasStation) isize {
    var total_gas: i64 = 0;
    var total_cost: i64 = 0;
    for (gas_stations) |station| {
        total_gas += station.gas_quantity;
        total_cost += station.cost;
    }
    if (total_gas < total_cost) return -1;

    var start: isize = 0;
    var net: i64 = 0;
    for (gas_stations, 0..) |station, i| {
        net += station.gas_quantity - station.cost;
        if (net < 0) {
            start = @intCast(i + 1);
            net = 0;
        }
    }
    return start;
}

test "gas station: python reference examples" {
    const alloc = testing.allocator;

    const stations1 = try getGasStations(
        alloc,
        &[_]i64{ 1, 2, 3, 4, 5 },
        &[_]i64{ 3, 4, 5, 1, 2 },
    );
    defer alloc.free(stations1);
    try testing.expectEqual(@as(isize, 3), canCompleteJourney(stations1));

    const stations2 = try getGasStations(
        alloc,
        &[_]i64{ 2, 3, 4 },
        &[_]i64{ 3, 4, 3 },
    );
    defer alloc.free(stations2);
    try testing.expectEqual(@as(isize, -1), canCompleteJourney(stations2));
}

test "gas station: edge and extreme cases" {
    const alloc = testing.allocator;

    try testing.expectError(
        GasStationError.LengthMismatch,
        getGasStations(alloc, &[_]i64{ 1, 2 }, &[_]i64{1}),
    );

    const empty = try getGasStations(alloc, &[_]i64{}, &[_]i64{});
    defer alloc.free(empty);
    try testing.expectEqual(@as(isize, 0), canCompleteJourney(empty));

    const uniform = try getGasStations(
        alloc,
        &[_]i64{ 5, 5, 5, 5 },
        &[_]i64{ 5, 5, 5, 5 },
    );
    defer alloc.free(uniform);
    try testing.expectEqual(@as(isize, 0), canCompleteJourney(uniform));
}
