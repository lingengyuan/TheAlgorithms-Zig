//! Tabu Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/tabu_search.py

const std = @import("std");
const testing = std.testing;

pub const TabuSearchError = std.mem.Allocator.Error || error{
    InvalidLine,
    InvalidNodeLabel,
    InvalidDistance,
    DuplicateEdge,
    EmptyGraph,
    MissingEdge,
    EmptyNeighborhood,
};

const inf_cost = std.math.maxInt(i32);

pub const Tour = struct {
    path: []u8,
    cost: i32,

    pub fn deinit(self: *Tour, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
    }
};

pub const TourCandidate = struct {
    path: []u8,
    cost: i32,

    pub fn deinit(self: *TourCandidate, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
    }
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: []u8,
    costs: []i32,
    start_node: u8,

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.nodes);
        self.allocator.free(self.costs);
    }

    fn indexOf(self: Graph, node: u8) ?usize {
        for (self.nodes, 0..) |candidate, index| {
            if (candidate == node) return index;
        }
        return null;
    }

    pub fn costBetween(self: Graph, from: u8, to: u8) ?i32 {
        const from_index = self.indexOf(from) orelse return null;
        const to_index = self.indexOf(to) orelse return null;
        const cost = self.costs[from_index * self.nodes.len + to_index];
        if (cost == inf_cost) return null;
        return cost;
    }
};

fn containsNode(nodes: []const u8, node: u8) bool {
    return std.mem.indexOfScalar(u8, nodes, node) != null;
}

fn routeCost(route: []const u8, graph: Graph) TabuSearchError!i32 {
    var total: i32 = 0;
    for (route[0 .. route.len - 1], 0..) |from, index| {
        const to = route[index + 1];
        const edge_cost = graph.costBetween(from, to) orelse return error.MissingEdge;
        total += edge_cost;
    }
    return total;
}

/// Parses a symmetric TSP graph from whitespace-separated lines: `A B 12`.
/// Time complexity: O(E * V), Space complexity: O(V^2)
pub fn parseGraph(text: []const u8, allocator: std.mem.Allocator) TabuSearchError!Graph {
    var nodes = std.ArrayListUnmanaged(u8){};
    defer nodes.deinit(allocator);
    var edges = std.ArrayListUnmanaged(struct { from: u8, to: u8, cost: i32 }){};
    defer edges.deinit(allocator);

    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var start_node: ?u8 = null;
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;

        var parts = std.mem.tokenizeAny(u8, line, " \t\r");
        const from_token = parts.next() orelse return error.InvalidLine;
        const to_token = parts.next() orelse return error.InvalidLine;
        const distance_token = parts.next() orelse return error.InvalidLine;
        if (parts.next() != null) return error.InvalidLine;
        if (from_token.len != 1 or to_token.len != 1) return error.InvalidNodeLabel;

        const from = from_token[0];
        const to = to_token[0];
        const cost = std.fmt.parseInt(i32, distance_token, 10) catch return error.InvalidDistance;

        if (start_node == null) start_node = from;
        if (!containsNode(nodes.items, from)) try nodes.append(allocator, from);
        if (!containsNode(nodes.items, to)) try nodes.append(allocator, to);
        try edges.append(allocator, .{ .from = from, .to = to, .cost = cost });
    }

    if (nodes.items.len == 0 or start_node == null) return error.EmptyGraph;

    const graph_nodes = try allocator.dupe(u8, nodes.items);
    errdefer allocator.free(graph_nodes);
    const costs = try allocator.alloc(i32, graph_nodes.len * graph_nodes.len);
    errdefer allocator.free(costs);
    @memset(costs, inf_cost);
    for (0..graph_nodes.len) |index| {
        costs[index * graph_nodes.len + index] = 0;
    }

    for (edges.items) |edge| {
        const from_index = std.mem.indexOfScalar(u8, graph_nodes, edge.from).?;
        const to_index = std.mem.indexOfScalar(u8, graph_nodes, edge.to).?;
        const forward_index = from_index * graph_nodes.len + to_index;
        const reverse_index = to_index * graph_nodes.len + from_index;
        if (costs[forward_index] != inf_cost or costs[reverse_index] != inf_cost) return error.DuplicateEdge;
        costs[forward_index] = edge.cost;
        costs[reverse_index] = edge.cost;
    }

    return .{
        .allocator = allocator,
        .nodes = graph_nodes,
        .costs = costs,
        .start_node = start_node.?,
    };
}

/// Builds the greedy initial tour using the same nearest-neighbor idea as Python.
/// Time complexity: O(V^2), Space complexity: O(V)
pub fn generateFirstSolution(graph: Graph, allocator: std.mem.Allocator) TabuSearchError!Tour {
    const node_count = graph.nodes.len;
    var path = try allocator.alloc(u8, node_count + 1);
    errdefer allocator.free(path);

    var visited = try allocator.alloc(bool, node_count);
    defer allocator.free(visited);
    @memset(visited, false);

    var current = graph.start_node;
    const start_index = graph.indexOf(current).?;
    visited[start_index] = true;
    path[0] = current;
    var path_len: usize = 1;
    var visited_count: usize = 1;
    var total_cost: i32 = 0;

    while (visited_count < node_count) {
        var best_index: ?usize = null;
        var best_cost: i32 = inf_cost;
        for (graph.nodes, 0..) |candidate, candidate_index| {
            if (visited[candidate_index]) continue;
            const candidate_cost = graph.costBetween(current, candidate) orelse continue;
            if (candidate_cost < best_cost) {
                best_cost = candidate_cost;
                best_index = candidate_index;
            }
        }
        const next_index = best_index orelse return error.MissingEdge;
        const next_node = graph.nodes[next_index];
        total_cost += best_cost;
        path[path_len] = next_node;
        path_len += 1;
        visited[next_index] = true;
        visited_count += 1;
        current = next_node;
    }

    total_cost += graph.costBetween(current, graph.start_node) orelse return error.MissingEdge;
    path[path_len] = graph.start_node;

    return .{ .path = path, .cost = total_cost };
}

fn swapRoute(route: []const u8, i: usize, j: usize, allocator: std.mem.Allocator) ![]u8 {
    const copy = try allocator.dupe(u8, route);
    std.mem.swap(u8, &copy[i], &copy[j]);
    return copy;
}

/// Generates the 1-1 exchange neighborhood sorted by cost ascending.
/// Time complexity: O(V^3), Space complexity: O(V^3)
pub fn findNeighborhood(route: []const u8, graph: Graph, allocator: std.mem.Allocator) TabuSearchError![]TourCandidate {
    var candidates = std.ArrayListUnmanaged(TourCandidate){};
    errdefer {
        for (candidates.items) |*candidate| candidate.deinit(allocator);
        candidates.deinit(allocator);
    }

    var i: usize = 1;
    while (i + 1 < route.len) : (i += 1) {
        var j: usize = i + 1;
        while (j + 1 < route.len) : (j += 1) {
            const swapped = try swapRoute(route, i, j, allocator);
            const cost = try routeCost(swapped, graph);
            try candidates.append(allocator, .{ .path = swapped, .cost = cost });
        }
    }

    if (candidates.items.len == 0) return error.EmptyNeighborhood;

    var sort_index: usize = 1;
    while (sort_index < candidates.items.len) : (sort_index += 1) {
        var j = sort_index;
        while (j > 0 and candidates.items[j].cost < candidates.items[j - 1].cost) : (j -= 1) {
            std.mem.swap(TourCandidate, &candidates.items[j], &candidates.items[j - 1]);
        }
    }

    return try candidates.toOwnedSlice(allocator);
}

const TabuMove = struct {
    first: u8,
    second: u8,
};

fn isTabu(moves: []const TabuMove, first: u8, second: u8) bool {
    for (moves) |move| {
        if ((move.first == first and move.second == second) or (move.first == second and move.second == first)) {
            return true;
        }
    }
    return false;
}

/// Runs tabu search over the TSP neighborhood generated by 1-1 exchanges.
/// Time complexity: O(iterations * V^3), Space complexity: O(size * V)
pub fn tabuSearch(
    allocator: std.mem.Allocator,
    first_solution: []const u8,
    first_cost: i32,
    graph: Graph,
    iterations: usize,
    size: usize,
) TabuSearchError!Tour {
    const solution = try allocator.dupe(u8, first_solution);
    defer allocator.free(solution);
    var best_solution = try allocator.dupe(u8, first_solution);
    errdefer allocator.free(best_solution);
    var best_cost = first_cost;

    var tabu_list = std.ArrayListUnmanaged(TabuMove){};
    defer tabu_list.deinit(allocator);

    var count: usize = 1;
    while (count <= iterations) : (count += 1) {
        const neighborhood = try findNeighborhood(solution, graph, allocator);
        defer {
            for (neighborhood) |*candidate| candidate.deinit(allocator);
            allocator.free(neighborhood);
        }

        var index_of_best_solution: usize = 0;
        var found = false;
        while (!found and index_of_best_solution < neighborhood.len) {
            const candidate = neighborhood[index_of_best_solution];
            var differing_index: usize = 0;
            while (differing_index < candidate.path.len and candidate.path[differing_index] == solution[differing_index]) : (differing_index += 1) {}
            if (differing_index == candidate.path.len) {
                index_of_best_solution += 1;
                continue;
            }

            const first_exchange_node = candidate.path[differing_index];
            const second_exchange_node = solution[differing_index];
            if (!isTabu(tabu_list.items, first_exchange_node, second_exchange_node)) {
                try tabu_list.append(allocator, .{ .first = first_exchange_node, .second = second_exchange_node });
                @memcpy(solution, candidate.path);
                found = true;

                if (candidate.cost < best_cost) {
                    allocator.free(best_solution);
                    best_solution = try allocator.dupe(u8, solution);
                    best_cost = candidate.cost;
                }
            } else {
                index_of_best_solution += 1;
            }
        }

        if (size > 0 and tabu_list.items.len >= size) {
            _ = tabu_list.orderedRemove(0);
        }
    }

    return .{ .path = best_solution, .cost = best_cost };
}

fn expectPathEqual(expected: []const u8, actual: []const u8) !void {
    try testing.expectEqualStrings(expected, actual);
}

const sample_graph_text =
    \\a b 20
    \\a c 18
    \\a d 22
    \\a e 26
    \\c b 10
    \\c d 23
    \\c e 24
    \\b d 11
    \\b e 12
    \\e d 40
;

test "tabu search: parse and greedy first solution" {
    var graph = try parseGraph(sample_graph_text, testing.allocator);
    defer graph.deinit();

    try testing.expectEqual(@as(usize, 5), graph.nodes.len);
    try testing.expectEqual(@as(u8, 'a'), graph.start_node);
    try testing.expectEqual(@as(?i32, 18), graph.costBetween('a', 'c'));
    try testing.expectEqual(@as(?i32, 18), graph.costBetween('c', 'a'));

    var first = try generateFirstSolution(graph, testing.allocator);
    defer first.deinit(testing.allocator);
    try expectPathEqual("acbdea", first.path);
    try testing.expectEqual(@as(i32, 105), first.cost);
}

test "tabu search: neighborhood matches python ordering" {
    var graph = try parseGraph(sample_graph_text, testing.allocator);
    defer graph.deinit();

    const route = "acbdea";
    const neighborhood = try findNeighborhood(route, graph, testing.allocator);
    defer {
        for (neighborhood) |*candidate| candidate.deinit(testing.allocator);
        testing.allocator.free(neighborhood);
    }

    try testing.expectEqual(@as(usize, 6), neighborhood.len);
    try expectPathEqual("aebdca", neighborhood[0].path);
    try testing.expectEqual(@as(i32, 90), neighborhood[0].cost);
    try expectPathEqual("acdbea", neighborhood[1].path);
    try testing.expectEqual(@as(i32, 90), neighborhood[1].cost);
    try expectPathEqual("abcdea", neighborhood[5].path);
    try testing.expectEqual(@as(i32, 119), neighborhood[5].cost);
}

test "tabu search: finds best sampled tour and handles extremes" {
    var graph = try parseGraph(sample_graph_text, testing.allocator);
    defer graph.deinit();

    var first = try generateFirstSolution(graph, testing.allocator);
    defer first.deinit(testing.allocator);

    var best = try tabuSearch(testing.allocator, first.path, first.cost, graph, 4, 3);
    defer best.deinit(testing.allocator);
    try expectPathEqual("adbeca", best.path);
    try testing.expectEqual(@as(i32, 87), best.cost);

    var immediate = try tabuSearch(testing.allocator, first.path, first.cost, graph, 0, 0);
    defer immediate.deinit(testing.allocator);
    try expectPathEqual(first.path, immediate.path);
    try testing.expectEqual(first.cost, immediate.cost);
}
