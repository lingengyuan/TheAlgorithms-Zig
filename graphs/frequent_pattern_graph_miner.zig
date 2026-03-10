//! Frequent Pattern Graph Miner - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/frequent_pattern_graph_miner.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const FrequencyEntry = struct {
    edge: []const u8,
    weight: usize,
    bitcode: []u8,
};

pub const NodeGroup = struct {
    bitcode: []const u8,
    edges: []const []const u8,
};

pub const ClusterBucket = struct {
    weight: usize,
    groups: []NodeGroup,
};

pub const EdgePair = struct {
    from: u8,
    to: u8,
};

pub const FrequentSubgraphResult = struct {
    edge_lists: [][]EdgePair,

    pub fn deinit(self: FrequentSubgraphResult, allocator: Allocator) void {
        for (self.edge_lists) |edges| allocator.free(edges);
        allocator.free(self.edge_lists);
    }
};

const GraphNode = struct {
    labels: []const []const u8,
    neighbors: []usize,
    bitcode: []const u8,
    weight: usize,
};

const FrequentGraph = struct {
    nodes: []GraphNode,
    header_index: usize,

    fn deinit(self: FrequentGraph, allocator: Allocator) void {
        for (self.nodes) |node| allocator.free(node.neighbors);
        allocator.free(self.nodes);
    }
};

/// Returns the normalized edge name for a raw token like `ab-e1`.
fn edgeName(raw: []const u8) []const u8 {
    return raw[0 .. std.mem.indexOfScalar(u8, raw, '-') orelse raw.len];
}

/// Returns distinct edge labels in first-seen order.
/// Time complexity: O(total_edges²) worst-case, Space complexity: O(total_edges)
pub fn getDistinctEdges(allocator: Allocator, edge_array: []const []const []const u8) ![][]const u8 {
    var distinct = std.ArrayListUnmanaged([]const u8){};
    defer distinct.deinit(allocator);

    for (edge_array) |row| {
        for (row) |raw| {
            const name = edgeName(raw);
            var found = false;
            for (distinct.items) |existing| {
                if (std.mem.eql(u8, existing, name)) {
                    found = true;
                    break;
                }
            }
            if (!found) try distinct.append(allocator, name);
        }
    }

    return try distinct.toOwnedSlice(allocator);
}

/// Returns a bitcode showing which graphs contain `distinct_edge`.
/// Time complexity: O(graphs * edges_per_graph), Space complexity: O(graphs)
pub fn getBitcode(
    allocator: Allocator,
    edge_array: []const []const []const u8,
    distinct_edge: []const u8,
) ![]u8 {
    const bitcode = try allocator.alloc(u8, edge_array.len);
    for (edge_array, 0..) |row, i| {
        bitcode[i] = '0';
        for (row) |raw| {
            if (std.mem.indexOf(u8, edgeName(raw), distinct_edge) != null) {
                bitcode[i] = '1';
                break;
            }
        }
    }
    return bitcode;
}

/// Builds the frequency table sorted by descending support.
/// Time complexity: O(total_edges² + distinct_edges * graphs * edges_per_graph), Space complexity: O(distinct_edges * graphs)
pub fn getFrequencyTable(
    allocator: Allocator,
    edge_array: []const []const []const u8,
) ![]FrequencyEntry {
    const distinct = try getDistinctEdges(allocator, edge_array);
    defer allocator.free(distinct);

    const table = try allocator.alloc(FrequencyEntry, distinct.len);
    errdefer {
        for (table[0..distinct.len]) |entry| allocator.free(entry.bitcode);
        allocator.free(table);
    }

    for (distinct, 0..) |edge, i| {
        const bitcode = try getBitcode(allocator, edge_array, edge);
        table[i] = .{
            .edge = edge,
            .weight = std.mem.count(u8, bitcode, "1"),
            .bitcode = bitcode,
        };
    }

    std.sort.heap(FrequencyEntry, table, {}, lessFrequencyEntry);
    return table;
}

fn lessFrequencyEntry(_: void, a: FrequencyEntry, b: FrequencyEntry) bool {
    if (a.weight != b.weight) return a.weight > b.weight;
    return std.mem.order(u8, a.edge, b.edge) == .lt;
}

/// Groups frequency entries by identical bitcode.
/// Time complexity: O(n²), Space complexity: O(n)
pub fn getNodes(allocator: Allocator, frequency_table: []const FrequencyEntry) ![]NodeGroup {
    var groups = std.ArrayListUnmanaged(NodeGroup){};
    defer groups.deinit(allocator);
    var edge_lists = std.ArrayListUnmanaged(std.ArrayListUnmanaged([]const u8)){};
    defer {
        for (edge_lists.items) |*list| list.deinit(allocator);
        edge_lists.deinit(allocator);
    }

    for (frequency_table) |entry| {
        var found_idx: ?usize = null;
        for (groups.items, 0..) |group, i| {
            if (std.mem.eql(u8, group.bitcode, entry.bitcode)) {
                found_idx = i;
                break;
            }
        }

        if (found_idx) |idx| {
            try edge_lists.items[idx].append(allocator, entry.edge);
        } else {
            try edge_lists.append(allocator, .{});
            try edge_lists.items[edge_lists.items.len - 1].append(allocator, entry.edge);
            try groups.append(allocator, .{
                .bitcode = entry.bitcode,
                .edges = &[_][]const u8{},
            });
        }
    }

    const out = try allocator.alloc(NodeGroup, groups.items.len);
    errdefer {
        for (out[0..groups.items.len]) |group| allocator.free(group.edges);
        allocator.free(out);
    }
    for (groups.items, 0..) |group, i| {
        out[i] = .{
            .bitcode = group.bitcode,
            .edges = try edge_lists.items[i].toOwnedSlice(allocator),
        };
    }
    return out;
}

/// Groups node groups by support weight in descending order.
/// Time complexity: O(n²), Space complexity: O(n)
pub fn getCluster(allocator: Allocator, nodes: []const NodeGroup) ![]ClusterBucket {
    var weights = std.ArrayListUnmanaged(usize){};
    defer weights.deinit(allocator);

    for (nodes) |group| {
        const weight = std.mem.count(u8, group.bitcode, "1");
        var seen = false;
        for (weights.items) |existing| {
            if (existing == weight) {
                seen = true;
                break;
            }
        }
        if (!seen) try weights.append(allocator, weight);
    }

    std.sort.heap(usize, weights.items, {}, descUsize);

    const buckets = try allocator.alloc(ClusterBucket, weights.items.len);
    errdefer {
        for (buckets[0..weights.items.len]) |bucket| {
            for (bucket.groups) |group| allocator.free(group.edges);
            allocator.free(bucket.groups);
        }
        allocator.free(buckets);
    }

    for (weights.items, 0..) |weight, i| {
        var matched = std.ArrayListUnmanaged(NodeGroup){};
        defer matched.deinit(allocator);
        for (nodes) |group| {
            if (std.mem.count(u8, group.bitcode, "1") == weight) {
                try matched.append(allocator, .{
                    .bitcode = group.bitcode,
                    .edges = try allocator.dupe([]const u8, group.edges),
                });
            }
        }
        buckets[i] = .{
            .weight = weight,
            .groups = try matched.toOwnedSlice(allocator),
        };
    }
    return buckets;
}

fn descUsize(_: void, a: usize, b: usize) bool {
    return a > b;
}

/// Returns support percentages for each cluster bucket.
/// Time complexity: O(k), Space complexity: O(k)
pub fn getSupport(allocator: Allocator, cluster: []const ClusterBucket) ![]f64 {
    const out = try allocator.alloc(f64, cluster.len);
    for (cluster, 0..) |bucket, i| {
        out[i] = (@as(f64, @floatFromInt(bucket.weight)) * 100.0) / @as(f64, @floatFromInt(cluster.len));
    }
    return out;
}

fn buildFrequentGraph(allocator: Allocator, cluster: []const ClusterBucket) !FrequentGraph {
    if (cluster.len == 0) return .{ .nodes = try allocator.alloc(GraphNode, 0), .header_index = 0 };

    const total_groups = blk: {
        var count: usize = 1; // header
        for (cluster) |bucket| count += bucket.groups.len;
        break :blk count;
    };

    const nodes = try allocator.alloc(GraphNode, total_groups);
    errdefer allocator.free(nodes);
    nodes[0] = .{
        .labels = &[_][]const u8{"Header"},
        .neighbors = &[_]usize{},
        .bitcode = "",
        .weight = cluster[0].weight + 1,
    };

    var bucket_indices = std.AutoHashMap(usize, std.ArrayListUnmanaged(usize)).init(allocator);
    defer {
        var it = bucket_indices.iterator();
        while (it.next()) |entry| entry.value_ptr.deinit(allocator);
        bucket_indices.deinit();
    }

    var next_index: usize = 1;
    for (cluster) |bucket| {
        var list = std.ArrayListUnmanaged(usize){};
        for (bucket.groups) |group| {
            nodes[next_index] = .{
                .labels = group.edges,
                .neighbors = &[_]usize{},
                .bitcode = group.bitcode,
                .weight = bucket.weight,
            };
            try list.append(allocator, next_index);
            next_index += 1;
        }
        try bucket_indices.put(bucket.weight, list);
    }

    const max_weight = cluster[0].weight;
    if (bucket_indices.get(max_weight)) |top_groups| {
        nodes[0].neighbors = try allocator.dupe(usize, top_groups.items);
        for (top_groups.items) |idx| {
            nodes[idx].neighbors = try allocator.dupe(usize, &[_]usize{0});
        }
    } else {
        nodes[0].neighbors = try allocator.alloc(usize, 0);
    }

    for (cluster) |bucket| {
        if (bucket.weight == max_weight) continue;
        if (bucket_indices.get(bucket.weight)) |from_indices| {
            for (from_indices.items) |from_idx| {
                var edges = std.ArrayListUnmanaged(usize){};
                defer edges.deinit(allocator);

                var candidate_weight = bucket.weight + 1;
                while (candidate_weight <= max_weight) : (candidate_weight += 1) {
                    if (bucket_indices.get(candidate_weight)) |to_indices| {
                        for (to_indices.items) |to_idx| {
                            if (isSubsetBitcode(nodes[from_idx].bitcode, nodes[to_idx].bitcode)) {
                                try edges.append(allocator, to_idx);
                            }
                        }
                        if (edges.items.len > 0) break;
                    }
                }

                nodes[from_idx].neighbors = try edges.toOwnedSlice(allocator);
            }
        }
    }

    return .{ .nodes = nodes, .header_index = 0 };
}

fn isSubsetBitcode(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, 0..) |ch, i| {
        if (ch == '1' and b[i] != '1') return false;
    }
    return true;
}

fn dfsPaths(
    allocator: Allocator,
    graph: FrequentGraph,
    current: usize,
    end: usize,
    path: *std.ArrayListUnmanaged(usize),
    all_paths: *std.ArrayListUnmanaged([]usize),
) !void {
    try path.append(allocator, current);
    defer _ = path.pop();

    if (current == end) {
        try all_paths.append(allocator, try allocator.dupe(usize, path.items));
        return;
    }

    for (graph.nodes[current].neighbors) |neighbor| {
        var already = false;
        for (path.items) |seen| {
            if (seen == neighbor) {
                already = true;
                break;
            }
        }
        if (!already) try dfsPaths(allocator, graph, neighbor, end, path, all_paths);
    }
}

/// Mines frequent subgraphs for a given support percentage.
/// Time complexity: input-dependent; worst-case exponential in DFS path enumeration.
/// Space complexity: O(number_of_nodes + path_storage)
pub fn mineFrequentSubgraphs(
    allocator: Allocator,
    edge_array: []const []const []const u8,
    support_percent: f64,
) !FrequentSubgraphResult {
    const frequency_table = try getFrequencyTable(allocator, edge_array);
    defer {
        for (frequency_table) |entry| allocator.free(entry.bitcode);
        allocator.free(frequency_table);
    }

    const nodes = try getNodes(allocator, frequency_table);
    defer {
        for (nodes) |group| allocator.free(group.edges);
        allocator.free(nodes);
    }

    const cluster = try getCluster(allocator, nodes);
    defer {
        for (cluster) |bucket| {
            for (bucket.groups) |group| allocator.free(group.edges);
            allocator.free(bucket.groups);
        }
        allocator.free(cluster);
    }

    var graph = try buildFrequentGraph(allocator, cluster);
    defer graph.deinit(allocator);
    if (cluster.len == 0) {
        return .{ .edge_lists = try allocator.alloc([]EdgePair, 0) };
    }

    const bucket_weight = @as(usize, @intFromFloat((support_percent / 100.0) * @as(f64, @floatFromInt(cluster.len))));
    if (bucket_weight == 0) {
        return .{ .edge_lists = try allocator.alloc([]EdgePair, 0) };
    }
    var target_indices = std.ArrayListUnmanaged(usize){};
    defer target_indices.deinit(allocator);
    for (graph.nodes, 0..) |node, idx| {
        if (idx == graph.header_index) continue;
        if (node.weight == bucket_weight) try target_indices.append(allocator, idx);
    }

    var path = std.ArrayListUnmanaged(usize){};
    defer path.deinit(allocator);
    var all_paths = std.ArrayListUnmanaged([]usize){};
    defer {
        for (all_paths.items) |p| allocator.free(p);
        all_paths.deinit(allocator);
    }

    for (target_indices.items) |start_idx| {
        try dfsPaths(allocator, graph, start_idx, graph.header_index, &path, &all_paths);
    }

    const edge_lists = try allocator.alloc([]EdgePair, all_paths.items.len);
    errdefer {
        for (edge_lists[0..all_paths.items.len]) |edges| allocator.free(edges);
        allocator.free(edge_lists);
    }

    for (all_paths.items, 0..) |node_path, i| {
        var pairs = std.ArrayListUnmanaged(EdgePair){};
        defer pairs.deinit(allocator);

        if (node_path.len >= 2) {
            for (node_path[0 .. node_path.len - 1]) |node_idx| {
                for (graph.nodes[node_idx].labels) |label| {
                    if (std.mem.eql(u8, label, "Header")) continue;
                    if (label.len < 2) continue;
                    try pairs.append(allocator, .{
                        .from = label[0],
                        .to = label[1],
                    });
                }
            }
        }

        edge_lists[i] = try pairs.toOwnedSlice(allocator);
    }

    return .{ .edge_lists = edge_lists };
}

test "frequent pattern graph miner: distinct edges and bitcode" {
    const alloc = testing.allocator;
    const edge_array = [_][]const []const u8{
        &[_][]const u8{ "ab-e1", "ac-e3", "bc-e4" },
        &[_][]const u8{ "ab-e1", "bc-e4" },
    };

    const distinct = try getDistinctEdges(alloc, &edge_array);
    defer alloc.free(distinct);
    try testing.expectEqual(@as(usize, 3), distinct.len);
    try testing.expectEqualStrings("ab", distinct[0]);
    try testing.expectEqualStrings("ac", distinct[1]);
    try testing.expectEqualStrings("bc", distinct[2]);

    const bitcode = try getBitcode(alloc, &edge_array, "ac");
    defer alloc.free(bitcode);
    try testing.expectEqualStrings("10", bitcode);
}

test "frequent pattern graph miner: get nodes sample" {
    const alloc = testing.allocator;
    const table = [_]FrequencyEntry{
        .{ .edge = "ab", .weight = 5, .bitcode = try alloc.dupe(u8, "11111") },
        .{ .edge = "ac", .weight = 5, .bitcode = try alloc.dupe(u8, "11111") },
        .{ .edge = "df", .weight = 5, .bitcode = try alloc.dupe(u8, "11111") },
        .{ .edge = "bd", .weight = 5, .bitcode = try alloc.dupe(u8, "11111") },
        .{ .edge = "bc", .weight = 5, .bitcode = try alloc.dupe(u8, "11111") },
    };
    defer for (table) |entry| alloc.free(entry.bitcode);

    const nodes = try getNodes(alloc, &table);
    defer {
        for (nodes) |group| alloc.free(group.edges);
        alloc.free(nodes);
    }

    try testing.expectEqual(@as(usize, 1), nodes.len);
    try testing.expectEqualStrings("11111", nodes[0].bitcode);
    try testing.expectEqual(@as(usize, 5), nodes[0].edges.len);
    try testing.expectEqualStrings("ab", nodes[0].edges[0]);
    try testing.expectEqualStrings("ac", nodes[0].edges[1]);
    try testing.expectEqualStrings("df", nodes[0].edges[2]);
    try testing.expectEqualStrings("bd", nodes[0].edges[3]);
    try testing.expectEqualStrings("bc", nodes[0].edges[4]);
}

test "frequent pattern graph miner: support sample" {
    const alloc = testing.allocator;
    const buckets = [_]ClusterBucket{
        .{ .weight = 5, .groups = &[_]NodeGroup{} },
        .{ .weight = 4, .groups = &[_]NodeGroup{} },
        .{ .weight = 3, .groups = &[_]NodeGroup{} },
        .{ .weight = 2, .groups = &[_]NodeGroup{} },
        .{ .weight = 1, .groups = &[_]NodeGroup{} },
    };

    const support = try getSupport(alloc, &buckets);
    defer alloc.free(support);
    try testing.expectEqual(@as(usize, 5), support.len);
    try testing.expectApproxEqAbs(@as(f64, 100.0), support[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 80.0), support[1], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 60.0), support[2], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 40.0), support[3], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 20.0), support[4], 1e-12);
}

test "frequent pattern graph miner: mine sample support" {
    const alloc = testing.allocator;
    const edge_array = [_][]const []const u8{
        &[_][]const u8{ "ab-e1", "ac-e3", "ad-e5", "bc-e4", "bd-e2", "be-e6", "bh-e12", "cd-e2", "ce-e4", "de-e1", "df-e8", "dg-e5", "dh-e10", "ef-e3", "eg-e2", "fg-e6", "gh-e6", "hi-e3" },
        &[_][]const u8{ "ab-e1", "ac-e3", "ad-e5", "bc-e4", "bd-e2", "be-e6", "cd-e2", "de-e1", "df-e8", "ef-e3", "eg-e2", "fg-e6" },
        &[_][]const u8{ "ab-e1", "ac-e3", "bc-e4", "bd-e2", "de-e1", "df-e8", "dg-e5", "ef-e3", "eg-e2", "eh-e12", "fg-e6", "fh-e10", "gh-e6" },
        &[_][]const u8{ "ab-e1", "ac-e3", "bc-e4", "bd-e2", "bh-e12", "cd-e2", "df-e8", "dh-e10" },
        &[_][]const u8{ "ab-e1", "ac-e3", "ad-e5", "bc-e4", "bd-e2", "cd-e2", "ce-e4", "de-e1", "df-e8", "dg-e5", "ef-e3", "eg-e2", "fg-e6" },
    };

    var result = try mineFrequentSubgraphs(alloc, &edge_array, 60.0);
    defer result.deinit(alloc);

    try testing.expect(result.edge_lists.len > 0);
    for (result.edge_lists) |edges| {
        try testing.expect(edges.len > 0);
        for (edges) |edge| {
            try testing.expect(edge.from >= 'a' and edge.from <= 'z');
            try testing.expect(edge.to >= 'a' and edge.to <= 'z');
        }
    }
}

test "frequent pattern graph miner: extreme repeated identical graphs" {
    const alloc = testing.allocator;
    const edge_array = [_][]const []const u8{
        &[_][]const u8{ "ab-e1", "bc-e2", "cd-e3" },
        &[_][]const u8{ "ab-e1", "bc-e2", "cd-e3" },
        &[_][]const u8{ "ab-e1", "bc-e2", "cd-e3" },
        &[_][]const u8{ "ab-e1", "bc-e2", "cd-e3" },
    };

    var result = try mineFrequentSubgraphs(alloc, &edge_array, 75.0);
    defer result.deinit(alloc);
    try testing.expectEqual(@as(usize, 0), result.edge_lists.len);
}
