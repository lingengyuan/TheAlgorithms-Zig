//! Directed and Undirected Weighted Graph Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/directed_and_undirected_weighted_graph.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Neighbor = struct {
    weight: i64,
    to: i64,
};

const AdjacencyMap = std.AutoArrayHashMap(i64, std.ArrayListUnmanaged(Neighbor));

pub const DirectedGraph = struct {
    allocator: Allocator,
    graph: AdjacencyMap,

    pub fn init(allocator: Allocator) DirectedGraph {
        return .{
            .allocator = allocator,
            .graph = AdjacencyMap.init(allocator),
        };
    }

    pub fn deinit(self: *DirectedGraph) void {
        var it = self.graph.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.graph.deinit();
    }

    pub fn addPair(self: *DirectedGraph, u: i64, v: i64, weight: i64) !void {
        try self.ensureNode(u);
        try self.ensureNode(v);

        if (!(try self.containsNeighbor(u, v, weight))) {
            try self.graph.getPtr(u).?.append(self.allocator, .{ .weight = weight, .to = v });
        }
    }

    pub fn removePair(self: *DirectedGraph, u: i64, v: i64) !void {
        if (self.graph.getPtr(u)) |neighbors| {
            _ = removeFirstNeighbor(neighbors, v);
        }
    }

    pub fn allNodes(self: *const DirectedGraph, allocator: Allocator) ![]i64 {
        const out = try allocator.alloc(i64, self.graph.count());
        for (self.graph.keys(), 0..) |key, i| out[i] = key;
        return out;
    }

    pub fn dfs(self: *const DirectedGraph, allocator: Allocator, start: ?i64, destination: ?i64) ![]i64 {
        return traverseDepthFirst(self.graph, allocator, start, destination);
    }

    pub fn bfs(self: *const DirectedGraph, allocator: Allocator, start: ?i64) ![]i64 {
        return traverseBreadthFirst(self.graph, allocator, start);
    }

    pub fn inDegree(self: *const DirectedGraph, node: i64) !usize {
        _ = self.graph.get(node) orelse return error.InvalidNode;
        var count: usize = 0;
        var it = self.graph.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.items) |neighbor| {
                if (neighbor.to == node) count += 1;
            }
        }
        return count;
    }

    pub fn outDegree(self: *const DirectedGraph, node: i64) !usize {
        const neighbors = self.graph.get(node) orelse return error.InvalidNode;
        return neighbors.items.len;
    }

    pub fn topologicalSort(self: *const DirectedGraph, allocator: Allocator, start: ?i64) ![]i64 {
        const start_node = start orelse self.firstNode() orelse return try allocator.alloc(i64, 0);
        if (!self.graph.contains(start_node)) return error.InvalidNode;

        var visited = std.AutoHashMap(i64, void).init(allocator);
        defer visited.deinit();
        var order = std.ArrayListUnmanaged(i64){};
        defer order.deinit(allocator);
        try topoDfs(self.graph, allocator, start_node, &visited, &order);
        return try order.toOwnedSlice(allocator);
    }

    pub fn hasCycle(self: *const DirectedGraph) !bool {
        var state = std.AutoHashMap(i64, u8).init(self.allocator);
        defer state.deinit();

        for (self.graph.keys()) |node| {
            if ((state.get(node) orelse 0) == 0) {
                if (try hasCycleDirected(self.graph, self.allocator, node, &state)) return true;
            }
        }
        return false;
    }

    pub fn cycleNodes(self: *const DirectedGraph, allocator: Allocator) ![]i64 {
        var state = std.AutoHashMap(i64, u8).init(allocator);
        defer state.deinit();
        var stack = std.ArrayListUnmanaged(i64){};
        defer stack.deinit(allocator);

        for (self.graph.keys()) |node| {
            if ((state.get(node) orelse 0) != 0) continue;
            if (try findDirectedCycle(self.graph, allocator, node, &state, &stack)) |cycle| {
                return cycle;
            }
        }
        return try allocator.alloc(i64, 0);
    }

    pub fn bfsTime(self: *const DirectedGraph, allocator: Allocator, start: ?i64) !u64 {
        const begin = std.time.nanoTimestamp();
        const visited = try self.bfs(allocator, start);
        defer allocator.free(visited);
        const end = std.time.nanoTimestamp();
        return @as(u64, @intCast(end - begin));
    }

    pub fn dfsTime(self: *const DirectedGraph, allocator: Allocator, start: ?i64, destination: ?i64) !u64 {
        const begin = std.time.nanoTimestamp();
        const visited = try self.dfs(allocator, start, destination);
        defer allocator.free(visited);
        const end = std.time.nanoTimestamp();
        return @as(u64, @intCast(end - begin));
    }

    fn ensureNode(self: *DirectedGraph, node: i64) !void {
        const gop = try self.graph.getOrPut(node);
        if (!gop.found_existing) gop.value_ptr.* = .{};
    }

    fn containsNeighbor(self: *const DirectedGraph, u: i64, v: i64, weight: i64) !bool {
        const neighbors = self.graph.get(u) orelse return error.InvalidNode;
        for (neighbors.items) |neighbor| {
            if (neighbor.to == v and neighbor.weight == weight) return true;
        }
        return false;
    }

    fn firstNode(self: *const DirectedGraph) ?i64 {
        if (self.graph.count() == 0) return null;
        return self.graph.keys()[0];
    }
};

pub const Graph = struct {
    allocator: Allocator,
    graph: AdjacencyMap,

    pub fn init(allocator: Allocator) Graph {
        return .{
            .allocator = allocator,
            .graph = AdjacencyMap.init(allocator),
        };
    }

    pub fn deinit(self: *Graph) void {
        var it = self.graph.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.graph.deinit();
    }

    pub fn addPair(self: *Graph, u: i64, v: i64, weight: i64) !void {
        try self.ensureNode(u);
        try self.ensureNode(v);
        if (!(try self.containsNeighbor(u, v, weight))) {
            try self.graph.getPtr(u).?.append(self.allocator, .{ .weight = weight, .to = v });
        }
        if (!(try self.containsNeighbor(v, u, weight))) {
            try self.graph.getPtr(v).?.append(self.allocator, .{ .weight = weight, .to = u });
        }
    }

    pub fn removePair(self: *Graph, u: i64, v: i64) !void {
        if (self.graph.getPtr(u)) |neighbors| {
            _ = removeFirstNeighbor(neighbors, v);
        }
        if (self.graph.getPtr(v)) |neighbors| {
            _ = removeFirstNeighbor(neighbors, u);
        }
    }

    pub fn allNodes(self: *const Graph, allocator: Allocator) ![]i64 {
        const out = try allocator.alloc(i64, self.graph.count());
        for (self.graph.keys(), 0..) |key, i| out[i] = key;
        return out;
    }

    pub fn dfs(self: *const Graph, allocator: Allocator, start: ?i64, destination: ?i64) ![]i64 {
        return traverseDepthFirst(self.graph, allocator, start, destination);
    }

    pub fn bfs(self: *const Graph, allocator: Allocator, start: ?i64) ![]i64 {
        return traverseBreadthFirst(self.graph, allocator, start);
    }

    pub fn degree(self: *const Graph, node: i64) !usize {
        const neighbors = self.graph.get(node) orelse return error.InvalidNode;
        return neighbors.items.len;
    }

    pub fn hasCycle(self: *const Graph) !bool {
        var visited = std.AutoHashMap(i64, void).init(self.allocator);
        defer visited.deinit();

        for (self.graph.keys()) |node| {
            if (!visited.contains(node)) {
                if (try hasCycleUndirected(self.graph, self.allocator, node, null, &visited)) return true;
            }
        }
        return false;
    }

    pub fn cycleNodes(self: *const Graph, allocator: Allocator) ![]i64 {
        var visited = std.AutoHashMap(i64, void).init(allocator);
        defer visited.deinit();
        var path = std.ArrayListUnmanaged(i64){};
        defer path.deinit(allocator);

        for (self.graph.keys()) |node| {
            if (!visited.contains(node)) {
                if (try findUndirectedCycle(self.graph, allocator, node, null, &visited, &path)) |cycle| {
                    return cycle;
                }
            }
        }
        return try allocator.alloc(i64, 0);
    }

    pub fn bfsTime(self: *const Graph, allocator: Allocator, start: ?i64) !u64 {
        const begin = std.time.nanoTimestamp();
        const visited = try self.bfs(allocator, start);
        defer allocator.free(visited);
        const end = std.time.nanoTimestamp();
        return @as(u64, @intCast(end - begin));
    }

    pub fn dfsTime(self: *const Graph, allocator: Allocator, start: ?i64, destination: ?i64) !u64 {
        const begin = std.time.nanoTimestamp();
        const visited = try self.dfs(allocator, start, destination);
        defer allocator.free(visited);
        const end = std.time.nanoTimestamp();
        return @as(u64, @intCast(end - begin));
    }

    fn ensureNode(self: *Graph, node: i64) !void {
        const gop = try self.graph.getOrPut(node);
        if (!gop.found_existing) gop.value_ptr.* = .{};
    }

    fn containsNeighbor(self: *const Graph, u: i64, v: i64, weight: i64) !bool {
        const neighbors = self.graph.get(u) orelse return error.InvalidNode;
        for (neighbors.items) |neighbor| {
            if (neighbor.to == v and neighbor.weight == weight) return true;
        }
        return false;
    }
};

fn removeFirstNeighbor(neighbors: *std.ArrayListUnmanaged(Neighbor), target: i64) bool {
    for (neighbors.items, 0..) |neighbor, i| {
        if (neighbor.to == target) {
            _ = neighbors.orderedRemove(i);
            return true;
        }
    }
    return false;
}

fn traverseDepthFirst(
    graph: AdjacencyMap,
    allocator: Allocator,
    start: ?i64,
    destination: ?i64,
) ![]i64 {
    if (start != null and destination != null and start.? == destination.?) {
        return try allocator.alloc(i64, 0);
    }
    const start_node = start orelse if (graph.count() > 0) graph.keys()[0] else return try allocator.alloc(i64, 0);
    if (!graph.contains(start_node)) return error.InvalidNode;

    var stack = std.ArrayListUnmanaged(i64){};
    defer stack.deinit(allocator);
    var visited_order = std.ArrayListUnmanaged(i64){};
    defer visited_order.deinit(allocator);
    var visited = std.AutoHashMap(i64, void).init(allocator);
    defer visited.deinit();

    try stack.append(allocator, start_node);
    try visited.put(start_node, {});
    try visited_order.append(allocator, start_node);

    while (stack.items.len > 0) {
        const current = stack.items[stack.items.len - 1];
        const neighbors = graph.get(current).?;
        var advanced = false;

        for (neighbors.items) |neighbor| {
            if (visited.contains(neighbor.to)) continue;
            if (destination) |dest| {
                if (neighbor.to == dest) {
                    try visited_order.append(allocator, dest);
                    return try visited_order.toOwnedSlice(allocator);
                }
            }

            try stack.append(allocator, neighbor.to);
            try visited.put(neighbor.to, {});
            try visited_order.append(allocator, neighbor.to);
            advanced = true;
            break;
        }

        if (!advanced) _ = stack.pop();
    }

    return try visited_order.toOwnedSlice(allocator);
}

fn traverseBreadthFirst(graph: AdjacencyMap, allocator: Allocator, start: ?i64) ![]i64 {
    const start_node = start orelse if (graph.count() > 0) graph.keys()[0] else return try allocator.alloc(i64, 0);
    if (!graph.contains(start_node)) return error.InvalidNode;

    var queue = std.ArrayListUnmanaged(i64){};
    defer queue.deinit(allocator);
    var visited_order = std.ArrayListUnmanaged(i64){};
    defer visited_order.deinit(allocator);
    var visited = std.AutoHashMap(i64, void).init(allocator);
    defer visited.deinit();

    try queue.append(allocator, start_node);
    try visited.put(start_node, {});
    try visited_order.append(allocator, start_node);

    var head: usize = 0;
    while (head < queue.items.len) {
        const current = queue.items[head];
        head += 1;
        const neighbors = graph.get(current).?;
        for (neighbors.items) |neighbor| {
            if (visited.contains(neighbor.to)) continue;
            try visited.put(neighbor.to, {});
            try queue.append(allocator, neighbor.to);
            try visited_order.append(allocator, neighbor.to);
        }
    }

    return try visited_order.toOwnedSlice(allocator);
}

fn topoDfs(
    graph: AdjacencyMap,
    allocator: Allocator,
    node: i64,
    visited: *std.AutoHashMap(i64, void),
    order: *std.ArrayListUnmanaged(i64),
) !void {
    try visited.put(node, {});
    const neighbors = graph.get(node).?;
    for (neighbors.items) |neighbor| {
        if (!visited.contains(neighbor.to)) {
            try topoDfs(graph, allocator, neighbor.to, visited, order);
        }
    }
    try order.append(allocator, node);
}

fn hasCycleDirected(
    graph: AdjacencyMap,
    allocator: Allocator,
    node: i64,
    state: *std.AutoHashMap(i64, u8),
) !bool {
    try state.put(node, 1);
    const neighbors = graph.get(node).?;
    for (neighbors.items) |neighbor| {
        const next_state = state.get(neighbor.to) orelse 0;
        if (next_state == 1) return true;
        if (next_state == 0 and try hasCycleDirected(graph, allocator, neighbor.to, state)) return true;
    }
    try state.put(node, 2);
    return false;
}

fn findDirectedCycle(
    graph: AdjacencyMap,
    allocator: Allocator,
    node: i64,
    state: *std.AutoHashMap(i64, u8),
    stack: *std.ArrayListUnmanaged(i64),
) !?[]i64 {
    try state.put(node, 1);
    try stack.append(allocator, node);

    const neighbors = graph.get(node).?;
    for (neighbors.items) |neighbor| {
        const next_state = state.get(neighbor.to) orelse 0;
        if (next_state == 0) {
            if (try findDirectedCycle(graph, allocator, neighbor.to, state, stack)) |cycle| return cycle;
        } else if (next_state == 1) {
            for (stack.items, 0..) |stack_node, i| {
                if (stack_node == neighbor.to) {
                    return try allocator.dupe(i64, stack.items[i..]);
                }
            }
        }
    }

    _ = stack.pop();
    try state.put(node, 2);
    return null;
}

fn hasCycleUndirected(
    graph: AdjacencyMap,
    allocator: Allocator,
    node: i64,
    parent: ?i64,
    visited: *std.AutoHashMap(i64, void),
) !bool {
    try visited.put(node, {});
    const neighbors = graph.get(node).?;
    for (neighbors.items) |neighbor| {
        if (parent != null and neighbor.to == parent.?) continue;
        if (visited.contains(neighbor.to)) return true;
        if (try hasCycleUndirected(graph, allocator, neighbor.to, node, visited)) return true;
    }
    return false;
}

fn findUndirectedCycle(
    graph: AdjacencyMap,
    allocator: Allocator,
    node: i64,
    parent: ?i64,
    visited: *std.AutoHashMap(i64, void),
    path: *std.ArrayListUnmanaged(i64),
) !?[]i64 {
    try visited.put(node, {});
    try path.append(allocator, node);
    const neighbors = graph.get(node).?;

    for (neighbors.items) |neighbor| {
        if (parent != null and neighbor.to == parent.?) continue;
        if (visited.contains(neighbor.to)) {
            for (path.items, 0..) |path_node, i| {
                if (path_node == neighbor.to) {
                    return try allocator.dupe(i64, path.items[i..]);
                }
            }
        }
        if (!visited.contains(neighbor.to)) {
            if (try findUndirectedCycle(graph, allocator, neighbor.to, node, visited, path)) |cycle| return cycle;
        }
    }

    _ = path.pop();
    return null;
}

test "directed weighted graph: add remove bfs dfs and degrees" {
    const alloc = testing.allocator;
    var graph = DirectedGraph.init(alloc);
    defer graph.deinit();

    try graph.addPair(1, 2, 7);
    try graph.addPair(1, 3, 9);
    try graph.addPair(2, 4, 1);
    try graph.addPair(1, 2, 7); // duplicate ignored

    const nodes = try graph.allNodes(alloc);
    defer alloc.free(nodes);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4 }, nodes);

    const bfs_order = try graph.bfs(alloc, 1);
    defer alloc.free(bfs_order);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4 }, bfs_order);

    const dfs_order = try graph.dfs(alloc, 1, null);
    defer alloc.free(dfs_order);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 4, 3 }, dfs_order);

    try testing.expectEqual(@as(usize, 0), try graph.inDegree(1));
    try testing.expectEqual(@as(usize, 2), try graph.outDegree(1));

    try graph.removePair(1, 3);
    try testing.expectEqual(@as(usize, 1), try graph.outDegree(1));
}

test "directed weighted graph: topo and cycle helpers" {
    const alloc = testing.allocator;
    var dag = DirectedGraph.init(alloc);
    defer dag.deinit();

    try dag.addPair(0, 1, 1);
    try dag.addPair(1, 2, 1);
    try dag.addPair(0, 3, 1);

    const topo = try dag.topologicalSort(alloc, 0);
    defer alloc.free(topo);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 1, 3, 0 }, topo);
    try testing.expect(!(try dag.hasCycle()));

    var cyclic = DirectedGraph.init(alloc);
    defer cyclic.deinit();
    try cyclic.addPair(1, 2, 1);
    try cyclic.addPair(2, 3, 1);
    try cyclic.addPair(3, 1, 1);

    try testing.expect(try cyclic.hasCycle());
    const cycle = try cyclic.cycleNodes(alloc);
    defer alloc.free(cycle);
    try testing.expect(cycle.len >= 3);
}

test "undirected weighted graph: mirrors edges and cycle detection" {
    const alloc = testing.allocator;
    var graph = Graph.init(alloc);
    defer graph.deinit();

    try graph.addPair(1, 2, 5);
    try graph.addPair(2, 3, 4);
    try graph.addPair(3, 1, 3);

    const bfs_order = try graph.bfs(alloc, 1);
    defer alloc.free(bfs_order);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3 }, bfs_order);

    const dfs_order = try graph.dfs(alloc, 1, null);
    defer alloc.free(dfs_order);
    try testing.expectEqual(@as(usize, 3), dfs_order.len);

    try testing.expectEqual(@as(usize, 2), try graph.degree(1));
    try testing.expect(try graph.hasCycle());

    const cycle = try graph.cycleNodes(alloc);
    defer alloc.free(cycle);
    try testing.expect(cycle.len >= 3);

    try graph.removePair(1, 2);
    try testing.expectEqual(@as(usize, 1), try graph.degree(1));
}

test "weighted graph utilities: invalid start and destination short circuit" {
    const alloc = testing.allocator;
    var graph = DirectedGraph.init(alloc);
    defer graph.deinit();
    try graph.addPair(0, 1, 1);

    try testing.expectError(error.InvalidNode, graph.bfs(alloc, 9));

    const same = try graph.dfs(alloc, 0, 0);
    defer alloc.free(same);
    try testing.expectEqual(@as(usize, 0), same.len);
}

test "weighted graph utilities: extreme chain and timing helpers" {
    const alloc = testing.allocator;
    var graph = Graph.init(alloc);
    defer graph.deinit();

    const n: i64 = 150;
    var i: i64 = 0;
    while (i + 1 < n) : (i += 1) {
        try graph.addPair(i, i + 1, 1);
    }

    const bfs_order = try graph.bfs(alloc, 0);
    defer alloc.free(bfs_order);
    try testing.expectEqual(@as(usize, @intCast(n)), bfs_order.len);

    const bfs_time = try graph.bfsTime(alloc, 0);
    const dfs_time = try graph.dfsTime(alloc, 0, null);
    try testing.expect(bfs_time >= 0);
    try testing.expect(dfs_time >= 0);
}
