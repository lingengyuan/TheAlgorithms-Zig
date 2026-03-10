//! Lanczos Eigenvectors - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/lanczos_eigenvectors.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const LanczosDecomposition = struct {
    tridiagonal: []f64,
    basis: []f64,
    num_nodes: usize,
    num_eigenvectors: usize,

    pub fn deinit(self: LanczosDecomposition, allocator: Allocator) void {
        allocator.free(self.tridiagonal);
        allocator.free(self.basis);
    }
};

pub const LanczosResult = struct {
    eigenvalues: []f64,
    eigenvectors: []f64,
    num_nodes: usize,
    num_eigenvectors: usize,

    pub fn deinit(self: LanczosResult, allocator: Allocator) void {
        allocator.free(self.eigenvalues);
        allocator.free(self.eigenvectors);
    }
};

/// Validates an adjacency-list graph for the Lanczos method.
/// Time complexity: O(V + E), Space complexity: O(1)
pub fn validateAdjacencyList(graph: []const []const usize) !void {
    for (graph, 0..) |neighbors, node_index| {
        _ = node_index;
        for (neighbors) |neighbor_index| {
            if (neighbor_index >= graph.len) {
                return error.InvalidNeighbor;
            }
        }
    }
}

/// Multiplies the adjacency-list graph by a vector.
/// Time complexity: O(V + E), Space complexity: O(V)
pub fn multiplyMatrixVector(
    allocator: Allocator,
    graph: []const []const usize,
    vector: []const f64,
) ![]f64 {
    if (vector.len != graph.len) return error.InvalidVectorLength;

    const result = try allocator.alloc(f64, graph.len);
    @memset(result, 0.0);
    for (graph, 0..) |neighbors, node_index| {
        for (neighbors) |neighbor_index| {
            result[node_index] += vector[neighbor_index];
        }
    }
    return result;
}

/// Constructs a tridiagonal matrix and orthonormal basis using the Lanczos method.
/// The initial vector is deterministic to keep tests stable across runs.
/// Time complexity: O(k * (V + E + k)), Space complexity: O(V * k + k²)
pub fn lanczosIteration(
    allocator: Allocator,
    graph: []const []const usize,
    num_eigenvectors: usize,
) !LanczosDecomposition {
    const num_nodes = graph.len;
    if (num_nodes == 0) return error.EmptyGraph;
    if (num_eigenvectors == 0 or num_eigenvectors > num_nodes) return error.InvalidEigenvectorCount;

    const tridiagonal = try allocator.alloc(f64, num_eigenvectors * num_eigenvectors);
    errdefer allocator.free(tridiagonal);
    const basis = try allocator.alloc(f64, num_nodes * num_eigenvectors);
    errdefer allocator.free(basis);
    @memset(tridiagonal, 0.0);
    @memset(basis, 0.0);

    // Deterministic start vector: [1, 2, ..., n] normalized.
    var norm_sq: f64 = 0.0;
    for (0..num_nodes) |i| {
        const value = @as(f64, @floatFromInt(i + 1));
        basis[i * num_eigenvectors] = value;
        norm_sq += value * value;
    }
    const start_norm = @sqrt(norm_sq);
    for (0..num_nodes) |i| {
        basis[i * num_eigenvectors] /= start_norm;
    }

    var prev_beta: f64 = 0.0;
    const work = try allocator.alloc(f64, num_nodes);
    defer allocator.free(work);
    const q_buffer = try allocator.alloc(f64, num_nodes);
    defer allocator.free(q_buffer);

    for (0..num_eigenvectors) |iter_index| {
        readColumn(basis, num_nodes, num_eigenvectors, iter_index, q_buffer);
        const aq = try multiplyMatrixVector(allocator, graph, q_buffer);
        defer allocator.free(aq);
        @memcpy(work, aq);

        if (iter_index > 0) {
            axpyColumnInPlace(work, basis, num_nodes, num_eigenvectors, iter_index - 1, -prev_beta);
        }

        const alpha = dot(q_buffer, work);
        axpyInPlace(work, q_buffer, -alpha);

        // Full re-orthogonalization keeps the deterministic basis numerically stable.
        for (0..iter_index) |prev_index| {
            const projection = dotColumn(basis, num_nodes, num_eigenvectors, prev_index, work);
            axpyColumnInPlace(work, basis, num_nodes, num_eigenvectors, prev_index, -projection);
        }

        const beta = vectorNorm(work);
        tridiagonal[iter_index * num_eigenvectors + iter_index] = alpha;
        if (iter_index + 1 < num_eigenvectors) {
            tridiagonal[iter_index * num_eigenvectors + (iter_index + 1)] = beta;
            tridiagonal[(iter_index + 1) * num_eigenvectors + iter_index] = beta;
        }

        if (iter_index + 1 < num_eigenvectors and beta > 1e-10) {
            writeNormalizedColumn(basis, num_nodes, num_eigenvectors, iter_index + 1, work, beta);
        }

        prev_beta = beta;
    }

    return .{
        .tridiagonal = tridiagonal,
        .basis = basis,
        .num_nodes = num_nodes,
        .num_eigenvectors = num_eigenvectors,
    };
}

/// Computes the largest eigenvalues and eigenvectors using Lanczos + Jacobi diagonalization.
/// Time complexity: O(k * (V + E + k) + k³), Space complexity: O(V * k + k²)
pub fn findLanczosEigenvectors(
    allocator: Allocator,
    graph: []const []const usize,
    num_eigenvectors: usize,
) !LanczosResult {
    try validateAdjacencyList(graph);
    var decomposition = try lanczosIteration(allocator, graph, num_eigenvectors);
    defer decomposition.deinit(allocator);

    var eig = try jacobiEigenDecomposition(
        allocator,
        decomposition.tridiagonal,
        decomposition.num_eigenvectors,
    );
    defer eig.deinit(allocator);

    sortEigenpairsDescending(eig.eigenvalues, eig.eigenvectors, decomposition.num_eigenvectors);

    const projected = try allocator.alloc(f64, decomposition.num_nodes * decomposition.num_eigenvectors);
    errdefer allocator.free(projected);
    @memset(projected, 0.0);

    for (0..decomposition.num_eigenvectors) |col| {
        for (0..decomposition.num_nodes) |row| {
            var sum: f64 = 0.0;
            for (0..decomposition.num_eigenvectors) |mid| {
                sum += decomposition.basis[row * decomposition.num_eigenvectors + mid] *
                    eig.eigenvectors[mid * decomposition.num_eigenvectors + col];
            }
            projected[col * decomposition.num_nodes + row] = sum;
        }
    }

    return .{
        .eigenvalues = try allocator.dupe(f64, eig.eigenvalues),
        .eigenvectors = projected,
        .num_nodes = decomposition.num_nodes,
        .num_eigenvectors = decomposition.num_eigenvectors,
    };
}

const JacobiResult = struct {
    eigenvalues: []f64,
    eigenvectors: []f64,
    size: usize,

    fn deinit(self: JacobiResult, allocator: Allocator) void {
        allocator.free(self.eigenvalues);
        allocator.free(self.eigenvectors);
    }
};

fn jacobiEigenDecomposition(
    allocator: Allocator,
    matrix: []const f64,
    size: usize,
) !JacobiResult {
    const a = try allocator.dupe(f64, matrix);
    errdefer allocator.free(a);
    const v = try allocator.alloc(f64, size * size);
    errdefer allocator.free(v);
    @memset(v, 0.0);
    for (0..size) |i| {
        v[i * size + i] = 1.0;
    }

    const max_iterations = 50 * size * size;
    var iteration: usize = 0;
    while (iteration < max_iterations) : (iteration += 1) {
        const pivot = largestOffDiagonal(a, size);
        if (pivot.value < 1e-12) break;

        const p = pivot.row;
        const q = pivot.col;
        const app = a[p * size + p];
        const aqq = a[q * size + q];
        const apq = a[p * size + q];

        const tau = (aqq - app) / (2.0 * apq);
        const t = if (tau >= 0)
            1.0 / (tau + @sqrt(1.0 + tau * tau))
        else
            -1.0 / (-tau + @sqrt(1.0 + tau * tau));
        const c = 1.0 / @sqrt(1.0 + t * t);
        const s = t * c;

        for (0..size) |k| {
            if (k == p or k == q) continue;
            const aik = a[k * size + p];
            const akq = a[k * size + q];
            a[k * size + p] = c * aik - s * akq;
            a[p * size + k] = a[k * size + p];
            a[k * size + q] = s * aik + c * akq;
            a[q * size + k] = a[k * size + q];
        }

        const new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        const new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * size + p] = new_app;
        a[q * size + q] = new_aqq;
        a[p * size + q] = 0.0;
        a[q * size + p] = 0.0;

        for (0..size) |k| {
            const vip = v[k * size + p];
            const viq = v[k * size + q];
            v[k * size + p] = c * vip - s * viq;
            v[k * size + q] = s * vip + c * viq;
        }
    }

    const eigenvalues = try allocator.alloc(f64, size);
    for (0..size) |i| {
        eigenvalues[i] = a[i * size + i];
    }
    allocator.free(a);

    return .{
        .eigenvalues = eigenvalues,
        .eigenvectors = v,
        .size = size,
    };
}

const OffDiagonalEntry = struct {
    row: usize,
    col: usize,
    value: f64,
};

fn largestOffDiagonal(matrix: []const f64, size: usize) OffDiagonalEntry {
    var best = OffDiagonalEntry{ .row = 0, .col = 0, .value = 0.0 };
    for (0..size) |row| {
        for (row + 1..size) |col| {
            const value = @abs(matrix[row * size + col]);
            if (value > best.value) {
                best = .{ .row = row, .col = col, .value = value };
            }
        }
    }
    return best;
}

fn sortEigenpairsDescending(eigenvalues: []f64, eigenvectors: []f64, size: usize) void {
    var i: usize = 0;
    while (i < size) : (i += 1) {
        var best = i;
        var j = i + 1;
        while (j < size) : (j += 1) {
            if (eigenvalues[j] > eigenvalues[best]) best = j;
        }
        if (best == i) continue;
        std.mem.swap(f64, &eigenvalues[i], &eigenvalues[best]);
        for (0..size) |row| {
            std.mem.swap(
                f64,
                &eigenvectors[row * size + i],
                &eigenvectors[row * size + best],
            );
        }
    }
}

fn readColumn(matrix: []const f64, rows: usize, cols: usize, col: usize, out: []f64) void {
    for (0..rows) |row| {
        out[row] = matrix[row * cols + col];
    }
}

fn writeNormalizedColumn(
    matrix: []f64,
    rows: usize,
    cols: usize,
    col: usize,
    values: []const f64,
    norm: f64,
) void {
    for (0..rows) |row| {
        matrix[row * cols + col] = values[row] / norm;
    }
}

fn dotColumn(matrix: []const f64, rows: usize, cols: usize, col: usize, vec: []const f64) f64 {
    var sum: f64 = 0.0;
    for (0..rows) |row| {
        sum += matrix[row * cols + col] * vec[row];
    }
    return sum;
}

fn axpyColumnInPlace(target: []f64, matrix: []const f64, rows: usize, cols: usize, col: usize, scale: f64) void {
    for (0..rows) |row| {
        target[row] += scale * matrix[row * cols + col];
    }
}

fn dot(a: []const f64, b: []const f64) f64 {
    var sum: f64 = 0.0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

fn vectorNorm(vector: []const f64) f64 {
    return @sqrt(dot(vector, vector));
}

fn axpyInPlace(target: []f64, basis: []const f64, scale: f64) void {
    for (target, basis) |*value, basis_value| {
        value.* += scale * basis_value;
    }
}

fn residualNorm(
    allocator: Allocator,
    graph: []const []const usize,
    eigenvector: []const f64,
    eigenvalue: f64,
) !f64 {
    const product = try multiplyMatrixVector(allocator, graph, eigenvector);
    defer allocator.free(product);
    for (product, eigenvector) |*value, v| {
        value.* -= eigenvalue * v;
    }
    return vectorNorm(product);
}

test "lanczos eigenvectors: validate adjacency list rejects invalid neighbor" {
    const graph = [_][]const usize{
        &[_]usize{1},
        &[_]usize{2},
    };

    try testing.expectError(error.InvalidNeighbor, validateAdjacencyList(&graph));
}

test "lanczos eigenvectors: multiply matrix vector matches python samples" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
    };

    const product_all = try multiplyMatrixVector(alloc, &graph, &[_]f64{ 1, 1, 1 });
    defer alloc.free(product_all);
    try testing.expectEqualSlices(f64, &[_]f64{ 2, 2, 2 }, product_all);

    const product_single = try multiplyMatrixVector(alloc, &graph, &[_]f64{ 0, 1, 0 });
    defer alloc.free(product_single);
    try testing.expectEqualSlices(f64, &[_]f64{ 1, 0, 1 }, product_single);
}

test "lanczos eigenvectors: iteration shapes and tridiagonal symmetry" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
    };

    var decomposition = try lanczosIteration(alloc, &graph, 2);
    defer decomposition.deinit(alloc);

    try testing.expectEqual(@as(usize, 4), decomposition.tridiagonal.len);
    try testing.expectEqual(@as(usize, 6), decomposition.basis.len);
    try testing.expectApproxEqAbs(
        decomposition.tridiagonal[1],
        decomposition.tridiagonal[2],
        1e-9,
    );
}

test "lanczos eigenvectors: triangle graph eigenvalues and residuals" {
    const alloc = testing.allocator;
    const graph = [_][]const usize{
        &[_]usize{ 1, 2 },
        &[_]usize{ 0, 2 },
        &[_]usize{ 0, 1 },
    };

    var result = try findLanczosEigenvectors(alloc, &graph, 2);
    defer result.deinit(alloc);

    try testing.expectApproxEqAbs(@as(f64, 2.0), result.eigenvalues[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, -1.0), result.eigenvalues[1], 1e-5);

    const first_vec = result.eigenvectors[0..result.num_nodes];
    const second_vec = result.eigenvectors[result.num_nodes .. 2 * result.num_nodes];
    try testing.expect((try residualNorm(alloc, &graph, first_vec, result.eigenvalues[0])) < 1e-5);
    try testing.expect((try residualNorm(alloc, &graph, second_vec, result.eigenvalues[1])) < 1e-4);
}

test "lanczos eigenvectors: single isolated node and extreme star graph" {
    const alloc = testing.allocator;
    const isolated = [_][]const usize{
        &[_]usize{},
    };

    var isolated_result = try findLanczosEigenvectors(alloc, &isolated, 1);
    defer isolated_result.deinit(alloc);
    try testing.expectApproxEqAbs(@as(f64, 0.0), isolated_result.eigenvalues[0], 1e-9);
    try testing.expectEqual(@as(usize, 1), isolated_result.eigenvectors.len);

    const star = [_][]const usize{
        &[_]usize{ 1, 2, 3, 4, 5, 6, 7 },
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{0},
        &[_]usize{0},
    };

    var star_result = try findLanczosEigenvectors(alloc, &star, 3);
    defer star_result.deinit(alloc);
    try testing.expect(star_result.eigenvalues[0] >= star_result.eigenvalues[1]);
    try testing.expect(star_result.eigenvalues[1] >= star_result.eigenvalues[2]);

    const principal = star_result.eigenvectors[0..star_result.num_nodes];
    try testing.expect((try residualNorm(alloc, &star, principal, star_result.eigenvalues[0])) < 1e-4);
}
