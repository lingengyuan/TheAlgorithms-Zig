//! Area - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/area.py

const std = @import("std");
const testing = std.testing;

pub const AreaError = error{
    NegativeValue,
    InvalidPolygonSides,
    InvalidTriangle,
    InvalidTorus,
};

fn ensureNonNegative(values: []const f64) AreaError!void {
    for (values) |value| {
        if (value < 0) return error.NegativeValue;
    }
}

/// Time complexity: O(1), Space complexity: O(1)
pub fn surfaceAreaCube(side_length: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{side_length});
    return 6.0 * side_length * side_length;
}

pub fn surfaceAreaCuboid(length: f64, breadth: f64, height: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ length, breadth, height });
    return 2.0 * ((length * breadth) + (breadth * height) + (length * height));
}

pub fn surfaceAreaSphere(radius: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{radius});
    return 4.0 * std.math.pi * radius * radius;
}

pub fn surfaceAreaHemisphere(radius: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{radius});
    return 3.0 * std.math.pi * radius * radius;
}

pub fn surfaceAreaCone(radius: f64, height: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ radius, height });
    return std.math.pi * radius * (radius + @sqrt(height * height + radius * radius));
}

pub fn surfaceAreaConicalFrustum(radius_1: f64, radius_2: f64, height: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ radius_1, radius_2, height });
    const slant_height = @sqrt(height * height + (radius_1 - radius_2) * (radius_1 - radius_2));
    return std.math.pi * ((slant_height * (radius_1 + radius_2)) + radius_1 * radius_1 + radius_2 * radius_2);
}

pub fn surfaceAreaCylinder(radius: f64, height: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ radius, height });
    return 2.0 * std.math.pi * radius * (height + radius);
}

pub fn surfaceAreaTorus(torus_radius: f64, tube_radius: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ torus_radius, tube_radius });
    if (torus_radius < tube_radius) return error.InvalidTorus;
    return 4.0 * std.math.pi * std.math.pi * torus_radius * tube_radius;
}

pub fn areaRectangle(length: f64, width: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ length, width });
    return length * width;
}

pub fn areaSquare(side_length: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{side_length});
    return side_length * side_length;
}

pub fn areaTriangle(base: f64, height: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ base, height });
    return (base * height) / 2.0;
}

pub fn areaTriangleThreeSides(side1: f64, side2: f64, side3: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ side1, side2, side3 });
    if (side1 + side2 < side3 or side1 + side3 < side2 or side2 + side3 < side1) {
        return error.InvalidTriangle;
    }
    const semi = (side1 + side2 + side3) / 2.0;
    return @sqrt(semi * (semi - side1) * (semi - side2) * (semi - side3));
}

pub fn areaParallelogram(base: f64, height: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ base, height });
    return base * height;
}

pub fn areaTrapezium(base1: f64, base2: f64, height: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ base1, base2, height });
    return 0.5 * (base1 + base2) * height;
}

pub fn areaCircle(radius: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{radius});
    return std.math.pi * radius * radius;
}

pub fn areaEllipse(radius_x: f64, radius_y: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ radius_x, radius_y });
    return std.math.pi * radius_x * radius_y;
}

pub fn areaRhombus(diagonal_1: f64, diagonal_2: f64) AreaError!f64 {
    try ensureNonNegative(&[_]f64{ diagonal_1, diagonal_2 });
    return 0.5 * diagonal_1 * diagonal_2;
}

pub fn areaRegPolygon(sides: i64, length: f64) AreaError!f64 {
    if (sides < 3) return error.InvalidPolygonSides;
    try ensureNonNegative(&[_]f64{length});
    return (@as(f64, @floatFromInt(sides)) * length * length) /
        (4.0 * @tan(std.math.pi / @as(f64, @floatFromInt(sides))));
}

test "area: surface formulas reference samples" {
    try testing.expectEqual(@as(f64, 6.0), try surfaceAreaCube(1));
    try testing.expectApproxEqAbs(@as(f64, 38.56), try surfaceAreaCuboid(1.6, 2.6, 3.6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 314.1592653589793), try surfaceAreaSphere(5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 235.61944901923448), try surfaceAreaHemisphere(5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 301.59289474462014), try surfaceAreaCone(6, 8), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 78.57907060751548), try surfaceAreaConicalFrustum(1.6, 2.6, 3.6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 527.7875658030853), try surfaceAreaCylinder(6, 8), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 39.47841760435743), try surfaceAreaTorus(1, 1), 1e-12);
}

test "area: planar formulas reference samples" {
    try testing.expectEqual(@as(f64, 200.0), try areaRectangle(10, 20));
    try testing.expectApproxEqAbs(@as(f64, 2.5600000000000005), try areaSquare(1.6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 50.0), try areaTriangle(10, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 30.0), try areaTriangleThreeSides(5, 12, 13), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 200.0), try areaParallelogram(10, 20), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 450.0), try areaTrapezium(10, 20, 30), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1256.6370614359173), try areaCircle(20), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 13.06902543893354), try areaEllipse(1.6, 2.6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 100.0), try areaRhombus(10, 20), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 43.301270189221945), try areaRegPolygon(3, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 100.00000000000001), try areaRegPolygon(4, 10), 1e-12);
}

test "area: invalid and extreme cases" {
    try testing.expectError(error.NegativeValue, surfaceAreaCube(-1));
    try testing.expectError(error.NegativeValue, surfaceAreaCone(1, -2));
    try testing.expectError(error.InvalidTorus, surfaceAreaTorus(3, 4));
    try testing.expectError(error.NegativeValue, areaRectangle(-1, 2));
    try testing.expectError(error.InvalidTriangle, areaTriangleThreeSides(2, 4, 7));
    try testing.expectError(error.InvalidPolygonSides, areaRegPolygon(0, 0));
    try testing.expectError(error.NegativeValue, areaRegPolygon(5, -2));

    try testing.expectEqual(@as(f64, 0.0), try surfaceAreaSphere(0));
    try testing.expectEqual(@as(f64, 0.0), try areaCircle(0));
    try testing.expectEqual(@as(f64, 0.0), try areaTriangleThreeSides(0, 0, 0));
}

