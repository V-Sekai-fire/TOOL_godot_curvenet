// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_VEC3_H
#define CURVENET_VEC3_H

#include <cmath>

namespace curvenet {

struct Vec3 {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;

	Vec3() = default;
	Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

	Vec3 operator+(const Vec3 &o) const { return { x + o.x, y + o.y, z + o.z }; }
	Vec3 operator-(const Vec3 &o) const { return { x - o.x, y - o.y, z - o.z }; }
	Vec3 operator*(double s) const { return { x * s, y * s, z * s }; }
	Vec3 operator/(double s) const { return { x / s, y / s, z / s }; }
	Vec3 &operator+=(const Vec3 &o) { x += o.x; y += o.y; z += o.z; return *this; }

	double dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }
	double length() const { return std::sqrt(x * x + y * y + z * z); }
};

inline Vec3 operator*(double s, const Vec3 &v) { return v * s; }

} // namespace curvenet

#endif
