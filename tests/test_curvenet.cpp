// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#include "test_helpers.h"

#include "curvenet/curvenet.h"

using curvenet::BoundaryCurve;
using curvenet::BoundFace;
using curvenet::BoundVertex;
using curvenet::CurveNet;
using curvenet::Vec3;
using curvenet::deform;

static rc::Gen<Vec3> gen_vec() {
	auto coord = rc::gen::map(rc::gen::inRange<int>(-100, 101),
			[](int v) { return static_cast<double>(v) * 0.1; });
	return rc::gen::construct<Vec3>(coord, coord, coord);
}

// 1-quad CurveNet with corner-consistent boundaries.
static rc::Gen<CurveNet> gen_quad_net() {
	return rc::gen::map(
			rc::gen::tuple(gen_vec(), gen_vec(), gen_vec(), gen_vec(),
					gen_vec(), gen_vec(),
					gen_vec(), gen_vec(),
					gen_vec(), gen_vec(),
					gen_vec(), gen_vec()),
			[](std::tuple<Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3, Vec3> t) {
				const Vec3 P00 = std::get<0>(t);
				const Vec3 P10 = std::get<1>(t);
				const Vec3 P11 = std::get<2>(t);
				const Vec3 P01 = std::get<3>(t);
				BoundFace f;
				f.boundaries.push_back(BoundaryCurve{ P00, std::get<4>(t), std::get<5>(t), P10 });
				f.boundaries.push_back(BoundaryCurve{ P10, std::get<6>(t), std::get<7>(t), P11 });
				f.boundaries.push_back(BoundaryCurve{ P11, std::get<8>(t), std::get<9>(t), P01 });
				f.boundaries.push_back(BoundaryCurve{ P01, std::get<10>(t), std::get<11>(t), P00 });
				CurveNet net;
				net.faces.push_back(f);
				return net;
			});
}

int main() {
	bool ok = true;

	ok &= rc::check("deform produces one output per binding", [] {
		CurveNet net = *gen_quad_net();
		int n = *rc::gen::inRange<int>(0, 16);
		std::vector<BoundVertex> bindings(n);
		for (auto &b : bindings) {
			b.face_index = 0;
			b.s = *cnt::unit_t();
			b.t = *cnt::unit_t();
		}
		auto out = deform(net, bindings);
		RC_ASSERT(out.size() == bindings.size());
	});

	ok &= rc::check("translating all curves by d translates each deformed vertex by d", [](Vec3 d) {
		CurveNet net = *gen_quad_net();
		CurveNet net2 = net;
		for (auto &face : net2.faces) {
			for (auto &b : face.boundaries) {
				b.c0 = b.c0 + d;
				b.c1 = b.c1 + d;
				b.c2 = b.c2 + d;
				b.c3 = b.c3 + d;
			}
		}
		std::vector<BoundVertex> bindings;
		for (int i = 0; i < 8; ++i) {
			BoundVertex b;
			b.face_index = 0;
			b.s = *cnt::unit_t();
			b.t = *cnt::unit_t();
			bindings.push_back(b);
		}
		auto a = deform(net, bindings);
		auto bb = deform(net2, bindings);
		for (std::size_t i = 0; i < a.size(); ++i) {
			RC_ASSERT(cnt::approx_eq(a[i] + d, bb[i], 1e-9));
		}
	});

	ok &= rc::check("binding at corner (0,0) recovers boundaries[0].c0", [] {
		CurveNet net = *gen_quad_net();
		std::vector<BoundVertex> b = { { 0, 0.0, 0.0 } };
		auto out = deform(net, b);
		RC_ASSERT(cnt::approx_eq(out[0], net.faces[0].boundaries[0].c0, 1e-9));
	});

	ok &= rc::check("binding at corner (1,0) recovers boundaries[0].c3", [] {
		CurveNet net = *gen_quad_net();
		std::vector<BoundVertex> b = { { 0, 1.0, 0.0 } };
		auto out = deform(net, b);
		RC_ASSERT(cnt::approx_eq(out[0], net.faces[0].boundaries[0].c3, 1e-9));
	});

	ok &= rc::check("bottom-edge binding (s, 0) lies on boundaries[0]", [] {
		CurveNet net = *gen_quad_net();
		double s = *cnt::unit_t();
		std::vector<BoundVertex> b = { { 0, s, 0.0 } };
		auto out = deform(net, b);
		Vec3 expected = net.faces[0].boundaries[0].evaluate(s);
		RC_ASSERT(cnt::approx_eq(out[0], expected, 1e-9));
	});

	return ok ? 0 : 1;
}
