#include "BVH8Converter.h"

#include <string.h>

#include "Core/IO.h"

void BVH8Converter::convert() {
	bvh8.indices.clear();
	bvh8.indices.reserve(bvh2.indices.size());

	bvh8.nodes.clear();
	bvh8.nodes.emplace_back(); // Root

	decisions.resize(bvh2.nodes.size() * 7);

	// Fill cost table using dynamic programming (bottom up)
	if (counterCollapse) {
		rootCounterSA = bvh2.nodes[0].aabb.surface_area() / (float)counter[0];
	}
	calculate_cost(0, bvh2.nodes);

	// Collapse SBVH into 8-way tree (top down)
	collapse(bvh2.nodes, bvh2.indices, 0, 0);
	ASSERT(bvh8.indices.size() == bvh2.indices.size());
}

void BVH8Converter::convert_greedy() {
	if (bvh2.nodes.size() == 0) {
		IO::print("ERROR: BVH8 greedy converter requires a non-empty BVH2!\n"_sv);
		IO::exit(1);
	}
	for (size_t i = 0; i < bvh2.nodes.size(); i++) {
		if (bvh2.nodes[i].is_leaf() && bvh2.nodes[i].count != 1) {
			IO::print("ERROR: BVH8 greedy converter expects BVH leaf nodes containing only 1 primitive!\n"_sv);
			IO::exit(1);
		}
	}

	bvh8.indices.clear();
	bvh8.indices.reserve(bvh2.indices.size());
	bvh8.nodes.clear();
	bvh8.nodes.emplace_back(); // Root

	collapse_greedy(bvh2.nodes, bvh2.indices, 0, 0);
	ASSERT(bvh8.indices.size() == bvh2.indices.size());
}

void BVH8Converter::convert_gu_rdtc() {
	ASSERT(guRDTC);
	ASSERT(counter.size() == bvh2.nodes.size());
	convert_greedy();
}

int BVH8Converter::calculate_cost(int node_index, const Array<BVHNode2> & nodes) {
	const BVHNode2 & node = nodes[node_index];

	int num_primitives;

	if (node.is_leaf()) {
		num_primitives = node.count;
		if (num_primitives != 1) {
			IO::print("ERROR: BVH8 Builder expects BVH with leaf Nodes containing only 1 primitive!\n"_sv);
			IO::exit(1);
		}

		// SAH cost
		float cost_leaf;
		if (counterCollapse) {
			cost_leaf = node.aabb.surface_area() + counter[node_index] * rootCounterSA;
		}
		else {
			cost_leaf = node.aabb.surface_area() * float(num_primitives);
		}

		for (int i = 0; i < 7; i++) {
			decisions[node_index * 7 + i].type = Decision::Type::LEAF;
			decisions[node_index * 7 + i].cost = cost_leaf;
		}
	} else {
		num_primitives =
			calculate_cost(node.left,     nodes) +
			calculate_cost(node.left + 1, nodes);

		// Separate case: i=0 (i=1 in the paper)
		{
			float cost_leaf;
			if (counterCollapse) {
				cost_leaf = num_primitives <= 3 ? node.aabb.surface_area() + counter[node_index] * rootCounterSA : INFINITY;
			}
			else {
				cost_leaf = num_primitives <= 3 ? float(num_primitives) * node.aabb.surface_area() : INFINITY;
			}

			float cost_distribute = INFINITY;

			char distribute_left  = INVALID;
			char distribute_right = INVALID;

			for (int k = 0; k < 7; k++) {
				float c =
					decisions[(node.left)     * 7 +     k].cost +
					decisions[(node.left + 1) * 7 + 6 - k].cost;

				if (c < cost_distribute) {
					cost_distribute = c;

					distribute_left  =     k;
					distribute_right = 6 - k;
				}
			}

			float cost_internal = cost_distribute + node.aabb.surface_area();
			if (counterCollapse) {
				cost_internal += counter[node_index] * rootCounterSA;
			}

			if (cost_leaf < cost_internal) {
				decisions[node_index * 7].type = Decision::Type::LEAF;
				decisions[node_index * 7].cost = cost_leaf;
			} else {
				decisions[node_index * 7].type = Decision::Type::INTERNAL;
				decisions[node_index * 7].cost = cost_internal;
			}

			decisions[node_index * 7].distribute_left  = distribute_left;
			decisions[node_index * 7].distribute_right = distribute_right;
		}

		// In the paper i=2..7
		for (int i = 1; i < 7; i++) {
			float cost_distribute = decisions[node_index * 7 + i - 1].cost;

			char distribute_left  = INVALID;
			char distribute_right = INVALID;

			for (int k = 0; k < i; k++) {
				float c =
					decisions[(node.left)     * 7 +     k    ].cost +
					decisions[(node.left + 1) * 7 + i - k - 1].cost;

				if (c < cost_distribute) {
					cost_distribute = c;

					distribute_left  =     k;
					distribute_right = i - k - 1;
				}
			}

			decisions[node_index * 7 + i].cost = cost_distribute;

			if (distribute_left != INVALID) {
				decisions[node_index * 7 + i].type = Decision::Type::DISTRIBUTE;
				decisions[node_index * 7 + i].distribute_left  = distribute_left;
				decisions[node_index * 7 + i].distribute_right = distribute_right;
			} else {
				decisions[node_index * 7 + i] = decisions[node_index * 7 + i - 1];
			}
		}
	}

	return num_primitives;
}

void BVH8Converter::get_children(int node_index, const Array<BVHNode2> & nodes, int children[8], int & child_count, int i) {
	const BVHNode2 & node = nodes[node_index];

	if (node.is_leaf()) {
		children[child_count++] = node_index;
		return;
	}

	char distribute_left  = decisions[node_index * 7 + i].distribute_left;
	char distribute_right = decisions[node_index * 7 + i].distribute_right;

	ASSERT(distribute_left  >= 0 && distribute_left  < 7);
	ASSERT(distribute_right >= 0 && distribute_right < 7);

	// Recurse on left child if it needs to distribute
	if (decisions[node.left * 7 + distribute_left].type == Decision::Type::DISTRIBUTE) {
		get_children(node.left, nodes, children, child_count, distribute_left);
	} else {
		children[child_count++] = node.left;
	}

	// Recurse on right child if it needs to distribute
	if (decisions[(node.left + 1) * 7 + distribute_right].type == Decision::Type::DISTRIBUTE) {
		get_children(node.left + 1, nodes, children, child_count, distribute_right);
	} else {
		children[child_count++] = node.left + 1;
	}
}

void BVH8Converter::get_children_greedy(int node_index, const Array<BVHNode2> & nodes, int children[8], int & child_count) {
	children[0] = node_index;
	child_count = 1;

	while (child_count < 8) {
		int largest_index = INVALID;
		float largest_area = -1.0f;
		for (int i = 0; i < child_count; i++) {
			const BVHNode2 & child = nodes[children[i]];
			if (child.is_leaf()) continue;
			if (guRDTC) {
				// CBTC-RD (Gu et al.): use the measured Pass-test probability only
				// after it has at least t observations. Sparse samples fall back to
				// the same normalized surface-area estimate as CBTC-SA/Wald.
				const uint64_t child_counter = counter[children[i]];
				const uint64_t parent_counter = counter[node_index];
				const float parent_area = nodes[node_index].aabb.surface_area();
				const float heuristic = child_counter >= guRDTCThreshold && parent_counter > 0
					? static_cast<float>(child_counter) / static_cast<float>(parent_counter)
					: (parent_area > 0.0f ? child.aabb.surface_area() / parent_area : 0.0f);

				if (largest_index == INVALID || heuristic > largest_area) {
					largest_area = heuristic;
					largest_index = i;
				}
			}
			else if (counterCollapse) {
				const float parent_area = nodes[node_index].aabb.surface_area();
				const float surface_share = parent_area > 0.0f
					? child.aabb.surface_area() / parent_area
					: 0.0f;
				// Counter values are uint64_t. Cast before division: integer division
				// loses the intended sibling ratio, and an unvisited subtree makes
				// the original 0 / 0 expression terminate the CPU process.
				const uint64_t parent_counter = counter[node_index];
				const float traversal_share = parent_counter > 0
					? static_cast<float>(counter[children[i]]) / static_cast<float>(parent_counter)
					: 0.0f;
				const float heuristic = surface_share + traversal_share;
				if (largest_index == INVALID || heuristic > largest_area) {
					largest_area = heuristic;
					largest_index = i;
				}
			}
			else{
				const float area = child.aabb.surface_area();
				if (largest_index == INVALID || area > largest_area) {
					largest_area = area;
					largest_index = i;
				}
			}
		}
		if (largest_index == INVALID) break;

		const int left = nodes[children[largest_index]].left;
		children[largest_index] = left;
		children[child_count++] = left + 1;
	}
}

void BVH8Converter::order_children(int node_index, const Array<BVHNode2> & nodes, int children[8], int child_count) {
	Vector3 p = nodes[node_index].aabb.get_center();

	float cost[8][8] = { };

	// Fill cost table
	for (int c = 0; c < child_count; c++) {
		for (int s = 0; s < 8; s++) {
			Vector3 direction(
				(s & 0b100) ? -1.0f : +1.0f,
				(s & 0b010) ? -1.0f : +1.0f,
				(s & 0b001) ? -1.0f : +1.0f
			);

			cost[c][s] = Vector3::dot(nodes[children[c]].aabb.get_center() - p, direction);
		}
	}

	int   assignment[8] = { INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID };
	bool slot_filled[8] = { };

	// Greedy child ordering, the paper mentions this as an alternative
	// that works about as well as the auction algorithm in practice
	while (true) {
		float min_cost = INFINITY;

		int min_slot  = INVALID;
		int min_index = INVALID;

		// Find cheapest unfilled slot of any unassigned child
		for (int c = 0; c < child_count; c++) {
			if (assignment[c] == INVALID) {
				for (int s = 0; s < 8; s++) {
					if (!slot_filled[s] && cost[c][s] < min_cost) {
						min_cost = cost[c][s];

						min_slot  = s;
						min_index = c;
					}
				}
			}
		}

		if (min_slot == INVALID) break;

		slot_filled[min_slot]  = true;
		assignment [min_index] = min_slot;
	}

	// Permute children array according to assignment
	int children_copy[8] = { };
	for (int i = 0; i < 8; i++) {
		children_copy[i] = children[i];
		children[i] = INVALID;
	}
	for (int i = 0; i < child_count; i++) {
		ASSERT(assignment   [i] != INVALID);
		ASSERT(children_copy[i] != INVALID);
		children[assignment[i]] = children_copy[i];
	}
}

// Recursively count triangles in subtree of the given Node
// Simultaneously fills the indices buffer of the BVH8
int BVH8Converter::count_primitives(int node_index, const Array<BVHNode2> & nodes, const Array<int> & indices) {
	const BVHNode2 & node = nodes[node_index];

	if (node.is_leaf()) {
		ASSERT(node.count == 1);

		for (unsigned i = 0; i < node.count; i++) {
			bvh8.indices.push_back(indices[node.first + i]);
		}

		return node.count;
	}

	return
		count_primitives(node.left,     nodes, indices) +
		count_primitives(node.left + 1, nodes, indices);
}

void BVH8Converter::collapse(const Array<BVHNode2> & nodes_bvh, const Array<int> & indices_bvh, int node_index_bvh8, int node_index_bvh2) {
	BVHNode8 & node = bvh8.nodes[node_index_bvh8];
	const AABB & aabb = nodes_bvh[node_index_bvh2].aabb;

	node.p = aabb.min;

	constexpr int Nq = 8;
	constexpr float denom = 1.0f / float((1 << Nq) - 1);

	Vector3 e(
		exp2f(ceilf(log2f((aabb.max.x - aabb.min.x) * denom))),
		exp2f(ceilf(log2f((aabb.max.y - aabb.min.y) * denom))),
		exp2f(ceilf(log2f((aabb.max.z - aabb.min.z) * denom)))
	);

	Vector3 one_over_e(1.0f / e.x, 1.0f / e.y, 1.0f / e.z);

	unsigned u_ex = Util::bit_cast<unsigned>(e.x);
	unsigned u_ey = Util::bit_cast<unsigned>(e.y);
	unsigned u_ez = Util::bit_cast<unsigned>(e.z);

	// Only the exponent bits can be non-zero
	ASSERT((u_ex & 0b10000000011111111111111111111111) == 0);
	ASSERT((u_ey & 0b10000000011111111111111111111111) == 0);
	ASSERT((u_ez & 0b10000000011111111111111111111111) == 0);

	// Store only 8 bit exponent
	node.e[0] = u_ex >> 23;
	node.e[1] = u_ey >> 23;
	node.e[2] = u_ez >> 23;

	int child_count = 0;
	int children[8] = { INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID };
	get_children(node_index_bvh2, nodes_bvh, children, child_count, 0);
	ASSERT(child_count <= 8);

	order_children(node_index_bvh2, nodes_bvh, children, child_count);

	node.imask = 0;

	node.base_index_triangle = unsigned(bvh8.indices.size());
	node.base_index_child    = unsigned(bvh8.nodes  .size());

	int num_internal_nodes = 0;
	int num_triangles      = 0;

	for (int i = 0; i < 8; i++) {
		int child_index = children[i];
		if (child_index == INVALID) continue; // Empty slot

		const AABB & child_aabb = nodes_bvh[child_index].aabb;

		node.quantized_min_x[i] = byte(floorf((child_aabb.min.x - node.p.x) * one_over_e.x));
		node.quantized_min_y[i] = byte(floorf((child_aabb.min.y - node.p.y) * one_over_e.y));
		node.quantized_min_z[i] = byte(floorf((child_aabb.min.z - node.p.z) * one_over_e.z));

		node.quantized_max_x[i] = byte(ceilf((child_aabb.max.x - node.p.x) * one_over_e.x));
		node.quantized_max_y[i] = byte(ceilf((child_aabb.max.y - node.p.y) * one_over_e.y));
		node.quantized_max_z[i] = byte(ceilf((child_aabb.max.z - node.p.z) * one_over_e.z));

		switch (decisions[child_index * 7].type) {
			case Decision::Type::LEAF: {
				int triangle_count = count_primitives(child_index, nodes_bvh, indices_bvh);
				ASSERT(triangle_count > 0 && triangle_count <= 3);

				// Three highest bits contain unary representation of triangle count
				for (int j = 0; j < triangle_count; j++) {
					node.meta[i] |= (1 << (j + 5));
				}

				node.meta[i] |= num_triangles;

				num_triangles += triangle_count;
				ASSERT(num_triangles <= 24);
				break;
			}
			case Decision::Type::INTERNAL: {
				node.meta[i] = (i + 24) | 0b00100000;

				node.imask |= (1 << i);
				num_internal_nodes++;
				break;
			}
			default: ASSERT_UNREACHABLE();
		}
	}

	for (int i = 0; i < num_internal_nodes; i++) {
		bvh8.nodes.emplace_back();
	}
	node = bvh8.nodes[node_index_bvh8]; // NOTE: 'node' may have been invalidated by emplace_back on previous line

	ASSERT(node.base_index_child    + num_internal_nodes == bvh8.nodes  .size());
	ASSERT(node.base_index_triangle + num_triangles      == bvh8.indices.size());

	// Recurse on Internal Nodes
	int offset = 0;
	for (int i = 0; i < 8; i++) {
		int child_index = children[i];
		if (child_index == INVALID) continue;

		if (node.imask & (1 << i)) {
			collapse(nodes_bvh, indices_bvh, node.base_index_child + offset++, child_index);
		}
	}
}

void BVH8Converter::collapse_greedy(const Array<BVHNode2> & nodes_bvh, const Array<int> & indices_bvh, int node_index_bvh8, int node_index_bvh2) {
	BVHNode8 & node = bvh8.nodes[node_index_bvh8];
	const AABB & aabb = nodes_bvh[node_index_bvh2].aabb;

	node.p = aabb.min;

	constexpr int Nq = 8;
	constexpr float denom = 1.0f / float((1 << Nq) - 1);
	Vector3 e(
		exp2f(ceilf(log2f((aabb.max.x - aabb.min.x) * denom))),
		exp2f(ceilf(log2f((aabb.max.y - aabb.min.y) * denom))),
		exp2f(ceilf(log2f((aabb.max.z - aabb.min.z) * denom)))
	);
	Vector3 one_over_e(1.0f / e.x, 1.0f / e.y, 1.0f / e.z);

	unsigned u_ex = Util::bit_cast<unsigned>(e.x);
	unsigned u_ey = Util::bit_cast<unsigned>(e.y);
	unsigned u_ez = Util::bit_cast<unsigned>(e.z);
	ASSERT((u_ex & 0b10000000011111111111111111111111) == 0);
	ASSERT((u_ey & 0b10000000011111111111111111111111) == 0);
	ASSERT((u_ez & 0b10000000011111111111111111111111) == 0);
	node.e[0] = u_ex >> 23;
	node.e[1] = u_ey >> 23;
	node.e[2] = u_ez >> 23;

	int child_count = 0;
	int children[8] = { INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID };
	get_children_greedy(node_index_bvh2, nodes_bvh, children, child_count);
	ASSERT(child_count <= 8);
	order_children(node_index_bvh2, nodes_bvh, children, child_count);

	node.imask = 0;
	node.base_index_triangle = unsigned(bvh8.indices.size());
	node.base_index_child    = unsigned(bvh8.nodes.size());

	int num_internal_nodes = 0;
	int num_triangles = 0;
	for (int i = 0; i < 8; i++) {
		const int child_index = children[i];
		if (child_index == INVALID) continue;

		const AABB & child_aabb = nodes_bvh[child_index].aabb;
		node.quantized_min_x[i] = byte(floorf((child_aabb.min.x - node.p.x) * one_over_e.x));
		node.quantized_min_y[i] = byte(floorf((child_aabb.min.y - node.p.y) * one_over_e.y));
		node.quantized_min_z[i] = byte(floorf((child_aabb.min.z - node.p.z) * one_over_e.z));
		node.quantized_max_x[i] = byte(ceilf((child_aabb.max.x - node.p.x) * one_over_e.x));
		node.quantized_max_y[i] = byte(ceilf((child_aabb.max.y - node.p.y) * one_over_e.y));
		node.quantized_max_z[i] = byte(ceilf((child_aabb.max.z - node.p.z) * one_over_e.z));

		if (nodes_bvh[child_index].is_leaf()) {
			const int triangle_count = count_primitives(child_index, nodes_bvh, indices_bvh);
			ASSERT(triangle_count == 1);
			node.meta[i] |= 0b00100000;
			node.meta[i] |= num_triangles++;
		} else {
			node.meta[i] = (i + 24) | 0b00100000;
			node.imask |= (1 << i);
			num_internal_nodes++;
		}
	}

	for (int i = 0; i < num_internal_nodes; i++) {
		bvh8.nodes.emplace_back();
	}
	node = bvh8.nodes[node_index_bvh8];
	ASSERT(node.base_index_child + num_internal_nodes == bvh8.nodes.size());
	ASSERT(node.base_index_triangle + num_triangles == bvh8.indices.size());

	int offset = 0;
	for (int i = 0; i < 8; i++) {
		const int child_index = children[i];
		if (child_index == INVALID) continue;
		if (node.imask & (1 << i)) {
			collapse_greedy(nodes_bvh, indices_bvh, node.base_index_child + offset++, child_index);
		}
	}
}
