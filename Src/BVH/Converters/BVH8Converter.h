#pragma once
#include "BVHConverter.h"
#include "Config.h"

struct BVH8Converter final : BVHConverter {
	      BVH8 & bvh8;
	const BVH2 & bvh2;

	//jichan add
	Array<uint64_t> counter;
	bool counterCollapse = false;
	// Gu et al. CBTC-RD selection: counter ratio after threshold t, SA ratio
	// before threshold t. This remains distinct from counterCollapse.
	bool guRDTC = false;
	uint64_t guRDTCThreshold = 32ull;
	float rootCounterSA;
	//jichan add end

	BVH8Converter(BVH8 & bvh8, const BVH2 & bvh2) : bvh8(bvh8), bvh2(bvh2) {
		bvh8.indices.reserve(bvh2.indices.size());
		bvh8.nodes  .reserve(bvh2.nodes  .size());
	}

	void convert() override;
	// Greedy surface-area collapse; preserves the original binary leaves.
	// Call explicitly instead of convert() to select this alternative.
	void convert_greedy();
	void convert_gu_rdtc();

private:
	struct Decision {
		enum struct Type : char {
			LEAF,
			INTERNAL,
			DISTRIBUTE
		} type;

		char distribute_left;
		char distribute_right;

		float cost;
	};

	Array<Decision> decisions;

	int calculate_cost(int node_index, const Array<BVHNode2> & nodes);

	void get_children  (int node_index, const Array<BVHNode2> & nodes, int children[8], int & child_count, int i);
	void get_children_greedy(int node_index, const Array<BVHNode2> & nodes, int children[8], int & child_count);
	void order_children(int node_index, const Array<BVHNode2> & nodes, int children[8], int   child_count);

	int count_primitives(int node_index, const Array<BVHNode2> & nodes, const Array<int> & indices_sbvh);

	void collapse(const Array<BVHNode2> & nodes_bvh, const Array<int> & indices_bvh, int node_index_bvh8, int node_index_bvh2);
	void collapse_greedy(const Array<BVHNode2> & nodes_bvh, const Array<int> & indices_bvh, int node_index_bvh8, int node_index_bvh2);
};
