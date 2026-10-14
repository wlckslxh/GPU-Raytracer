#pragma once
#include "BVH/BVH.h"

#include "Core/Array.h"
#include "Core/BitArray.h"

// 필요한거 정리하면
//1. 기존 bvh 데이터와 이걸 수정한 후의 데이터
//2. 이걸 gpu로 올려야 하는데 그 부분은 integrator에서 하는게 낫나?
//3. 실제 rebuild할 함수. rebuild시 기존 bvh8에서 하나하나 읽어들여서 처리하는게 좋아보임

constexpr uint32_t INVALID_NODE = UINT32_MAX;

enum WorkState {
	Unvisited,
	Finished
};

enum ChilKind {
	Empty,
	Internal,
	Object
};

struct Child {
	ChilKind kind = Empty;
	uint32_t index = INVALID_NODE; // internal: local BVH node, object: first local primitive
	uint8_t primitiveCount = 0; //object일 때만 활성화, 보통 1~3
	uint64_t visitCount = 0;
	AABB aabb;
};

struct WorkingNode8 {
	uint32_t childCount = 0;
	uint32_t objectCount = 0;
	Child child[8];
	uint64_t estimatedVisitCount = 0;
	byte state = Unvisited;
	uint32_t depth;
};

struct serializePendingNode {
	uint32_t workingIndex;
	uint32_t outputIndex;
};

struct BVH8RebuildMetadata {
	uint64_t old_node_count = 0;
	uint64_t new_node_count = 0;
	uint64_t promotion_count = 0;
	float promotion_heuristic_avg = NAN;
	float promotion_heuristic_min = NAN;
	float promotion_heuristic_max = NAN;
	float promotion_ray_share_avg = NAN;
	float promotion_ray_share_min = NAN;
	float promotion_ray_share_max = NAN;
	float promotion_remain_sa_avg = NAN;
	float promotion_remain_sa_min = NAN;
	float promotion_remain_sa_max = NAN;
	uint64_t merge_count = 0;
	float merge_heuristic_avg = NAN;
	float merge_heuristic_min = NAN;
	float merge_heuristic_max = NAN;
	float merge_ray_share_avg = NAN;
	float merge_ray_share_min = NAN;
	float merge_ray_share_max = NAN;
	float merge_after_sa_avg = NAN;
	float merge_after_sa_min = NAN;
	float merge_after_sa_max = NAN;
};

struct BVH8RebuildRayDistribution {
	const BVH8& oldBvh;
	const Array<uint64_t>& nodeCounter;
	const Array<uint64_t>& objectCounter;
	BVH8& newBvh;
	bool is_tlas;

	Array<WorkingNode8> workingBvh;
	Array<uint32_t> queue;
	uint32_t globalNodeOffset;
	uint32_t globalObjectOffset;
	Array<serializePendingNode> serializeQueue;
	
	//script control
	bool mergeOn = true;
	bool promotionOn = true;
	bool promotionLoop = false;
	bool mergeAABBAware = true;
	bool promotionAABBAware = true;
	float mergeThreshold = 0.25;
	float maxCounterThreshold = 0.3;

	//metadatas
	uint64_t mergeCount = 0;
	uint64_t promotionCount = 0;
	float mergeHeuristicMin = FLT_MAX;
	float mergeHeuristicMax = -FLT_MAX;
	float mergeHeuristicAvg = 0;
	float mergeRayShareMin = FLT_MAX;
	float mergeRayShareMax = -FLT_MAX;
	float mergeRayShareAvg = 0;
	float mergeAfterSAMin = FLT_MAX;
	float mergeAfterSAMax = -FLT_MAX;
	float mergeAfterSAAvg = 0;
	float promotionHeuristicMin = FLT_MAX;
	float promotionHeuristicMax = -FLT_MAX;
	float promotionHeuristicAvg = 0;
	float promotionRayShareMin = FLT_MAX;
	float promotionRayShareMax = -FLT_MAX;
	float promotionRayShareAvg = 0;
	float promotionRemainSAMin = FLT_MAX;
	float promotionRemainSAMax = -FLT_MAX;
	float promotionRemainSAAvg = 0;
	uint32_t maxDepth = 0;
	Array<uint64_t> mergePerDepth;
	Array<uint64_t> promotionPerDepth;

	BVH8RebuildRayDistribution(const BVH8& oldBvh, const Array<uint64_t>& nodeCounter, const Array<uint64_t>& objectCounter, BVH8& newBvh, uint32_t globalNodeOffset, uint32_t globalObjectOffset, bool is_tlas,
		bool merge_on = true, bool promotion_on = true, bool promotion_loop = false, bool merge_aabb_aware = true, bool promotion_aabb_aware = true,
		float merge_threshold = 0.25f, float max_counter_threshold = 0.3f)
		: oldBvh(oldBvh), nodeCounter(nodeCounter), objectCounter(objectCounter), newBvh(newBvh), is_tlas(is_tlas), globalNodeOffset(globalNodeOffset), globalObjectOffset(globalObjectOffset),
		mergeOn(merge_on), promotionOn(promotion_on), promotionLoop(promotion_loop), mergeAABBAware(merge_aabb_aware), promotionAABBAware(promotion_aabb_aware),
		mergeThreshold(merge_threshold), maxCounterThreshold(max_counter_threshold) {
		initializeWorkingBVH();
	}

	uint64_t GetVisitCount(const WorkingNode8& node);
	void initializeWorkingBVH();
	bool promote_most_visited_grandchild(WorkingNode8& workingNode);
	void collapse();
	void rebuild();
	void serialize();
	BVH8RebuildMetadata get_metadata() const;
	void printMetadata();
};
