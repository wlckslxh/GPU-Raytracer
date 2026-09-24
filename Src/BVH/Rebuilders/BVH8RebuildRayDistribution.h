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
	AABB aabb;
};

struct WorkingNode8 {
	uint32_t childCount = 0;
	uint32_t objectCount = 0;
	Child child[8];
	uint64_t estimatedVisitCount = 0;
	byte state = Unvisited;
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

	BVH8RebuildRayDistribution(const BVH8& oldBvh, const Array<uint64_t>& nodeCounter, const Array<uint64_t>& objectCounter, BVH8& newBvh, uint32_t globalNodeOffset, uint32_t globalObjectOffset, bool is_tlas) : oldBvh(oldBvh), nodeCounter(nodeCounter), objectCounter(objectCounter), newBvh(newBvh), globalNodeOffset(globalNodeOffset), globalObjectOffset(globalObjectOffset), is_tlas(is_tlas) {	
		initializeWorkingBVH();
	}

	uint64_t GetVisitCount(const WorkingNode8& node);
	void initializeWorkingBVH();
	void promote_most_visited_grandchild(WorkingNode8& workingNode);
	void collapse();
	void rebuild();
	void resort();
};
