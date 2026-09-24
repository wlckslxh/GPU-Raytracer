#include "BVH8RebuildRayDistribution.h"

#include <cmath>

static uint8_t count_leaf_primitives(byte meta) {
	uint8_t count = 0;
	for (unsigned mask = unsigned(meta) >> 5; mask != 0; mask &= mask - 1) ++count;
	return count;
}

static AABB restore_child_aabb(const BVHNode8& node, int slot) {
	const Vector3 scale(
		std::ldexp(1.0f, int(node.e[0]) - 127),
		std::ldexp(1.0f, int(node.e[1]) - 127),
		std::ldexp(1.0f, int(node.e[2]) - 127)
	);

	AABB aabb;
	aabb.min = node.p + Vector3(float(node.quantized_min_x[slot]) * scale.x, float(node.quantized_min_y[slot]) * scale.y, float(node.quantized_min_z[slot]) * scale.z);
	aabb.max = node.p + Vector3(float(node.quantized_max_x[slot]) * scale.x, float(node.quantized_max_y[slot]) * scale.y, float(node.quantized_max_z[slot]) * scale.z);
	return aabb;
}

uint64_t BVH8RebuildRayDistribution::GetVisitCount(const WorkingNode8& node) {
	return node.estimatedVisitCount;
}

void BVH8RebuildRayDistribution::initializeWorkingBVH() {
	workingBvh.clear();
	workingBvh.reserve(oldBvh.nodes.size());

	for (uint32_t nodeIndex = 0; nodeIndex < oldBvh.nodes.size(); ++nodeIndex) {
		const BVHNode8& oldNode = oldBvh.nodes[nodeIndex];
		WorkingNode8 newNode = WorkingNode8{};

		uint32_t internalChildrenBefore = 0;
		for (int slot = 0; slot < 8; slot++) {
			const byte meta = oldNode.meta[slot];
			Child& child = newNode.child[slot];
			child.aabb = AABB::create_empty();

			if (meta == 0) continue;

			child.aabb = restore_child_aabb(oldNode, slot);
			if (oldNode.imask & (1u << slot)) {
				child.kind = Internal;
				child.index = oldNode.base_index_child + internalChildrenBefore;
				++internalChildrenBefore;
				++newNode.childCount;
			} else {
				child.kind = Object;
				child.index = oldNode.base_index_triangle + (unsigned(meta) & 0x1f);
				child.primitiveCount = count_leaf_primitives(meta);
				++newNode.objectCount;
			}
		}

		ASSERT(globalNodeOffset + nodeIndex < nodeCounter.size());
		newNode.estimatedVisitCount = nodeCounter[globalNodeOffset + nodeIndex];
		workingBvh.push_back(newNode);
	}
}


void BVH8RebuildRayDistribution::collapse() {

}

void BVH8RebuildRayDistribution::promote_most_visited_grandchild(WorkingNode8& workingNode) {
	uint64_t maxCounter = 0;
	uint8_t maxChild;
	//child8 중 누가 ray 비중이 제일 큰지 탐색
	for (int i = 0; i < 8; i++) {
		const Child& child = workingNode.child[i];
		if (child.kind == Internal && maxCounter < nodeCounter[child.index + globalNodeOffset]) {
			maxCounter = nodeCounter[child.index + globalNodeOffset];
			maxChild = i;
		}
	}

	WorkingNode8& candidateNode = workingBvh[workingNode.child[maxChild].index];
	uint64_t candidateCountSum = 0;
	uint64_t candidateMaxCounter = 0;
	uint8_t candidateMaxChild;
	//ray 비중이 제일 큰 child에서, ray 비중이 제일 큰 child 탐색
	for (int i = 0; i < 8; i++) {
		const Child& child = candidateNode.child[i];
		if (child.kind == Empty) continue;
		//aabb child node와 object child node 분리해서 counter 셀 필요 있음
		if (child.kind == Internal) {
			const uint64_t childCounter = nodeCounter[child.index + globalNodeOffset];
			candidateCountSum += childCounter;
			if (candidateMaxCounter < childCounter) {
				candidateMaxCounter = childCounter;
				candidateMaxChild = i;
			}
		} else {
			uint64_t curChildCounter = 0;
			for (int j = 0; j < child.primitiveCount; j++) {
				const uint64_t objectVisitCount = objectCounter[globalObjectOffset + child.index + j];
				candidateCountSum += objectVisitCount;
				curChildCounter += objectVisitCount;
			}
			if (candidateMaxCounter < curChildCounter) {
				candidateMaxCounter = curChildCounter;
				candidateMaxChild = i;
			}
		}
	}
	//해당 grandchild를 위로 올리면 현 child의 aabb가 얼마나 줄어드는지 계산하여 올리기
	//일단 지금은 aabb 따로 고려 안 하고 ray 비중만 갖고 무조건 올리는걸로 실험
	AABB candidateAABB = AABB::create_empty();
	for (int i = 0; i < 8; i++) {
		const Child& child = candidateNode.child[i];
		if (i != candidateMaxChild && child.kind != Empty) {
			candidateAABB.expand(child.aabb);
		}
	}

	for (int i = 0; i < 8; i++) {
		if (workingNode.child[i].kind != Empty) continue;

		const Child promotedChild = candidateNode.child[candidateMaxChild];
		workingNode.child[i] = promotedChild;

		if (promotedChild.kind == Internal) {
			++workingNode.childCount;
			--candidateNode.childCount;
		} else {
			++workingNode.objectCount;
			--candidateNode.objectCount;
		}

		candidateNode.child[candidateMaxChild].kind = Empty;
		candidateNode.child[candidateMaxChild].index = INVALID_NODE;
		candidateNode.child[candidateMaxChild].primitiveCount = 0;
		candidateNode.child[candidateMaxChild].aabb = candidateAABB;
		break;
	}
}

//생각해 볼 조건 0. internal box node만 건드릴지, 삼각형도 건드릴지
//우선 보수적으로 internal node에 대해서만 해보고, 그 뒤에 확장을 함. 다만 암만 생각해도 leaf node merging은 아닌 것 같은데...
//논리적으로 현재 하고자 하는게, ray가 많이 가는 node는 위로, 적게 가는 node는 아래로 보내는 것이라서 구분을 안 해도 괜찮을 듯?
void BVH8RebuildRayDistribution::rebuild() {
	//initialize queue with index 0 of workingBvh8
	queue.clear();
	queue.push_back(0);
	uint32_t workingIndex = 0;
	while (queue.size() != 0) {
		workingIndex = queue.back();
		queue.pop_back();

		auto& workingNode = workingBvh[workingIndex];

		//early termination on nodes without bvh child nodes
		if (workingNode.childCount == 0) {
			workingNode.state = Finished;
			continue;
		}
		//if there is an empty child node, do promotion only
		//생각해 볼 조건 1. 무조건 promotion이 좋은가?
		//promotion은 grandchild가 존재해야 하므로, leaf child node는 후보에서 자연스럽게 제외
		//단, 후보 node에서 뭘 올릴지는 상관없이 ray count 기반으로 수행하는게 맞다고 봄
		//여기서 생각해볼게, 무턱대고 올리기 보다는 SA 변화량이랑 ray count 감소율(예측)을 통해 수치를 잡는게 좋다고 생각함
		if (workingNode.childCount + workingNode.objectCount < 8) {
			promote_most_visited_grandchild(workingNode);

			//조정 이후 현재 node의 child들 push
			for (int slot = 7; slot > -1; slot--) {
				const Child& child = workingNode.child[slot];

				if (child.kind == Internal) {
					queue.push_back(child.index);
				}
			}
			workingNode.state = Finished;
		}
		//if not, merge first and do promotion
		else {
			uint64_t counterSum = 0;
			uint64_t minACounter = -1;
			uint64_t minBCounter = -1;
			uint8_t minAChild, minBChild;
			for (int i = 0; i < 8; i++) {
				const Child& child = workingNode.child[i];
				if (child.kind == Internal) {
					counterSum += nodeCounter[child.index + globalNodeOffset];
				}
				if (child.kind == Object) {
					uint64_t curChildCounter = 0;
					for (int j = 0; j < child.primitiveCount; j++) {
						const uint64_t objectVisitCount = objectCounter[globalObjectOffset + child.index + j];
						curChildCounter += objectVisitCount;
					}
					if (minACounter > curChildCounter) {
						minACounter = curChildCounter;
						minAChild = i;
					}
					else if (minBCounter > nodeCounter[child.index + globalObjectOffset]) {
						minBCounter = curChildCounter;
						minBChild = i;
					}
					counterSum += curChildCounter;
				}
			}
			bool doMerge = minAChild + minBChild / counterSum < 0.3;

			if (doMerge) {
				WorkingNode8 mergeNode;

				//merge 이후 promotion 진행
				promote_most_visited_grandchild(workingNode);
			}
		}
	}
}

void BVH8RebuildRayDistribution::resort() {

}
