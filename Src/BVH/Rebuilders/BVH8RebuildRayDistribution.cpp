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
				ASSERT(globalNodeOffset + child.index < nodeCounter.size());
				child.visitCount = nodeCounter[globalNodeOffset + child.index];
				++internalChildrenBefore;
				++newNode.childCount;
			} else {
				child.kind = Object;
				child.index = oldNode.base_index_triangle + (unsigned(meta) & 0x1f);
				child.primitiveCount = count_leaf_primitives(meta);
				for (uint32_t primitive = 0; primitive < child.primitiveCount; ++primitive) {
					ASSERT(globalObjectOffset + child.index + primitive < objectCounter.size());
					child.visitCount += objectCounter[globalObjectOffset + child.index + primitive];
				}
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
	uint8_t maxChild = UINT8_MAX;
	//child8 중 누가 ray 비중이 제일 큰지 탐색
	for (int i = 0; i < 8; i++) {
		const Child& child = workingNode.child[i];
		if (child.kind == Internal && (maxChild == UINT8_MAX || maxCounter < child.visitCount)) {
			maxCounter = child.visitCount;
			maxChild = i;
		}
	}
	ASSERT(maxChild != UINT8_MAX);

	WorkingNode8& candidateNode = workingBvh[workingNode.child[maxChild].index];
	uint64_t candidateCountSum = 0;
	uint64_t candidateMaxCounter = 0;
	uint8_t candidateMaxChild = UINT8_MAX;
	//ray 비중이 제일 큰 child에서, ray 비중이 제일 큰 child 탐색
	for (int i = 0; i < 8; i++) {
		const Child& child = candidateNode.child[i];
		if (child.kind == Empty) continue;
		//aabb child node와 object child node 분리해서 counter 셀 필요 있음
		candidateCountSum += child.visitCount;
		if (candidateMaxChild == UINT8_MAX || candidateMaxCounter < child.visitCount) {
			candidateMaxCounter = child.visitCount;
			candidateMaxChild = i;
		}
	}
	ASSERT(candidateMaxChild != UINT8_MAX);
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
		workingNode.child[maxChild].aabb = candidateAABB;
		candidateNode.child[candidateMaxChild].aabb = AABB::create_empty();
		break;
	}
	promotionCount++;
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
			uint64_t counterSum = 0, maxCounter = 0, minACounter = -1, minBCounter = -1;
			uint8_t minAChild = 0, minBChild = 0, maxChild = 0;
			//check both min two childs and max single child for merge condition
			for (int i = 0; i < 8; i++) {
				const Child& child = workingNode.child[i];
				if (child.kind == Internal) {
					if (maxCounter < child.visitCount) {
						maxCounter = child.visitCount;
						maxChild = i;
					}
					if (minACounter > child.visitCount) {
						minBCounter = minACounter;
						minACounter = child.visitCount;
						minBChild = minAChild;
						minAChild = i;
					}
					else if (minBCounter > child.visitCount) {
						minBCounter = child.visitCount;
						minBChild = i;
					}
					counterSum += child.visitCount;
				}
				if (child.kind == Object) {
					const uint64_t curChildCounter = child.visitCount;
					if (maxCounter < curChildCounter) {
						maxCounter = curChildCounter;
						maxChild = i;
					}
					if (minACounter > curChildCounter) {
						minBCounter = minACounter;
						minACounter = curChildCounter;
						minBChild = minAChild;
						minAChild = i;
					}
					else if (minBCounter > curChildCounter) {
						minBCounter = curChildCounter;
						minBChild = i;
					}
					counterSum += curChildCounter;
				}
			}
			bool doMerge = counterSum != 0 && (((float)(minACounter + minBCounter) / counterSum) < 0.25) && (((float)maxCounter / counterSum) > 0.3);

			if (doMerge) {
				//merge node를 만들면, child가 둘 뿐이므로 무조건 빈 틈이 생겨 promotion을 진행하는데, 이걸 냅둬도 되나...?
				WorkingNode8 mergeNode;
				mergeNode.child[0] = workingNode.child[minAChild];
				mergeNode.child[1] = workingNode.child[minBChild];
				if (mergeNode.child[0].kind == Internal) {
					mergeNode.childCount++;
					workingNode.childCount--;
				}
				else {
					mergeNode.objectCount++;
					workingNode.objectCount--;
				}
				if (mergeNode.child[1].kind == Internal) {
					mergeNode.childCount++;
					workingNode.childCount--;
				}
				else {
					mergeNode.objectCount++;
					workingNode.objectCount--;
				}
				AABB mergeAABB = AABB::unify(mergeNode.child[0].aabb, mergeNode.child[1].aabb);
				const uint64_t mergeVisitCount = mergeNode.child[0].visitCount + mergeNode.child[1].visitCount;
				mergeNode.estimatedVisitCount = mergeVisitCount;
				const uint32_t mergeNodeIndex = uint32_t(workingBvh.size());
				workingBvh.push_back(mergeNode);

				WorkingNode8& updatedWorkingNode = workingBvh[workingIndex];
				updatedWorkingNode.child[minAChild] = Child{Internal, mergeNodeIndex, 0, mergeVisitCount, mergeAABB};
				updatedWorkingNode.child[minBChild] = Child{};
				updatedWorkingNode.child[minBChild].aabb = AABB::create_empty();
				updatedWorkingNode.childCount++;

				mergeCount++;
				//merge 이후 promotion 진행
				promote_most_visited_grandchild(updatedWorkingNode);
			}

			WorkingNode8& updatedWorkingNode = workingBvh[workingIndex];
			for (int slot = 7; slot > -1; slot--) {
				const Child& child = updatedWorkingNode.child[slot];

				if (child.kind == Internal) {
					queue.push_back(child.index);
				}
			}
			updatedWorkingNode.state = Finished;
		}
	}
}

//rebuild한 working bvh를 원래 bvh 구조로 되돌림
void BVH8RebuildRayDistribution::serialize() {
	//고려사항
	//1. internal child 연속 배치
	//2. internal child meta/imask 재생성
	//3. object child도 연속 배치
	//4. AABB는 bottom-up으로 다시 계산
	//5. 여기 코드는 기존 bvh를 dfs 방식으로 child들을 만들어서, 이 방법을 그대로 따라가기로 결정
	serializePendingNode startNode = { 0, 0 };
	serializeQueue.clear();
	serializeQueue.push_back(startNode);

	newBvh.nodes.clear();
	newBvh.indices.clear();
	newBvh.nodes.emplace_back();

	while (serializeQueue.size()) {
		serializePendingNode pending = serializeQueue.back();
		serializeQueue.pop_back();
		
		const WorkingNode8& workingNode = workingBvh[pending.workingIndex];
		
		BVHNode8 outputNode = {};
		
		//AABB 계산
		AABB nodeAabb = AABB::create_empty();
		uint32_t activeCount = 0;
		uint32_t activeChild[8] = {};
		for (int i = 0; i < 8; i++) {
			if (workingNode.child[i].kind != Empty) {
				nodeAabb.expand(workingNode.child[i].aabb);
				activeChild[activeCount++] = i;
			}
		}
		ASSERT(nodeAabb.is_valid());

		outputNode.p = nodeAabb.min;

		//exponent 계산
		constexpr int Nq = 8;
		constexpr float denom = 1.0f / 255.0f;

		Vector3 e(exp2f(ceilf(log2f((nodeAabb.max.x - nodeAabb.min.x) * denom))),
			exp2f(ceilf(log2f((nodeAabb.max.y - nodeAabb.min.y) * denom))),
			exp2f(ceilf(log2f((nodeAabb.max.z - nodeAabb.min.z) * denom))));

		Vector3 one_over_e(1.0f / e.x, 1.0f / e.y, 1.0f / e.z);

		unsigned u_ex = Util::bit_cast<unsigned>(e.x);
		unsigned u_ey = Util::bit_cast<unsigned>(e.y);
		unsigned u_ez = Util::bit_cast<unsigned>(e.z);

		outputNode.e[0] = u_ex >> 23;
		outputNode.e[1] = u_ey >> 23;
		outputNode.e[2] = u_ez >> 23;

		//internal child node 구성 및 enqueue
		outputNode.base_index_child = uint32_t(newBvh.nodes.size());

		for (int i = 0; i < workingNode.childCount; i++) {
			//뒤에서 자리 잡아 줄 internal node를 미리 예약 함
			newBvh.nodes.emplace_back();
		}
		//object child node 구성
		outputNode.base_index_triangle = uint32_t(newBvh.indices.size());

		//이 코드에서 사용하는 ordering에 기반하여 child reorder
		Vector3 p = nodeAabb.get_center();

		float cost[8][8] = {};

		//현재는 정작 비어있는 노드를 읽고, 차 있는 노드를 안 읽을 가능성 높음
		for (int c = 0; c < activeCount; c++) {
			uint32_t index = activeChild[c];
			for (int s = 0; s < 8; s++) {
				Vector3 direction((s & 0b100) ? -1.0f : 1.0f,
					(s & 0b010) ? -1.0f : 1.0f,
					(s & 0b001) ? -1.0f : 1.0f);

				cost[c][s] = Vector3::dot(workingNode.child[index].aabb.get_center() - p, direction);
			}
		}

		uint32_t assignment[8] = { INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE };
		bool slot_filled[8] = { };

		while (true) {
			float min_cost = INFINITY;
			int min_slot = INVALID;
			int min_index = INVALID;

			for (int c = 0; c < activeCount; c++) {
				if (assignment[c] == INVALID_NODE) {
					for (int s = 0; s < 8; s++) {
						if (!slot_filled[s] && cost[c][s] < min_cost) {
							min_cost = cost[c][s];

							min_slot = s;
							min_index = c;
						}
					}
				}
			}

			if (min_slot == INVALID) break;
			slot_filled[min_slot] = true;
			assignment[min_index] = min_slot;
		}

		uint32_t reorderedChild[8] = { INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE };
		for (int i = 0; i < activeCount; i++) {
			reorderedChild[assignment[i]] = activeChild[i];
		}

		serializePendingNode internalPending[8];
		int serializePendingCount = 0;
		uint32_t internalOffset = 0;
		uint32_t objectOffset = 0;

		for (int i = 0; i < 8; i++) {
			uint32_t source = reorderedChild[i];
			if (source == INVALID_NODE) continue;

			const Child& child = workingNode.child[source];
			if (child.kind == Empty) continue;

			//internal child node 양자화
			outputNode.quantized_min_x[i] = byte(floorf((child.aabb.min.x - outputNode.p.x) * one_over_e.x));
			outputNode.quantized_min_y[i] = byte(floorf((child.aabb.min.y - outputNode.p.y) * one_over_e.y));
			outputNode.quantized_min_z[i] = byte(floorf((child.aabb.min.z - outputNode.p.z) * one_over_e.z));

			outputNode.quantized_max_x[i] = byte(ceilf((child.aabb.max.x - outputNode.p.x) * one_over_e.x));
			outputNode.quantized_max_y[i] = byte(ceilf((child.aabb.max.y - outputNode.p.y) * one_over_e.y));
			outputNode.quantized_max_z[i] = byte(ceilf((child.aabb.max.z - outputNode.p.z) * one_over_e.z));

			if (child.kind == Internal) {
				uint32_t childOutputIndex = outputNode.base_index_child + internalOffset;
				outputNode.imask |= 1u << i;
				outputNode.meta[i] = uint8_t((i + 24) | 0x20);
				internalPending[serializePendingCount++] = { child.index, childOutputIndex };
				internalOffset++;
			}
			else if (child.kind == Object) {
				uint32_t objectCount = child.primitiveCount;
				uint32_t unaryCount = ((1u << objectCount) - 1u) << 5;
				outputNode.meta[i] = uint8_t(objectOffset | unaryCount);

				for (int j = 0; j < objectCount; j++) {
					newBvh.indices.push_back(oldBvh.indices[child.index + j]);
				}
				objectOffset += objectCount;
			}
		}
		ASSERT(objectOffset <= 24);

		//실제 위치에 bvh8 node 할당
		newBvh.nodes[pending.outputIndex] = outputNode;

		for (int i = serializePendingCount - 1; i >= 0; i--) {
			serializeQueue.push_back(internalPending[i]);
		}
	}

	//최종 indice 보존 검사
	ASSERT(newBvh.indices.size() == oldBvh.indices.size());
}
