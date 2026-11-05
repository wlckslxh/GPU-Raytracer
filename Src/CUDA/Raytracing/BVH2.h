#pragma once
#include "BVH.h"

struct AABB {
	float3 min;
	float3 max;

	__device__ inline bool intersects(const Ray & ray, float max_distance) const {
		float3 t0 = (min - ray.origin) / ray.direction;
		float3 t1 = (max - ray.origin) / ray.direction;

		float t_near = vmin_max(t0.x, t1.x, vmin_max(t0.y, t1.y, vmin_max(t0.z, t1.z, 0.0f)));
		float t_far  = vmax_min(t0.x, t1.x, vmax_min(t0.y, t1.y, vmax_min(t0.z, t1.z, max_distance)));

		return t_near < t_far;
	}
};

struct BVH2Node {
	AABB aabb;
	union {
		int left;
		int first;
	};
	unsigned count : 30;
	unsigned axis  : 2;

	__device__ inline bool is_leaf() const {
		return count > 0;
	}

	__device__ inline bool should_visit_left_first(const Ray & ray) const {
		switch (axis) {
			case 0: return ray.direction.x > 0.0f;
			case 1: return ray.direction.y > 0.0f;
			case 2: return ray.direction.z > 0.0f;
		}

		assert(false && "Invalid BVH axis!");
		return false;
	}
};

__device__ __constant__ BVH2Node * bvh2_nodes;
//jichan add
__device__ __constant__ unsigned long long * bvh_counter;
__device__ __constant__ unsigned long long * triangle_counter;
__device__ __constant__ unsigned long long * mesh_counter;
__device__ __constant__ bool * counter_mode;
__device__ __constant__ bool * closest_hit_path_counter_mode;
__device__ __constant__ int * bvh2_parent_indices;
__device__ __constant__ int bvh2_parent_index_count;

// TLAS and every BLAS are separate roots in the aggregated node array. The
// caller invokes this once for the final TLAS leaf and once for the final
// BLAS leaf.
__device__ inline void bvh2_count_hit_path(int leaf_node_index) {
	if (leaf_node_index == INVALID || !bvh_counter || !bvh2_parent_indices) return;

	for (int node_index = leaf_node_index, step = 0;
		node_index != INVALID && step < bvh2_parent_index_count;
		node_index = bvh2_parent_indices[node_index], ++step) {
		if (node_index < 0 || node_index >= bvh2_parent_index_count) return;
		atomicAdd(&bvh_counter[node_index], 1ull);
	}
}

__device__ void bvh2_trace(TraversalData * traversal_data, int ray_count, int * rays_retired) {
	extern __shared__ int shared_stack_bvh2[];

	int stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int    ray_index;
	Ray    ray;
	RayHit ray_hit;

	int  tlas_stack_size;
	int  mesh_id;
	bool mesh_has_identity_transform;
	int  active_tlas_leaf;
	int  closest_tlas_leaf;
	int  closest_blas_leaf;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomicAdd(rays_retired, 1);
			if (ray_index >= ray_count) return;

			ray.origin    = traversal_data->ray_origin   .get(ray_index);
			ray.direction = traversal_data->ray_direction.get(ray_index);

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = INVALID;

			tlas_stack_size = INVALID;
			active_tlas_leaf = INVALID;
			closest_tlas_leaf = INVALID;
			closest_blas_leaf = INVALID;

			// Push root on stack
			stack_size                              = 1;
			shared_stack_bvh2[SHARED_STACK_INDEX(0)] = 0;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;
				active_tlas_leaf = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = traversal_data->ray_origin   .get(ray_index);
					ray.direction = traversal_data->ray_direction.get(ray_index);
				}
			}

			// Pop Node of the stack
			int node_index = stack_pop(shared_stack_bvh2, stack, stack_size);

			const BVH2Node & node = bvh2_nodes[node_index];

			if (node.aabb.intersects(ray, ray_hit.t)) {
				if (*counter_mode && !*closest_hit_path_counter_mode) {
					atomicAdd(&bvh_counter[node_index], 1ull);
				}
				if (node.is_leaf()) {
					if (tlas_stack_size == INVALID) {
						tlas_stack_size = stack_size;
						active_tlas_leaf = node_index;

						mesh_id = node.first;
						if(*counter_mode){
							atomicAdd(&mesh_counter[mesh_id], 1ull);
						}

						int root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform);

						if (!mesh_has_identity_transform) {
							Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
							matrix3x4_transform_position (transform_inv, ray.origin);
							matrix3x4_transform_direction(transform_inv, ray.direction);
						}

						stack_push(shared_stack_bvh2, stack, stack_size, root_index);
					} else {
						for (int i = node.first; i < node.first + node.count; i++) {
							if(*counter_mode){
								atomicAdd(&triangle_counter[i], 1ull);
							}
							float previous_hit_distance = ray_hit.t;
							triangle_intersect(mesh_id, i, ray, ray_hit);
							if (ray_hit.t < previous_hit_distance) {
								closest_tlas_leaf = active_tlas_leaf;
								closest_blas_leaf = node_index;
							}
						}
					}
				} else {
					int first, second;

					if (node.should_visit_left_first(ray)) {
						second = node.left + 1;
						first  = node.left;
					} else {
						second = node.left;
						first  = node.left + 1;
					}

					stack_push(shared_stack_bvh2, stack, stack_size, second);
					stack_push(shared_stack_bvh2, stack, stack_size, first);
				}
			}

			if (stack_size == 0) {
				if (*counter_mode && *closest_hit_path_counter_mode && closest_blas_leaf != INVALID) {
					bvh2_count_hit_path(closest_tlas_leaf);
					bvh2_count_hit_path(closest_blas_leaf);
				}
				traversal_data->hits.set(ray_index, ray_hit);
				break;
			}
		}
	}
}

template<typename OnMissCallback>
__device__ void bvh2_trace_shadow(ShadowTraversalData * traveral_data, int ray_count, int * rays_retired, OnMissCallback callback) {
	extern __shared__ int shared_stack_bvh2[];

	int stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int ray_index;
	Ray ray;

	float max_distance;

	int  tlas_stack_size;
	int  mesh_id;
	bool mesh_has_identity_transform;
	int  active_tlas_leaf;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomicAdd(rays_retired, 1);
			if (ray_index >= ray_count) return;

			ray.origin    = traveral_data->ray_origin   .get(ray_index);
			ray.direction = traveral_data->ray_direction.get(ray_index);

			max_distance = traveral_data->max_distance[ray_index];

			tlas_stack_size = INVALID;
			active_tlas_leaf = INVALID;

			// Push root on stack
			stack_size                              = 1;
			shared_stack_bvh2[SHARED_STACK_INDEX(0)] = 0;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;
				active_tlas_leaf = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = traveral_data->ray_origin   .get(ray_index);
					ray.direction = traveral_data->ray_direction.get(ray_index);
				}
			}

			// Pop Node of the stack
			int node_index = stack_pop(shared_stack_bvh2, stack, stack_size);

			const BVH2Node & node = bvh2_nodes[node_index];

			if (node.aabb.intersects(ray, max_distance)) {
				if (*counter_mode && !*closest_hit_path_counter_mode) {
					atomicAdd(&bvh_counter[node_index], 1ull);
				}
				if (node.is_leaf()) {
					if (tlas_stack_size == INVALID) {
						tlas_stack_size = stack_size;
						active_tlas_leaf = node_index;

						mesh_id = node.first;
						if(*counter_mode){
							atomicAdd(&mesh_counter[mesh_id], 1ull);
						}

						int root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform);

						if (!mesh_has_identity_transform) {
							Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
							matrix3x4_transform_position (transform_inv, ray.origin);
							matrix3x4_transform_direction(transform_inv, ray.direction);
						}

						stack_push(shared_stack_bvh2, stack, stack_size, root_index);
					} else {
						bool hit = false;
						for (int i = node.first; i < node.first + node.count; i++) {
							if(*counter_mode){
								atomicAdd(&triangle_counter[i], 1ull);
							}
							if (triangle_intersect_shadow(i, ray, max_distance)) {
								hit = true;
								break;
							}
						}
						if (hit) {
							if (*counter_mode && *closest_hit_path_counter_mode) {
								bvh2_count_hit_path(active_tlas_leaf);
								bvh2_count_hit_path(node_index);
							}
							stack_size = 0;
							break;
						}
					}
				} else {
					int first, second;

					if (node.should_visit_left_first(ray)) {
						second = node.left + 1;
						first  = node.left;
					} else {
						second = node.left;
						first  = node.left + 1;
					}

					stack_push(shared_stack_bvh2, stack, stack_size, second);
					stack_push(shared_stack_bvh2, stack, stack_size, first);
				}
			}

			if (stack_size == 0) {
				// We didn't hit anything, call callback
				callback(ray_index);
				break;
			}
		}
	}
}
