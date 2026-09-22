#pragma once
#include "Renderer/Triangle.h"

#include "BVH/BVH.h"

#include "Core/Array.h"
#include "Core/OwnPtr.h"

struct MeshData {
	Array<Triangle> triangles;
	// Optional per-original-triangle material IDs. INVALID means use the
	// material assigned to the owning Mesh instance.
	Array<int>      material_ids;
	OwnPtr<BVH>     bvh;
};
