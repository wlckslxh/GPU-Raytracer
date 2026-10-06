#pragma once
#include <cstdlib>
#include <stdint.h>

#include "Config.h"

#include "Core/String.h"
#include "Core/StringView.h"

#include "BVH/BVH.h"
#include "Renderer/MeshData.h"

namespace BVHLoader {
	inline constexpr const char * BVH_FILE_EXTENSION = ".bvh";
	inline constexpr int          BVH_FILETYPE_VERSION = 7;
	// Cache of the final, local wide BVH.  This is intentionally separate from
	// the legacy .bvh cache, which stores the intermediate binary BVH2.
	inline constexpr const char * BVH8_FILE_EXTENSION = ".bvh8";
	inline constexpr int          BVH8_FILETYPE_VERSION = 1;

	String get_bvh_filename(StringView filename, Allocator * allocator);
	String get_bvh8_filename(StringView filename, Allocator * allocator);

	bool try_to_load(const String & filename, const String & bvh_filename, MeshData * mesh_data, BVH2 * bvh);
	bool save(const String & bvh_filename, const MeshData & mesh_data, const BVH2 & bvh);

	// The caller owns the current triangle data. It is still parsed from glTF so
	// materials, textures and lights remain current; only construction and
	// conversion of the initial BVH8 are skipped on a cache hit.
	bool try_to_load_bvh8(const String & filename, const String & bvh_filename, const Array<Triangle> & triangles, BVH8 * bvh);
	bool save_bvh8(const String & bvh_filename, const Array<Triangle> & triangles, const BVH8 & bvh);
}
