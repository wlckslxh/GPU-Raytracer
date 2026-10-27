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
	// glTF keeps triangle and material data in the importer. These files cache
	// only the binary BVH2 topology used as input to wide-BVH conversion.
	inline constexpr const char * BVH2_SAH_CACHE_EXTENSION  = ".bvh2.sah";
	inline constexpr const char * BVH2_SBVH_CACHE_EXTENSION = ".bvh2.sbvh";
	inline constexpr uint32_t     BVH2_CACHE_FILETYPE_VERSION = 1;
	// Cache of the final, local wide BVH.  This is intentionally separate from
	// the legacy .bvh cache, which stores the intermediate binary BVH2.
	inline constexpr const char * BVH8_SAH_CACHE_EXTENSION  = ".bvh8.sah";
	inline constexpr const char * BVH8_SBVH_CACHE_EXTENSION = ".bvh8.sbvh";
	inline constexpr int          BVH8_FILETYPE_VERSION = 2;

	String get_bvh_filename(StringView filename, Allocator * allocator);
	String get_bvh2_cache_filename(StringView filename, Allocator * allocator);
	String get_bvh8_filename(StringView filename, Allocator * allocator);

	bool try_to_load(const String & filename, const String & bvh_filename, MeshData * mesh_data, BVH2 * bvh);
	bool save(const String & bvh_filename, const MeshData & mesh_data, const BVH2 & bvh);
	bool try_to_load_bvh2_cache(const String & filename, const String & bvh_filename, const Array<Triangle> & triangles, BVH2 * bvh);
	bool save_bvh2_cache(const String & bvh_filename, const Array<Triangle> & triangles, const BVH2 & bvh);

	// The caller owns the current triangle data. It is still parsed from glTF so
	// materials, textures and lights remain current; only construction and
	// conversion of the initial BVH8 are skipped on a cache hit.
	bool try_to_load_bvh8(const String & filename, const String & bvh_filename, const Array<Triangle> & triangles, BVH8 * bvh);
	bool save_bvh8(const String & bvh_filename, const Array<Triangle> & triangles, const BVH8 & bvh);
}
