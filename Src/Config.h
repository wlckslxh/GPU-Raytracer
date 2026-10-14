#pragma once
#include "CUDA/Common.h"

#include "Core/Array.h"
#include "Core/String.h"

inline GPUConfig gpu_config = { };

enum struct IntegratorType {
	PATHTRACER,
	AO
};

enum struct OutputFormat {
	EXR,
	PPM
};

enum struct MipmapFilterType {
	BOX,
	LANCZOS,
	KAISER
};

enum struct BVHType {
	BVH,  // Binary SAH-based BVH
	SBVH, // Binary SAH-based Spatial BVH
	BVH4, // Quaternary BVH,              constructed by collapsing the binary BVH
	BVH8  // Compressed Wide BVH (8 way), constructed by collapsing the binary BVH
};

struct CPUConfig {
	// CUDA render target resolution. Scene film settings and --width/--height modify these.
	int initial_width  = 2560;
	int initial_height = 1440;

	// SDL window size. This only controls presentation and GUI size.
	int initial_window_width  = 1280;
	int initial_window_height = 720;

	Array<String> scene_filenames;
	String        sky_filename;

	IntegratorType integrator = IntegratorType::PATHTRACER;

	OutputFormat screenshot_format = OutputFormat::PPM;

	int    output_sample_index = INVALID;
	String output_filename     = "render.ppm"_sv;

	bool bvh_force_rebuild        = false;
	bool enable_bvh_optimization  = false;
	bool enable_block_compression = true;
	bool enable_scene_update      = false;

	MipmapFilterType mipmap_filter = MipmapFilterType::BOX;

	BVHType bvh_type = BVHType::BVH8;

	float sah_cost_node = 4.0f;
	float sah_cost_leaf = 1.0f;

	float sbvh_alpha = 10e-5f; // Alpha parameter for SBVH construction, alpha == 1 means regular BVH, alpha == 0 means full SBVH

	int bvh_optimizer_max_time        = 60000; // Time limit in milliseconds
	int bvh_optimizer_max_num_batches = 1000;

	// Runtime controls for BVH8RebuildRayDistribution. Keep these in CPUConfig
	// so an external sweep can change them through command-line arguments
	// without editing and recompiling the rebuild header.
	bool  rebuild_merge_on                = true;
	bool  rebuild_promotion_on            = true;
	bool  rebuild_promotion_loop          = false;
	bool  rebuild_merge_aabb_aware        = true;
	bool  rebuild_promotion_aabb_aware    = true;
	float rebuild_merge_threshold         = 0.25f;
	float rebuild_max_counter_threshold   = 0.3f;
	// Optional suffix appended to rebuild statistics CSV filenames. This keeps
	// remeasurements separate from an earlier run with identical controls.
	String rebuild_statistics_suffix;
	bool  exit_after_rebuild_statistics   = false;
};

inline CPUConfig cpu_config = { };
