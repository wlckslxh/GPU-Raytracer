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

	// Reproducible camera-path experiments. Counter collection and FPS timing
	// are deliberately independent: a normal FPS run does not allocate or
	// update traversal counters.
	// Master switch for the BVH2 -> BVH8 conversion/measurement experiment.
	// When false, the renderer keeps and traces the BVH selected by bvh_type.
	bool     enable_bvh2_to_bvh8_experiment = false;
	bool     enable_bvh_counter_collection = true;
	bool     enable_experiment_fps_measurement = true;
	bool     exit_after_experiment = false;
	// Controls the BVH2 -> BVH8 experiment conversion performed after counter
	// collection. These are kept in CPUConfig so a batch run can set them per
	// process without editing source files.
	bool     enable_counter_collapse = true;
	bool     enable_wald_collapse = false;
	uint64_t counter_collection_frames = 200ull;
	uint64_t fps_warmup_frames = 10ull;
	uint64_t fps_measurement_frames = 200ull;

	MipmapFilterType mipmap_filter = MipmapFilterType::BOX;

	BVHType bvh_type = BVHType::BVH;

	float sah_cost_node = 4.0f;
	float sah_cost_leaf = 1.0f;

	float sbvh_alpha = 10e-5f; // Alpha parameter for SBVH construction, alpha == 1 means regular BVH, alpha == 0 means full SBVH

	int bvh_optimizer_max_time        = 60000; // Time limit in milliseconds
	int bvh_optimizer_max_num_batches = 1000;
};

inline CPUConfig cpu_config = { };
