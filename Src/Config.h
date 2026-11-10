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
	// Optional one-based counter-camera view selected for an ordinary render.
	// A negative value preserves the normal scene camera preset behavior.
	int counter_camera_view_index = -1;

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
	// Prefer counter_camera_views during pixel-counter collection when the
	// manifest provides them. Disable this to collect and measure FPS on the
	// same camera_keyframes path.
	bool     use_dedicated_counter_camera_views = true;
	// Keep the historical FPS path by default. Experiments can opt into timing
	// the same dedicated counter-camera path used for counter collection.
	bool     use_dedicated_fps_camera_views = false;
	// Records the number of camera/path rays traced at every bounce while the
	// source BVH2 is active. This is a measurement-only mode: it stops after
	// collection and does not collapse to BVH8.
	bool     enable_bounce_ray_count_dump = false;
	bool     enable_experiment_fps_measurement = true;
	// Bracket only MEASURING_FPS with the CUDA profiler API. Nsight Systems can
	// use this range to exclude counter collection, collapse, upload and warmup.
	bool     profile_fps_cuda_kernels = false;
	bool     exit_after_experiment = false;
	// Optional filename tag used to keep otherwise identical experiment rows
	// from different camera/evaluation protocols in separate CSV files.
	String   experiment_tag;
	// Controls the BVH2 -> BVH8 experiment conversion performed after counter
	// collection. These are kept in CPUConfig so a batch run can set them per
	// process without editing source files.
	bool     enable_counter_collapse = true;
	bool     enable_wald_collapse = false;
	// Relative weight of the measured traversal term. Zero recovers the
	// surface-area collapse; one preserves the original counter blend.
	float    counter_cost_lambda = 1.0f;
	// Exclusive BVH2 depth limit for counter-guided collapse. -1 applies the
	// counter at every depth; 0 reproduces the surface-area-only collapse.
	int      counter_collapse_max_depth = -1;
	// Preserve the original counter-collapse leaf cost by default. When false,
	// a collapsed 2- or 3-primitive leaf is additionally charged per primitive.
	bool     legacy_counter_primitive_cost = true;
	// Gu et al. CBTC-RD: use visit-count ratio for sufficiently sampled
	// children, otherwise fall back to the surface-area ratio. This is kept
	// separate from the project's existing counter-aware heuristic.
	bool     enable_gu_rdtc = false;
	uint64_t gu_rdtc_threshold = 32ull;
	// The collection pass and the timed render may intentionally use different
	// path lengths.  This permits, for example, a primary-ray distribution to
	// guide a full path-tracing measurement.
	int      counter_collection_bounces = 1;
	// When enabled, collection traces continue until counter_collection_bounces
	// unless they miss or are absorbed by the scene.  The user's normal Russian
	// roulette setting is restored before the post-conversion FPS pass.
	bool     disable_russian_roulette_during_counter_collection = false;
	// Replace camera rays with counter-only paths emitted from unique scene
	// vertices into the cosine-weighted hemisphere around each vertex normal.
	// These paths reuse traversal and BSDF sampling but never write render AOVs.
	bool     enable_vertex_counter_rays = false;
	int      counter_rays_per_vertex = 1;
	// Attribute counter samples only to final closest-hit paths for regular
	// rays and first blocking paths for shadow rays.
	bool     enable_closest_hit_path_counter = false;
	int      measurement_bounces = 10;
	// Number of complete camera paths dispatched per pixel in one rendered
	// frame. Each path uses an independent sample index.
	int      rays_per_pixel_per_frame = 1;
	uint64_t counter_collection_frames = 200ull;
	uint64_t fps_warmup_frames = 10ull;
	uint64_t fps_measurement_frames = 200ull;

	MipmapFilterType mipmap_filter = MipmapFilterType::BOX;

	BVHType bvh_type = BVHType::BVH8;

	float sah_cost_node = 4.0f;
	float sah_cost_leaf = 1.0f;

	float sbvh_alpha = 10e-5f; // Alpha parameter for SBVH construction, alpha == 1 means regular BVH, alpha == 0 means full SBVH

	int bvh_optimizer_max_time        = 60000; // Time limit in milliseconds
	int bvh_optimizer_max_num_batches = 1000;
};

inline CPUConfig cpu_config = { };
