#include "Pathtracer.h"

#include <cmath>

#include <Imgui/imgui.h>

#include "Core/Allocators/LinearAllocator.h"

#include "CUDA/Common.h"

void Pathtracer::cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) {
	init_module();
	init_globals();

	pinned_buffer_sizes = CUDAMemory::malloc_pinned<BufferSizes>();
	pinned_buffer_sizes->reset(BATCH_SIZE);

	global_buffer_sizes = cuda_module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	resize_init(frame_buffer_handle, screen_width, screen_height);

	init_materials();
	init_geometry();
	init_sky();
	init_rng();
	init_events();
	init_luts();

	ray_buffer_trace_0.init(BATCH_SIZE);
	ray_buffer_trace_1.init(BATCH_SIZE);
	cuda_module.get_global("ray_buffer_trace_0").set_value(ray_buffer_trace_0);
	cuda_module.get_global("ray_buffer_trace_1").set_value(ray_buffer_trace_1);

	global_ray_buffer_shadow = cuda_module.get_global("ray_buffer_shadow");

	global_svgf_data = cuda_module.get_global("svgf_data");

	global_lights_total_weight = cuda_module.get_global("lights_total_weight");
	global_lights_total_weight.set_value(0.0f);

	Integrator::cuda_init(frame_buffer_handle, screen_width, screen_height);
}

void Pathtracer::cuda_free() {
	Integrator::cuda_free();

	free_luts();
	free_materials();
	free_geometry();
	free_sky();
	free_rng();

	CUDAMemory::free_pinned(pinned_buffer_sizes);

	if (scene.has_lights) {
		if (ptr_light_triangle_indices.ptr) CUDAMemory::free(ptr_light_triangle_indices);
		if (ptr_light_triangle_cumulative_probability.ptr) CUDAMemory::free(ptr_light_triangle_cumulative_probability);
		if (ptr_light_triangle_mesh_indices.ptr) CUDAMemory::free(ptr_light_triangle_mesh_indices);

		ray_buffer_shadow.free();
	}

	ray_buffer_trace_0.free();
	ray_buffer_trace_1.free();

	for (size_t i = 0; i < material_ray_buffers.size(); i++) {
		material_ray_buffers[i].free();
	}
	material_ray_buffers.clear();

	CUDAMemory::free(ptr_material_ray_buffers);
}

void Pathtracer::init_module() {
	cuda_module.init("Pathtracer"_sv, "Src/CUDA/Pathtracer.cu"_sv, CUDAContext::compute_capability, MAX_REGISTERS);

	kernel_integrate_dielectric.init(&cuda_module, "kernel_integrate_dielectric");
	kernel_integrate_conductor .init(&cuda_module, "kernel_integrate_conductor");
	kernel_average_dielectric  .init(&cuda_module, "kernel_average_dielectric");
	kernel_average_conductor   .init(&cuda_module, "kernel_average_conductor");
	kernel_generate            .init(&cuda_module, "kernel_generate");
	kernel_trace_bvh2          .init(&cuda_module, "kernel_trace_bvh2");
	kernel_trace_bvh4          .init(&cuda_module, "kernel_trace_bvh4");
	kernel_trace_bvh8          .init(&cuda_module, "kernel_trace_bvh8");
	kernel_sort                .init(&cuda_module, "kernel_sort");
	kernel_material_diffuse    .init(&cuda_module, "kernel_material_diffuse");
	kernel_material_plastic    .init(&cuda_module, "kernel_material_plastic");
	kernel_material_dielectric .init(&cuda_module, "kernel_material_dielectric");
	kernel_material_conductor  .init(&cuda_module, "kernel_material_conductor");
	kernel_trace_shadow_bvh2   .init(&cuda_module, "kernel_trace_shadow_bvh2");
	kernel_trace_shadow_bvh4   .init(&cuda_module, "kernel_trace_shadow_bvh4");
	kernel_trace_shadow_bvh8   .init(&cuda_module, "kernel_trace_shadow_bvh8");
	kernel_svgf_reproject      .init(&cuda_module, "kernel_svgf_reproject");
	kernel_svgf_variance       .init(&cuda_module, "kernel_svgf_variance");
	kernel_svgf_atrous         .init(&cuda_module, "kernel_svgf_atrous");
	kernel_svgf_finalize       .init(&cuda_module, "kernel_svgf_finalize");
	kernel_taa                 .init(&cuda_module, "kernel_taa");
	kernel_taa_finalize        .init(&cuda_module, "kernel_taa_finalize");
	kernel_accumulate          .init(&cuda_module, "kernel_accumulate");

	switch (cpu_config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: kernel_trace = &kernel_trace_bvh2; kernel_trace_shadow = &kernel_trace_shadow_bvh2; break;
		case BVHType::BVH4: kernel_trace = &kernel_trace_bvh4; kernel_trace_shadow = &kernel_trace_shadow_bvh4; break;
		case BVHType::BVH8: kernel_trace = &kernel_trace_bvh8; kernel_trace_shadow = &kernel_trace_shadow_bvh8; break;
		default: ASSERT_UNREACHABLE();
	}

	// Set Block dimensions for all Kernels
	kernel_integrate_dielectric.set_block_dim(256, 1, 1);
	kernel_integrate_conductor .set_block_dim(256, 1, 1);
	kernel_average_dielectric  .set_block_dim(256, 1, 1);
	kernel_average_conductor   .set_block_dim(256, 1, 1);
	kernel_generate            .set_block_dim(256, 1, 1);
	kernel_sort                .set_block_dim(256, 1, 1);
	kernel_material_diffuse    .set_block_dim(256, 1, 1);
	kernel_material_plastic    .set_block_dim(256, 1, 1);
	kernel_material_dielectric .set_block_dim(256, 1, 1);
	kernel_material_conductor  .set_block_dim(256, 1, 1);

	kernel_svgf_reproject.occupancy_max_block_size_2d();
	kernel_svgf_variance .occupancy_max_block_size_2d();
	kernel_svgf_atrous   .occupancy_max_block_size_2d();
	kernel_svgf_finalize .occupancy_max_block_size_2d();
	kernel_taa           .occupancy_max_block_size_2d();
	kernel_taa_finalize  .occupancy_max_block_size_2d();
	kernel_accumulate    .occupancy_max_block_size_2d();

	// BVH8 uses a stack of int2's (8 bytes)
	// Other BVH's use a stack of ints (4 bytes)
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_bvh2);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_bvh4);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_bvh8);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_bvh2);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_bvh4);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_shadow_bvh8);

	kernel_integrate_dielectric.set_grid_dim(Math::divide_round_up(LUT_DIELECTRIC_DIM_IOR * LUT_DIELECTRIC_DIM_ROUGHNESS * LUT_DIELECTRIC_DIM_COS_THETA, kernel_integrate_dielectric.block_dim_x), 1, 1);
	kernel_average_dielectric  .set_grid_dim(Math::divide_round_up(LUT_DIELECTRIC_DIM_IOR * LUT_DIELECTRIC_DIM_ROUGHNESS,                                kernel_average_dielectric  .block_dim_x), 1, 1);

	kernel_integrate_conductor.set_grid_dim(Math::divide_round_up(LUT_CONDUCTOR_DIM_ROUGHNESS * LUT_CONDUCTOR_DIM_COS_THETA, kernel_integrate_conductor.block_dim_x), 1, 1);
	kernel_average_conductor  .set_grid_dim(Math::divide_round_up(LUT_CONDUCTOR_DIM_ROUGHNESS,                               kernel_average_conductor  .block_dim_x), 1, 1);
}

void Pathtracer::init_events() {
	int display_order = 0;
	event_desc_primary = { display_order++, "Primary"_sv, "Primary"_sv };

	for (int i = 0; i < MAX_BOUNCES; i++) {
		String category = Format().format("Bounce {}"_sv, i);

		event_desc_trace              [i] = CUDAEvent::Desc { display_order, category, "Trace"_sv };
		event_desc_sort               [i] = CUDAEvent::Desc { display_order, category, "Sort"_sv };
		event_desc_material_diffuse   [i] = CUDAEvent::Desc { display_order, category, "Diffuse"_sv };
		event_desc_material_plastic   [i] = CUDAEvent::Desc { display_order, category, "Plastic"_sv };
		event_desc_material_dielectric[i] = CUDAEvent::Desc { display_order, category, "Dielectric"_sv };
		event_desc_material_conductor [i] = CUDAEvent::Desc { display_order, category, "Conductor"_sv };
		event_desc_shadow_trace       [i] = CUDAEvent::Desc { display_order, category, "Shadow"_sv };

		display_order++;
	}

	event_desc_svgf_reproject = CUDAEvent::Desc { display_order, "SVGF"_sv, "Reproject"_sv };
	event_desc_svgf_variance  = CUDAEvent::Desc { display_order, "SVGF"_sv, "Variance"_sv };

	for (int i = 0; i < MAX_ATROUS_ITERATIONS; i++) {
		event_desc_svgf_atrous[i] = CUDAEvent::Desc { display_order, "SVGF"_sv, Format().format("A Trous {}"_sv, i) };
	}
	event_desc_svgf_finalize = CUDAEvent::Desc { display_order++, "SVGF"_sv, "Finalize"_sv };

	event_desc_taa         = CUDAEvent::Desc { display_order, "Post"_sv, "TAA"_sv };
	event_desc_reconstruct = CUDAEvent::Desc { display_order, "Post"_sv, "Reconstruct"_sv };
	event_desc_accumulate  = CUDAEvent::Desc { display_order, "Post"_sv, "Accumulate"_sv };

	event_desc_end = CUDAEvent::Desc { ++display_order, "END"_sv, "END"_sv };
}

#include "Core/Timer.h"

void Pathtracer::init_luts() {
	struct KullaContyLUT {
		CUarray lut_directional_albedo;
		CUarray lut_albedo;
	};

	auto create_lut_dielectric = [this](bool entering_material) -> KullaContyLUT {
		CUarray array_lut_directional_albedo = CUDAMemory::create_array(LUT_DIELECTRIC_DIM_IOR, LUT_DIELECTRIC_DIM_ROUGHNESS, LUT_DIELECTRIC_DIM_COS_THETA, 1, CUarray_format::CU_AD_FORMAT_FLOAT);
		CUarray array_lut_albedo             = CUDAMemory::create_array(LUT_DIELECTRIC_DIM_IOR, LUT_DIELECTRIC_DIM_ROUGHNESS,                               1, CUarray_format::CU_AD_FORMAT_FLOAT);

		CUsurfObject surf_lut_directional_albedo = CUDAMemory::create_surface(array_lut_directional_albedo);
		CUsurfObject surf_lut_albedo             = CUDAMemory::create_surface(array_lut_albedo);

		kernel_integrate_dielectric.execute(entering_material, surf_lut_directional_albedo);
		kernel_average_dielectric  .execute(surf_lut_directional_albedo, surf_lut_albedo);

		CUDAMemory::free_surface(surf_lut_directional_albedo);
		CUDAMemory::free_surface(surf_lut_albedo);

		return KullaContyLUT { array_lut_directional_albedo, array_lut_albedo };
	};
	KullaContyLUT lut_dielectric_enter = create_lut_dielectric(true);
	KullaContyLUT lut_dielectric_leave = create_lut_dielectric(false);

	lut_dielectric_directional_albedo_enter.init(lut_dielectric_enter.lut_directional_albedo);
	lut_dielectric_directional_albedo_leave.init(lut_dielectric_leave.lut_directional_albedo);
	lut_dielectric_albedo_enter            .init(lut_dielectric_enter.lut_albedo);
	lut_dielectric_albedo_leave            .init(lut_dielectric_leave.lut_albedo);

	cuda_module.get_global("lut_dielectric_directional_albedo_enter").set_value(lut_dielectric_directional_albedo_enter.texture);
	cuda_module.get_global("lut_dielectric_directional_albedo_leave").set_value(lut_dielectric_directional_albedo_leave.texture);
	cuda_module.get_global("lut_dielectric_albedo_enter")            .set_value(lut_dielectric_albedo_enter.texture);
	cuda_module.get_global("lut_dielectric_albedo_leave")            .set_value(lut_dielectric_albedo_leave.texture);

	auto create_lut_conductor = [this]() -> KullaContyLUT {
		CUarray array_lut_directional_albedo = CUDAMemory::create_array(LUT_CONDUCTOR_DIM_ROUGHNESS, LUT_CONDUCTOR_DIM_COS_THETA, 1, CUarray_format::CU_AD_FORMAT_FLOAT);
		CUarray array_lut_albedo             = CUDAMemory::create_array(LUT_CONDUCTOR_DIM_ROUGHNESS, 1,                           1, CUarray_format::CU_AD_FORMAT_FLOAT);

		CUDAMemory::Ptr<float> ptr_lut_directional_albedo = CUDAMemory::malloc<float>(LUT_CONDUCTOR_DIM_ROUGHNESS * LUT_CONDUCTOR_DIM_COS_THETA);
		CUDAMemory::Ptr<float> ptr_lut_albedo             = CUDAMemory::malloc<float>(LUT_CONDUCTOR_DIM_ROUGHNESS);

		kernel_integrate_conductor.execute(ptr_lut_directional_albedo);

		CUDACALL(cuStreamSynchronize(nullptr));

		kernel_average_conductor  .execute(ptr_lut_directional_albedo, ptr_lut_albedo);
		CUDACALL(cuStreamSynchronize(nullptr));

		CUDAMemory::copy_array(array_lut_directional_albedo, LUT_CONDUCTOR_DIM_ROUGHNESS * sizeof(float), LUT_CONDUCTOR_DIM_COS_THETA, ptr_lut_directional_albedo.ptr);
		CUDAMemory::copy_array(array_lut_albedo,             LUT_CONDUCTOR_DIM_ROUGHNESS * sizeof(float), 1,                           ptr_lut_albedo.ptr);

		CUDAMemory::free(ptr_lut_directional_albedo);
		CUDAMemory::free(ptr_lut_albedo);

		return KullaContyLUT { array_lut_directional_albedo, array_lut_albedo };
	};
	KullaContyLUT lut_conductor = create_lut_conductor();

	lut_conductor_directional_albedo.init(lut_conductor.lut_directional_albedo);
	lut_conductor_albedo            .init(lut_conductor.lut_albedo);

	cuda_module.get_global("lut_conductor_directional_albedo").set_value(lut_conductor_directional_albedo.texture);
	cuda_module.get_global("lut_conductor_albedo")            .set_value(lut_conductor_albedo.texture);
}

void Pathtracer::free_luts() {
	lut_dielectric_directional_albedo_enter.free();
	lut_dielectric_directional_albedo_leave.free();
	lut_dielectric_albedo_enter            .free();
	lut_dielectric_albedo_leave            .free();
	lut_conductor_directional_albedo.free();
}

void Pathtracer::resize_init(unsigned frame_buffer_handle, int width, int height) {
	screen_width  = width;
	screen_height = height;
	screen_pitch  = Math::round_up(width, WARP_SIZE);

	pixel_count = width * height;

	cuda_module.get_global("screen_width") .set_value(screen_width);
	cuda_module.get_global("screen_pitch") .set_value(screen_pitch);
	cuda_module.get_global("screen_height").set_value(screen_height);

	// Create Frame Buffers
	init_aovs();
	aov_enable(AOVType::RADIANCE);

	// Set Accumulator to a CUDA resource mapping of the GL frame buffer texture
	resource_accumulator = CUDAMemory::resource_register(frame_buffer_handle, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
	surf_accumulator     = CUDAMemory::create_surface(CUDAMemory::resource_get_array(resource_accumulator));
	cuda_module.get_global("accumulator").set_value(surf_accumulator);

	// Set Grid dimensions for screen size dependent Kernels
	kernel_svgf_reproject.set_grid_dim(screen_pitch / kernel_svgf_reproject.block_dim_x, Math::divide_round_up(height, kernel_svgf_reproject.block_dim_y), 1);
	kernel_svgf_variance .set_grid_dim(screen_pitch / kernel_svgf_variance .block_dim_x, Math::divide_round_up(height, kernel_svgf_variance .block_dim_y), 1);
	kernel_svgf_atrous   .set_grid_dim(screen_pitch / kernel_svgf_atrous   .block_dim_x, Math::divide_round_up(height, kernel_svgf_atrous   .block_dim_y), 1);
	kernel_svgf_finalize .set_grid_dim(screen_pitch / kernel_svgf_finalize .block_dim_x, Math::divide_round_up(height, kernel_svgf_finalize .block_dim_y), 1);
	kernel_taa           .set_grid_dim(screen_pitch / kernel_taa           .block_dim_x, Math::divide_round_up(height, kernel_taa           .block_dim_y), 1);
	kernel_taa_finalize  .set_grid_dim(screen_pitch / kernel_taa_finalize  .block_dim_x, Math::divide_round_up(height, kernel_taa_finalize  .block_dim_y), 1);
	kernel_accumulate    .set_grid_dim(screen_pitch / kernel_accumulate    .block_dim_x, Math::divide_round_up(height, kernel_accumulate    .block_dim_y), 1);

	kernel_generate           .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_generate           .block_dim_x), 1, 1);
	kernel_sort               .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_sort               .block_dim_x), 1, 1);
	kernel_material_diffuse   .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_diffuse   .block_dim_x), 1, 1);
	kernel_material_plastic   .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_plastic   .block_dim_x), 1, 1);
	kernel_material_dielectric.set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_dielectric.block_dim_x), 1, 1);
	kernel_material_conductor .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_conductor .block_dim_x), 1, 1);

	scene.camera.resize(width, height);
	invalidated_camera = true;

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(Math::min(BATCH_SIZE, pixel_count));
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	sample_index = 0;

	if (gpu_config.enable_svgf) svgf_init();
}

void Pathtracer::resize_free() {
	CUDACALL(cuStreamSynchronize(memory_stream));

	free_aovs();

	CUDAMemory::resource_unregister(resource_accumulator);
	CUDAMemory::free_surface(surf_accumulator);

	if (gpu_config.enable_svgf) {
		svgf_free();
	}
}

void Pathtracer::svgf_init() {
	// GBuffers
	array_gbuffer_normal_and_depth        = CUDAMemory::create_array(screen_pitch, screen_height, 4, CU_AD_FORMAT_FLOAT);
	array_gbuffer_mesh_id_and_triangle_id = CUDAMemory::create_array(screen_pitch, screen_height, 2, CU_AD_FORMAT_SIGNED_INT32);
	array_gbuffer_screen_position_prev    = CUDAMemory::create_array(screen_pitch, screen_height, 2, CU_AD_FORMAT_FLOAT);

	surf_gbuffer_normal_and_depth        = CUDAMemory::create_surface(array_gbuffer_normal_and_depth);
	surf_gbuffer_mesh_id_and_triangle_id = CUDAMemory::create_surface(array_gbuffer_mesh_id_and_triangle_id);
	surf_gbuffer_screen_position_prev    = CUDAMemory::create_surface(array_gbuffer_screen_position_prev);

	cuda_module.get_global("gbuffer_normal_and_depth")       .set_value(surf_gbuffer_normal_and_depth);
	cuda_module.get_global("gbuffer_mesh_id_and_triangle_id").set_value(surf_gbuffer_mesh_id_and_triangle_id);
	cuda_module.get_global("gbuffer_screen_position_prev")   .set_value(surf_gbuffer_screen_position_prev);

	// Frame Buffers
	aov_enable(AOVType::RADIANCE_DIRECT);
	aov_enable(AOVType::RADIANCE_INDIRECT);
	aov_enable(AOVType::ALBEDO);

	ptr_frame_buffer_moment = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	cuda_module.get_global("frame_buffer_moment").set_value(ptr_frame_buffer_moment);

	// History Buffers
	ptr_history_length           = CUDAMemory::malloc<int>   (screen_pitch * screen_height);
	ptr_history_direct           = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_history_indirect         = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_history_moment           = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_history_normal_and_depth = CUDAMemory::malloc<float4>(screen_pitch * screen_height);

	cuda_module.get_global("history_length")          .set_value(ptr_history_length);
	cuda_module.get_global("history_direct")          .set_value(ptr_history_direct);
	cuda_module.get_global("history_indirect")        .set_value(ptr_history_indirect);
	cuda_module.get_global("history_moment")          .set_value(ptr_history_moment);
	cuda_module.get_global("history_normal_and_depth").set_value(ptr_history_normal_and_depth);

	// Frame Buffers for Temporal Anti-Aliasing
	ptr_taa_frame_prev = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_taa_frame_curr = CUDAMemory::malloc<float4>(screen_pitch * screen_height);

	cuda_module.get_global("taa_frame_prev").set_value(ptr_taa_frame_prev);
	cuda_module.get_global("taa_frame_curr").set_value(ptr_taa_frame_curr);
}

void Pathtracer::svgf_free() {
	CUDAMemory::free_array(array_gbuffer_normal_and_depth);
	CUDAMemory::free_array(array_gbuffer_mesh_id_and_triangle_id);
	CUDAMemory::free_array(array_gbuffer_screen_position_prev);

	CUDAMemory::free_surface(surf_gbuffer_normal_and_depth);
	CUDAMemory::free_surface(surf_gbuffer_mesh_id_and_triangle_id);
	CUDAMemory::free_surface(surf_gbuffer_screen_position_prev);

	aov_disable(AOVType::RADIANCE_DIRECT);
	aov_disable(AOVType::RADIANCE_INDIRECT);
	aov_disable(AOVType::ALBEDO);

	CUDAMemory::free(ptr_frame_buffer_moment);

	CUDAMemory::free(ptr_history_length);
	CUDAMemory::free(ptr_history_direct);
	CUDAMemory::free(ptr_history_indirect);
	CUDAMemory::free(ptr_history_moment);
	CUDAMemory::free(ptr_history_normal_and_depth);

	CUDAMemory::free(ptr_taa_frame_prev);
	CUDAMemory::free(ptr_taa_frame_curr);
}

void Pathtracer::calc_light_power(Allocator * frame_allocator) {
	(void)frame_allocator;
	// Build one record for every emissive triangle. This supports both the
	// legacy mesh-wide material and a glTF per-triangle material assignment.
	light_triangle_data.clear();
	for (int mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
		Mesh & mesh = scene.meshes[mesh_index];
		const MeshData & mesh_data = scene.asset_manager.get_mesh_data(mesh.mesh_data_handle);
		mesh.light.weight = 0.0f;

		for (int triangle_index = 0; triangle_index < mesh_data.triangles.size(); ++triangle_index) {
			int material_id = mesh.material_handle.handle;
			if (mesh_data.material_ids.size() == mesh_data.triangles.size()) {
				material_id = mesh_data.material_ids[triangle_index];
			}
			const Material & material = scene.asset_manager.get_material({ material_id });
			if (!material.is_light()) continue;

			const Triangle & triangle = mesh_data.triangles[triangle_index];
			const float area = 0.5f * Vector3::length(Vector3::cross(
				triangle.position_1 - triangle.position_0,
				triangle.position_2 - triangle.position_0));
			const double weight = Math::luminance(material.emission) * area * mesh.scale * mesh.scale;
			if (weight <= 0.0) continue;

			light_triangle_data.push_back({
				reverse_indices[mesh_data_triangle_offsets[mesh.mesh_data_handle.handle] + triangle_index],
				mesh_index,
				weight });
		}
	}
}

// Construct Top Level Acceleration Structure (TLAS) over the Meshes in the Scene
void Pathtracer::calc_light_mesh_weights() {
	if (ptr_light_triangle_indices.ptr) CUDAMemory::free(ptr_light_triangle_indices);
	if (ptr_light_triangle_cumulative_probability.ptr) CUDAMemory::free(ptr_light_triangle_cumulative_probability);
	if (ptr_light_triangle_mesh_indices.ptr) CUDAMemory::free(ptr_light_triangle_mesh_indices);

	if (light_triangle_data.empty()) {
		cuda_module.get_global("light_triangle_count").set_value_async(0, memory_stream);
		global_lights_total_weight.set_value_async(0.0f, memory_stream);
		return;
	}

	std::vector<int> transform_indices(scene.meshes.size(), INVALID);
	for (int transform_index = 0; transform_index < tlas->indices.size(); ++transform_index) {
		transform_indices[tlas->indices[transform_index]] = transform_index;
	}

	std::vector<int> triangle_indices;
	std::vector<int> triangle_mesh_indices;
	std::vector<float> cumulative_probability;
	triangle_indices.reserve(light_triangle_data.size());
	triangle_mesh_indices.reserve(light_triangle_data.size());
	cumulative_probability.reserve(light_triangle_data.size());

	double total_weight = 0.0;
	for (const LightTriangleData & light : light_triangle_data) {
		const int transform_index = transform_indices[light.scene_mesh_index];
		if (transform_index == INVALID) continue;
		total_weight += light.weight;
		triangle_indices.push_back(light.triangle_index);
		triangle_mesh_indices.push_back(transform_index);
		cumulative_probability.push_back(float(total_weight));
	}

	if (triangle_indices.empty() || total_weight <= 0.0) {
		cuda_module.get_global("light_triangle_count").set_value_async(0, memory_stream);
		global_lights_total_weight.set_value_async(0.0f, memory_stream);
		return;
	}
	for (float & value : cumulative_probability) value /= float(total_weight);

	ptr_light_triangle_indices = CUDAMemory::malloc(triangle_indices.data(), triangle_indices.size());
	ptr_light_triangle_cumulative_probability = CUDAMemory::malloc(cumulative_probability.data(), cumulative_probability.size());
	ptr_light_triangle_mesh_indices = CUDAMemory::malloc(triangle_mesh_indices.data(), triangle_mesh_indices.size());
	cuda_module.get_global("light_triangle_indices").set_value_async(ptr_light_triangle_indices, memory_stream);
	cuda_module.get_global("light_triangle_cumulative_probability").set_value_async(ptr_light_triangle_cumulative_probability, memory_stream);
	cuda_module.get_global("light_triangle_mesh_indices").set_value_async(ptr_light_triangle_mesh_indices, memory_stream);
	cuda_module.get_global("light_triangle_count").set_value_async(int(triangle_indices.size()), memory_stream);
	global_lights_total_weight.set_value_async(float(total_weight), memory_stream);
}

void Pathtracer::update(float delta, Allocator * frame_allocator) {
	if (invalidated_sky) {
		invalidated_sky = false;
		global_sky_scale.set_value_async(scene.sky.scale, memory_stream);

		sample_index = 0;
	}

	if (invalidated_materials) {
		const Array<Material> & materials = scene.asset_manager.materials;

		Array<Material::Type> cuda_material_types(materials.size(), frame_allocator);
		Array<CUDAMaterial>   cuda_materials     (materials.size(), frame_allocator);

		for (int i = 0; i < materials.size(); i++) {
			const Material & material = materials[i];

			cuda_material_types[i] = material.type;

			switch (material.type) {
				case Material::Type::LIGHT: {
					cuda_materials[i].light.emission = material.emission;
					break;
				}
				case Material::Type::DIFFUSE: {
					cuda_materials[i].diffuse.diffuse    = material.diffuse;
					cuda_materials[i].diffuse.texture_id = material.texture_handle.handle;
					break;
				}
				case Material::Type::PLASTIC: {
					cuda_materials[i].plastic.diffuse          = material.diffuse;
					cuda_materials[i].plastic.texture_id       = material.texture_handle.handle;
					cuda_materials[i].plastic.linear_roughness = material.linear_roughness;
					break;
				}
				case Material::Type::DIELECTRIC: {
					cuda_materials[i].dielectric.medium_id        = material.medium_handle.handle;
					cuda_materials[i].dielectric.ior              = Math::max(material.index_of_refraction, 1.0001f);
					cuda_materials[i].dielectric.linear_roughness = material.linear_roughness;
					break;
				}
				case Material::Type::CONDUCTOR: {
					cuda_materials[i].conductor.eta              = material.eta;
					cuda_materials[i].conductor.linear_roughness = material.linear_roughness;
					cuda_materials[i].conductor.k                = material.k;
					break;
				}
				default: ASSERT_UNREACHABLE();
			}
		}

		CUDAMemory::memcpy_async(ptr_material_types, cuda_material_types.data(), materials.size(), memory_stream);
		CUDAMemory::memcpy_async(ptr_materials,      cuda_materials     .data(), materials.size(), memory_stream);

		bool had_diffuse    = scene.has_diffuse;
		bool had_plastic    = scene.has_plastic;
		bool had_dielectric = scene.has_dielectric;
		bool had_conductor  = scene.has_conductor;
		bool had_lights     = scene.has_lights;

		scene.check_materials();

		bool material_types_changed =
			(had_diffuse    ^ scene.has_diffuse) |
			(had_plastic    ^ scene.has_plastic) |
			(had_dielectric ^ scene.has_dielectric) |
			(had_conductor  ^ scene.has_conductor);

		if (material_types_changed) {
			int num_different_materials =
				int(scene.has_diffuse) +
				int(scene.has_plastic) +
				int(scene.has_dielectric) +
				int(scene.has_conductor);
			// 2 different Materials can share the same MaterialBuffer, one growing left to right, one growing right to left
			int num_material_buffers_needed = Math::divide_round_up(num_different_materials, 2);

			if (num_material_buffers_needed < material_ray_buffers.size()) {
				// Free MaterialBuffers that are not needed
				for (int i = num_material_buffers_needed; i < material_ray_buffers.size(); i++) {
					material_ray_buffers[i].free();
				}
			} else if (num_material_buffers_needed > material_ray_buffers.size()) {
				// Allocate new required MaterialBuffers
				for (int i = material_ray_buffers.size(); i < num_material_buffers_needed; i++) {
					material_ray_buffers.emplace_back().init(BATCH_SIZE);
				}
			}

			if (ptr_material_ray_buffers.ptr) {
				CUDAMemory::free(ptr_material_ray_buffers);
			}
			ptr_material_ray_buffers = CUDAMemory::malloc(material_ray_buffers);

			int material_buffer_index = 0;

			auto set_material_buffer = [&](const char * global_name) {
				// Two consecutive buffers can share the same underlying allocation, every odd buffer is reversed
				CUDAMemory::Ptr<MaterialBuffer> ptr_buffer = ptr_material_ray_buffers + material_buffer_index / 2;
				bool reversed = material_buffer_index & 1;

				ASSERT((ptr_buffer.ptr & 1) == 0);
				uintptr_t packed = uintptr_t(ptr_buffer.ptr) | uintptr_t(reversed);
				cuda_module.get_global(global_name).set_value_async(packed, memory_stream);

				material_buffer_index++;
			};
			if (scene.has_diffuse)    set_material_buffer("material_buffer_diffuse");
			if (scene.has_plastic)    set_material_buffer("material_buffer_plastic");
			if (scene.has_dielectric) set_material_buffer("material_buffer_dielectric");
			if (scene.has_conductor)  set_material_buffer("material_buffer_conductor");
		}

		// Handle (dis)appearance of Light materials
		bool lights_changed = had_lights ^ scene.has_lights;
		if (lights_changed) {
			if (scene.has_lights) {
				ray_buffer_shadow.init(BATCH_SIZE);

				invalidated_scene = true;
			} else {
				ray_buffer_shadow.free();

				if (ptr_light_triangle_indices.ptr) CUDAMemory::free(ptr_light_triangle_indices);
				if (ptr_light_triangle_cumulative_probability.ptr) CUDAMemory::free(ptr_light_triangle_cumulative_probability);
				if (ptr_light_triangle_mesh_indices.ptr) CUDAMemory::free(ptr_light_triangle_mesh_indices);

				global_lights_total_weight.set_value_async(0.0f, memory_stream);

				for (int i = 0; i < scene.meshes.size(); i++) {
					scene.meshes[i].light.weight = 0.0f;
				}
			}

			global_ray_buffer_shadow.set_value_async(ray_buffer_shadow, memory_stream);
		}

		if (had_lights) {
			if (ptr_light_triangle_indices.ptr) CUDAMemory::free(ptr_light_triangle_indices);
			if (ptr_light_triangle_cumulative_probability.ptr) CUDAMemory::free(ptr_light_triangle_cumulative_probability);
			if (ptr_light_triangle_mesh_indices.ptr) CUDAMemory::free(ptr_light_triangle_mesh_indices);
		}
		if (scene.has_lights) {
			calc_light_power(frame_allocator);
		}

		sample_index = 0;
		invalidated_materials = false;
	}

	if (invalidated_mediums) {
		size_t medium_count = scene.asset_manager.media.size();
		if (medium_count > 0) {
			Array<CUDAMedium> cuda_mediums(medium_count, frame_allocator);

			for (size_t i = 0; i < medium_count; i++) {
				const Medium & medium = scene.asset_manager.media[i];
				medium.to_sigmas(cuda_mediums[i].sigma_a, cuda_mediums[i].sigma_s);
				cuda_mediums[i].g = medium.g;
			}

			CUDAMemory::memcpy_async(ptr_media, cuda_mediums.data(), medium_count, memory_stream);
		}

		sample_index = 0;
		invalidated_mediums = false;
	}

	// Save this here, TLAS will be updated in Integrator::update if invalidated_scene == true,
	// calc_light_mesh_weights() will need to be called AFTER the TLAS has been updated.
	bool invalidated_light_mesh_weights = invalidated_scene;

	if (gpu_config.enable_svgf) {
		struct SVGFData {
			alignas(16) Matrix4 view_projection;
			alignas(16) Matrix4 view_projection_prev;
		} svgf_data;

		svgf_data.view_projection      = scene.camera.view_projection;
		svgf_data.view_projection_prev = scene.camera.view_projection_prev;

		global_svgf_data.set_value_async(svgf_data, memory_stream);
	}

	if (invalidated_aovs) {
		if (gpu_config.enable_svgf && !aov_is_enabled(AOVType::ALBEDO)) {
			aov_enable(AOVType::ALBEDO); // SVGF cannot function without ALBEDO
		}
	}

	Integrator::update(delta, frame_allocator);

	if (invalidated_light_mesh_weights) {
		calc_light_mesh_weights();

		// If SVGF is enabled we can handle Scene updates using reprojection,
		// otherwise 'frames_accumulated' needs to be reset in order to avoid ghosting
		if (!gpu_config.enable_svgf) {
			sample_index = 0;
		}
	}
}

static int get_child_node_index(const BVHNode8& node, int child_slot) {
	unsigned bit = 1u << child_slot;

	// Leaf child는 triangle을 가리킬 뿐 BVH8 child node가 없음
	if ((unsigned(node.imask) & bit) == 0) {
		return INVALID;
	}

	int internal_children_before = 0;

	for (int i = 0; i < child_slot; i++) {
		if (unsigned(node.imask) & (1u << i)) {
			internal_children_before++;
		}
	}

	return int(node.base_index_child) + internal_children_before;
}

static void get_child_aabb(const BVHNode8& node, int child_slot, float bounds[6]) {
	// e stores IEEE-754 exponent bits for a power-of-two quantization scale.
	const Vector3 scale(
		std::ldexp(1.0f, int(node.e[0]) - 127),
		std::ldexp(1.0f, int(node.e[1]) - 127),
		std::ldexp(1.0f, int(node.e[2]) - 127)
	);

	bounds[0] = node.p.x + float(node.quantized_min_x[child_slot]) * scale.x;
	bounds[1] = node.p.y + float(node.quantized_min_y[child_slot]) * scale.y;
	bounds[2] = node.p.z + float(node.quantized_min_z[child_slot]) * scale.z;
	bounds[3] = node.p.x + float(node.quantized_max_x[child_slot]) * scale.x;
	bounds[4] = node.p.y + float(node.quantized_max_y[child_slot]) * scale.y;
	bounds[5] = node.p.z + float(node.quantized_max_z[child_slot]) * scale.z;
}

static const char* get_node_type(
	size_t node_index,
	size_t tlas_node_count,
	const Array<int>& mesh_data_bvh_offsets,
	const Scene& scene
) {
	if (node_index < tlas_node_count) {
		return "TLAS";
	}

	for (int i = 0; i < scene.asset_manager.mesh_datas.size(); i++) {
		size_t begin = mesh_data_bvh_offsets[i];
		size_t end =
			begin + scene.asset_manager.mesh_datas[i].bvh->node_count();

		if (node_index >= begin && node_index < end) {
			return "BLAS";
		}
	}

	return "RESERVED";
}

void Pathtracer::render() {
	event_pool.reset();

	CUDACALL(cuStreamSynchronize(memory_stream));

	int pixels_left = pixel_count;
	int batch_size  = Math::min(BATCH_SIZE, pixel_count);

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = pixel_count - pixels_left;
		int pixel_count  = Math::min(batch_size, pixels_left);

		event_pool.record(&event_desc_primary);

		// Generate primary Rays from the current Camera orientation
		kernel_generate.execute(sample_index, pixel_offset, pixel_count);

		for (int bounce = 0; bounce < gpu_config.num_bounces; bounce++) {
			// Extend all Rays that are still alive to their next Triangle intersection
			event_pool.record(&event_desc_trace[bounce]);
			kernel_trace->execute(bounce);

			event_pool.record(&event_desc_sort[bounce]);
			kernel_sort.execute(bounce, sample_index);

			// Process the various Material types in different Kernels
			if (scene.has_diffuse) {
				event_pool.record(&event_desc_material_diffuse[bounce]);
				kernel_material_diffuse.execute(bounce, sample_index);
			}
			if (scene.has_plastic) {
				event_pool.record(&event_desc_material_plastic[bounce]);
				kernel_material_plastic.execute(bounce, sample_index);
			}
			if (scene.has_dielectric) {
				event_pool.record(&event_desc_material_dielectric[bounce]);
				kernel_material_dielectric.execute(bounce, sample_index);
			}
			if (scene.has_conductor) {
				event_pool.record(&event_desc_material_conductor[bounce]);
				kernel_material_conductor.execute(bounce, sample_index);
			}

			// Trace shadow Rays
			if (scene.has_lights && gpu_config.enable_next_event_estimation) {
				event_pool.record(&event_desc_shadow_trace[bounce]);
				kernel_trace_shadow->execute(bounce);
			}
		}

		pixels_left -= batch_size;

		if (pixels_left > 0) {
			// Set buffer sizes to appropriate pixel count for next Batch
			pinned_buffer_sizes->reset(Math::min(batch_size, pixels_left));
			global_buffer_sizes.set_value(*pinned_buffer_sizes);
		}
	}

	//jichan add
	if (counter_mode) {
		static uint64_t counter_limit = 200ull;
		if (bvh_counter_dumped + 1 == counter_limit) {
			CUDACALL(cuStreamSynchronize(nullptr));

			Array<uint64_t> host_counters(bvh_counter_count);
			CUDAMemory::memcpy<uint64_t>(host_counters.data(), ptr_bvh_counter, bvh_counter_count);

			Array<BVHNode8> host_nodes(bvh_counter_count);
			CUDAMemory::memcpy<BVHNode8>(host_nodes.data(), ptr_bvh_nodes_8, bvh_counter_count);

			Array<uint64_t> host_triangle_counters(triangle_counter_count);
			CUDAMemory::memcpy<uint64_t>(host_triangle_counters.data(), ptr_triangle_counter, triangle_counter_count);

			Array<uint64_t> host_mesh_counters(mesh_counter_count);
			CUDAMemory::memcpy<uint64_t>(host_mesh_counters.data(), ptr_mesh_counter, mesh_counter_count);

			FILE* file = nullptr;
			fopen_s(&file, "bvh8.csv", "wb");
			if (file) {
				fprintf(file, "node_index,node_type,node_count");
				for (int child_slot = 0; child_slot < 8; child_slot++) {
					fprintf(file,
						",child_index_%d,child_count_%d,"
						"child_min_x_%d,child_min_y_%d,child_min_z_%d,"
						"child_max_x_%d,child_max_y_%d,child_max_z_%d",
						child_slot, child_slot,
						child_slot, child_slot, child_slot,
						child_slot, child_slot, child_slot);
				}
				fprintf(file, "\n");

				for (size_t i = 0; i < bvh_counter_count; i++) {
					const BVHNode8& node = host_nodes[i];

					const char* node_type = get_node_type(i, tlas->node_count(), mesh_data_bvh_offsets, scene);

					if (strcmp(node_type, "RESERVED") == 0) {
						continue;
					}

					int child_indices[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
					uint64_t child_counts[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
					float child_bounds[8][6];
					for (int child_slot = 0; child_slot < 8; child_slot++) {
						for (int component = 0; component < 6; component++) child_bounds[child_slot][component] = NAN;
					}

					for (int child_slot = 0; child_slot < 8; child_slot++) {
						const unsigned meta = unsigned(node.meta[child_slot]);

						if (meta == 0) {
							continue;
						}
						get_child_aabb(node, child_slot, child_bounds[child_slot]);

						const bool is_internal = (unsigned(node.imask) & (1u << child_slot)) != 0;

						if (is_internal) {
							const int child_node_index = get_child_node_index(node, child_slot);
							child_indices[child_slot] = child_node_index;
							child_counts[child_slot] = host_counters[child_node_index];
						}
						else {
							// meta 하위 5 bit는 이 node primitive group 내 시작 offset,
							// 상위 3 bit의 unary mask는 이 child가 보유한 primitive 개수다.
							const int first_primitive = int(node.base_index_triangle) + int(meta & 0x1f);
							const unsigned primitive_mask = meta >> 5;
							int primitive_count = 0;
							for (unsigned bits = primitive_mask; bits != 0; bits &= bits - 1) {
								primitive_count++;
							}

							// A leaf child has no physical BVH8 node. Record its first primitive
							// as a negative index; for a BLAS leaf, sum all triangle test counters.
							child_indices[child_slot] = -(first_primitive + 1);
							if (strcmp(node_type, "BLAS") == 0) {
								for (int primitive_offset = 0; primitive_offset < primitive_count; primitive_offset++) {
									child_counts[child_slot] += host_triangle_counters[first_primitive + primitive_offset];
								}
							}
							else {
								for (int primitive_offset = 0; primitive_offset < primitive_count; primitive_offset++) {
									child_counts[child_slot] += host_mesh_counters[first_primitive + primitive_offset];
								}
							}
						}
					}

					fprintf(file, "%zu,%s,%llu", i, node_type, static_cast<unsigned long long>(host_counters[i]));
					for (int child_slot = 0; child_slot < 8; child_slot++) {
						const float* bounds = child_bounds[child_slot];
						fprintf(file, ",%d,%llu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",
							child_indices[child_slot], static_cast<unsigned long long>(child_counts[child_slot]),
							bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);
					}
					fprintf(file, "\n");
				}
				fclose(file);
				std::printf("Successfully wrote BVH meatadata to BVH8.csv");
			}
			counter_mode = false;
			CUDAMemory::memset_async(ptr_counter_mode, counter_mode, 1, memory_stream);
		}
		else if (bvh_counter_dumped < counter_limit) {
			bvh_counter_dumped++;
		}
	}
	
	
	if (gpu_config.enable_svgf) {
		// Temporal reprojection + integration
		event_pool.record(&event_desc_svgf_reproject);
		kernel_svgf_reproject.execute(sample_index);

		CUdeviceptr direct_in    = get_aov(AOVType::RADIANCE_DIRECT)  .framebuffer.ptr;
		CUdeviceptr indirect_in  = get_aov(AOVType::RADIANCE_INDIRECT).framebuffer.ptr;
		CUdeviceptr direct_out   = get_aov(AOVType::RADIANCE_DIRECT)  .accumulator.ptr;
		CUdeviceptr indirect_out = get_aov(AOVType::RADIANCE_INDIRECT).accumulator.ptr;

		if (gpu_config.enable_spatial_variance) {
			// Estimate Variance spatially
			event_pool.record(&event_desc_svgf_variance);
			kernel_svgf_variance.execute(direct_in, indirect_in, direct_out, indirect_out);

			Util::swap(direct_in,   direct_out);
			Util::swap(indirect_in, indirect_out);
		}

		// �-Trous Filter
		for (int i = 0; i < gpu_config.num_atrous_iterations; i++) {
			int step_size = 1 << i;

			event_pool.record(&event_desc_svgf_atrous[i]);
			kernel_svgf_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);

			// Ping-Pong the Frame Buffers
			Util::swap(direct_in,   direct_out);
			Util::swap(indirect_in, indirect_out);
		}

		event_pool.record(&event_desc_svgf_finalize);
		kernel_svgf_finalize.execute(direct_in, indirect_in);

		if (gpu_config.enable_taa) {
			event_pool.record(&event_desc_taa);

			kernel_taa         .execute(sample_index);
			kernel_taa_finalize.execute();
		}
	} else {
		event_pool.record(&event_desc_accumulate);
		kernel_accumulate.execute(float(sample_index));
	}

	event_pool.record(&event_desc_end);

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(batch_size);
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	aovs_clear_to_zero();

	// If a pixel query was previously pending, it has just been resolved in the current frame
	if (pixel_query_status == PixelQueryStatus::PENDING) {
		pixel_query_status =  PixelQueryStatus::OUTPUT_READY;
	}
}

void Pathtracer::render_gui() {
	if (ImGui::CollapsingHeader("Integrator", ImGuiTreeNodeFlags_DefaultOpen)) {
		invalidated_gpu_config |= ImGui::SliderInt("Num Bounces", &gpu_config.num_bounces, 0, MAX_BOUNCES);

		invalidated_gpu_config |= ImGui::Checkbox("NEE", &gpu_config.enable_next_event_estimation);
		invalidated_gpu_config |= ImGui::Checkbox("MIS", &gpu_config.enable_multiple_importance_sampling);

		invalidated_gpu_config |= ImGui::Checkbox("Russian Roulete", &gpu_config.enable_russian_roulette);
	}

	if (ImGui::CollapsingHeader("Auxilary AOVs", ImGuiTreeNodeFlags_DefaultOpen)) {
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::ALBEDO,   "Albedo");
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::NORMAL,   "Normal");
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::POSITION, "Position");
	}

	if (ImGui::CollapsingHeader("SVGF")) {
		if (ImGui::Checkbox("Enable", &gpu_config.enable_svgf)) {
			if (gpu_config.enable_svgf) {
				svgf_init();
			} else {
				svgf_free();
			}
			invalidated_gpu_config = true;
		}

		invalidated_gpu_config |= ImGui::Checkbox("Spatial Variance", &gpu_config.enable_spatial_variance);
		invalidated_gpu_config |= ImGui::Checkbox("TAA",              &gpu_config.enable_taa);

		invalidated_gpu_config |= ImGui::SliderInt("A Trous iterations", &gpu_config.num_atrous_iterations, 0, MAX_ATROUS_ITERATIONS);

		invalidated_gpu_config |= ImGui::SliderFloat("Alpha colour", &gpu_config.alpha_colour, 0.0f, 1.0f);
		invalidated_gpu_config |= ImGui::SliderFloat("Alpha moment", &gpu_config.alpha_moment, 0.0f, 1.0f);
	}
}
