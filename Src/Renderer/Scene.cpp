#include "Scene.h"

#include <ctype.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdio.h>
#include <vector>

#include "Core/IO.h"

#include "Assets/OBJLoader.h"
#include "Assets/PLYLoader.h"
#include "Assets/GLTFLoader.h"
#include "Assets/Mitsuba/MitsubaLoader.h"
#include "Assets/json.hpp"

#include "Material.h"

#include "Util/Util.h"
#include "Util/StringUtil.h"

namespace {

using nlohmann::json;

bool read_vec3(const json & value, Vector3 & result) {
	if (!value.is_array() || value.size() != 3) return false;
	if (!value[0].is_number() || !value[1].is_number() || !value[2].is_number()) return false;
	result = Vector3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
	return true;
}

bool read_quaternion(const json & value, Quaternion & result) {
	if (!value.is_array() || value.size() != 4) return false;
	if (!value[0].is_number() || !value[1].is_number() || !value[2].is_number() || !value[3].is_number()) return false;
	result = Quaternion(value[0].get<float>(), value[1].get<float>(), value[2].get<float>(), value[3].get<float>());
	const float length = Quaternion::length(result);
	if (length == 0.0f) return false;
	result = Quaternion::normalize(result);
	return true;
}

bool has_member(const json & value, const char * name) {
	return value.is_object() && value.find(name) != value.end();
}

void load_scene_file(Scene & scene, const String & scene_filename);

void load_scene_manifest(Scene & scene, const String & manifest_filename) {
	std::ifstream stream(manifest_filename.data());
	if (!stream) {
		IO::print("ERROR: Cannot open scene manifest '{}'\n"_sv, manifest_filename);
		IO::exit(1);
	}

	json manifest;
	try {
		stream >> manifest;
	} catch (const std::exception & error) {
		IO::print("ERROR: Cannot parse scene manifest '{}': {}\n"_sv, manifest_filename, error.what());
		IO::exit(1);
	}

	if (!has_member(manifest, "filenames") || !manifest["filenames"].is_array()) {
		IO::print("ERROR: Scene manifest '{}' has no 'filenames' array\n"_sv, manifest_filename);
		IO::exit(1);
	}

	const std::filesystem::path base_directory = std::filesystem::path(manifest_filename.data()).parent_path();
	for (const json & filename : manifest["filenames"]) {
		if (!filename.is_string()) {
			IO::print("ERROR: Scene manifest '{}' has a non-string filename\n"_sv, manifest_filename);
			IO::exit(1);
		}
		const std::string resolved_filename = (base_directory / filename.get<std::string>()).string();
		load_scene_file(scene, String(resolved_filename.c_str(), scene.allocator));
	}

	if (has_member(manifest, "camera") && manifest["camera"].is_array()) {
		for (const json & value : manifest["camera"]) {
			Vector3 position, rotation_degrees;
			if (!value.is_object() || !has_member(value, "position") || !has_member(value, "rotation") ||
				!read_vec3(value["position"], position) || !read_vec3(value["rotation"], rotation_degrees)) {
				IO::print("WARNING: Ignoring invalid camera preset in '{}'\n"_sv, manifest_filename);
				continue;
			}
			const Quaternion rotation = Quaternion::from_euler(
				Math::deg_to_rad(rotation_degrees.z), Math::deg_to_rad(rotation_degrees.y), Math::deg_to_rad(rotation_degrees.x));
			scene.camera_presets.push_back({ position, rotation });
		}
	}

	if (has_member(manifest, "counter_camera_views") && manifest["counter_camera_views"].is_array()) {
		for (const json & value : manifest["counter_camera_views"]) {
			CameraPreset view;
			if (!value.is_object() || !has_member(value, "position") || !has_member(value, "rotation") ||
				!read_vec3(value["position"], view.position) || !read_quaternion(value["rotation"], view.rotation)) {
				IO::print("WARNING: Ignoring invalid counter camera view in '{}'\n"_sv, manifest_filename);
				continue;
			}
			scene.counter_camera_views.push_back(view);
		}
	}
	if (has_member(manifest, "counter_camera_target")) {
		if (read_vec3(manifest["counter_camera_target"], scene.counter_camera_target)) {
			scene.has_counter_camera_target = true;
		} else {
			IO::print("WARNING: Ignoring invalid counter camera target in '{}'\n"_sv, manifest_filename);
		}
	}

	if (has_member(manifest, "camera_keyframes") && manifest["camera_keyframes"].is_array()) {
		for (const json & value : manifest["camera_keyframes"]) {
			CameraKeyframe keyframe;
			if (!value.is_object() || !has_member(value, "timesec") || !has_member(value, "position") || !has_member(value, "lookat") ||
				!value["timesec"].is_number() || !read_vec3(value["position"], keyframe.position) ||
				!read_vec3(value["lookat"], keyframe.lookat)) {
				IO::print("WARNING: Ignoring invalid camera keyframe in '{}'\n"_sv, manifest_filename);
				continue;
			}
			keyframe.time = value["timesec"].get<float>();
			keyframe.phantom = has_member(value, "phantom") && value["phantom"].is_boolean() && value["phantom"].get<bool>();
			scene.camera_keyframes.push_back(keyframe);
		}
	}
}

void load_scene_file(Scene & scene, const String & scene_filename) {
	StringView file_extension = Util::get_file_extension(scene_filename.view());
	if (file_extension.is_empty()) {
		IO::print("ERROR: File '{}' has no file extension, cannot deduce file format!\n"_sv, scene_filename);
		IO::exit(1);
	}
	if (file_extension == "obj") {
		scene.add_mesh(scene_filename, scene.asset_manager.add_mesh_data(scene_filename, OBJLoader::load));
	} else if (file_extension == "ply") {
		scene.add_mesh(scene_filename, scene.asset_manager.add_mesh_data(scene_filename, PLYLoader::load));
	} else if (file_extension == "gltf" || file_extension == "glb") {
		GLTFLoader::load_scene(scene_filename, scene);
	} else if (file_extension == "xml") {
		MitsubaLoader::load(scene_filename, scene.allocator, scene);
	} else if (file_extension == "json") {
		load_scene_manifest(scene, scene_filename);
	} else {
		IO::print("ERROR: '{}' file format is not supported!\n"_sv, file_extension);
		IO::exit(1);
	}
}

Quaternion make_world_up_camera_rotation(const Vector3 & direction) {
	const Vector3 reference_up = fabsf(direction.y) < 0.999f
		? Vector3(0.0f, 1.0f, 0.0f) : Vector3(1.0f, 0.0f, 0.0f);
	return Quaternion::normalize(Quaternion::look_rotation(-direction, reference_up));
}

} // namespace

Scene::Scene(Allocator * allocator) : allocator(allocator), asset_manager(allocator), camera(Math::deg_to_rad(85.0f)), meshes(allocator), punctual_lights(allocator), directional_lights(allocator), camera_presets(allocator), counter_camera_views(allocator), uniform_counter_quaternions(allocator), uniform_fps_quaternions(allocator), camera_keyframes(allocator) {
	LinearAllocator<MEGABYTES(4)> load_allocator;

	for (int i = 0; i < cpu_config.scene_filenames.size(); i++) {
		load_scene_file(*this, cpu_config.scene_filenames[i]);
	}

	set_camera_preset(0);

	sky.load(cpu_config.sky_filename);
}

Mesh & Scene::add_mesh(String name, Handle<MeshData> mesh_data_handle, Handle<Material> material_handle) {
	return meshes.emplace_back(std::move(name), mesh_data_handle, material_handle);
}

bool Scene::set_camera_preset(size_t preset_index) {
	if (preset_index >= camera_presets.size()) return false;
	camera.position = camera_presets[preset_index].position;
	camera.rotation = camera_presets[preset_index].rotation;
	return true;
}

bool Scene::set_counter_camera_sample(uint64_t sample_index, uint64_t sample_count) {
	if (counter_camera_views.size() == 0 || sample_count == 0) return false;
	auto set_counter_rotation = [this](const Quaternion & fallback) {
		if (!has_counter_camera_target) {
			camera.rotation = fallback;
			return;
		}

		const Vector3 look_direction = camera.position - counter_camera_target;
		if (Vector3::length_squared(look_direction) == 0.0f) {
			camera.rotation = fallback;
			return;
		}

		const Vector3 normalized_direction = Vector3::normalize(look_direction);
		const Vector3 world_up(0.0f, 1.0f, 0.0f);
		const Vector3 alternate_up(0.0f, 0.0f, 1.0f);
		const Vector3 up = fabsf(Vector3::dot(normalized_direction, world_up)) < 0.999f
			? world_up : alternate_up;
		camera.rotation = Quaternion::look_rotation(look_direction, up);
	};
	const uint64_t clamped_sample = Math::min(sample_index, sample_count - 1);
	if (counter_camera_views.size() == 1 || sample_count == 1) {
		camera.position = counter_camera_views[0].position;
		set_counter_rotation(counter_camera_views[0].rotation);
		if (clamped_sample == 0) {
			IO::print("Counter camera view 1/1 (frame 1/{})\n"_sv, sample_count);
		}
		return true;
	}

	const float progress = float(clamped_sample) / float(sample_count - 1);
	const float scaled_segment = progress * float(counter_camera_views.size() - 1);
	const size_t segment = Math::min(
		counter_camera_views.size() - 2,
		static_cast<size_t>(scaled_segment));
	const float t = Math::clamp(scaled_segment - float(segment), 0.0f, 1.0f);

	if (clamped_sample == 0) {
		IO::print("Counter camera segment 1/{}: view 1 -> 2 (frame 1/{})\n"_sv,
			counter_camera_views.size() - 1, sample_count);
	} else {
		const float previous_progress = float(clamped_sample - 1) / float(sample_count - 1);
		const size_t previous_segment = Math::min(
			counter_camera_views.size() - 2,
			static_cast<size_t>(previous_progress * float(counter_camera_views.size() - 1)));
		if (segment != previous_segment) {
			IO::print("Counter camera segment {}/{}: view {} -> {} (frame {}/{})\n"_sv,
				segment + 1, counter_camera_views.size() - 1,
				segment + 1, segment + 2, clamped_sample + 1, sample_count);
		}
	}

	const CameraPreset & a = counter_camera_views[segment];
	const CameraPreset & b = counter_camera_views[segment + 1];
	camera.position = a.position * (1.0f - t) + b.position * t;

	Quaternion rotation_b = b.rotation;
	const float rotation_dot =
		a.rotation.x * rotation_b.x + a.rotation.y * rotation_b.y +
		a.rotation.z * rotation_b.z + a.rotation.w * rotation_b.w;
	if (rotation_dot < 0.0f) {
		rotation_b = Quaternion(-rotation_b.x, -rotation_b.y, -rotation_b.z, -rotation_b.w);
	}
	set_counter_rotation(Quaternion::nlerp(a.rotation, rotation_b, t));
	return true;
}

bool Scene::set_uniform_quaternion_counter_sample(
	uint64_t render_index, uint64_t position_count, uint64_t quaternion_count,
	CounterQuaternionDistribution distribution, float horizon_power, float horizon_mix,
	CounterQuaternionRollMode roll_mode) {
	if (position_count == 0 || quaternion_count == 0 || camera_keyframes.size() == 0) return false;

	horizon_power = Math::max(horizon_power, 0.0f);
	horizon_mix = Math::clamp(horizon_mix, 0.0f, 1.0f);
	const bool sampling_changed =
		cached_counter_quaternion_distribution != distribution ||
		cached_counter_quaternion_horizon_power != horizon_power ||
		cached_counter_quaternion_horizon_mix != horizon_mix ||
		cached_counter_quaternion_roll_mode != roll_mode;
	if (uniform_counter_quaternions.size() != quaternion_count || sampling_changed) {
		uniform_counter_quaternions.clear();
		uniform_counter_quaternions.reserve(quaternion_count);
		cached_counter_quaternion_distribution = distribution;
		cached_counter_quaternion_horizon_power = horizon_power;
		cached_counter_quaternion_horizon_mix = horizon_mix;
		cached_counter_quaternion_roll_mode = roll_mode;
		// Keep the historical fixed-seed Shoemake path byte-for-byte equivalent
		// so existing uniform experiments remain reproducible.
		std::mt19937 random(0x51A7E123u);
		std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
		double sum_abs_world_y = 0.0;
		double sum_squared_world_y = 0.0;
		double sum_world_up_alignment = 0.0;
		for (uint64_t i = 0; i < quaternion_count; ++i) {
			Quaternion rotation;
			if (distribution == CounterQuaternionDistribution::UNIFORM) {
				const float u1 = uniform(random);
				const float u2 = uniform(random);
				const float u3 = uniform(random);
				const float root_one_minus_u1 = sqrtf(1.0f - u1);
				const float root_u1 = sqrtf(u1);
				const float angle_2 = TWO_PI * u2;
				const float angle_3 = TWO_PI * u3;
				rotation = Quaternion(
					root_one_minus_u1 * sinf(angle_2),
					root_one_minus_u1 * cosf(angle_2),
					root_u1 * sinf(angle_3),
					root_u1 * cosf(angle_3));
				if (roll_mode == CounterQuaternionRollMode::WORLD_UP) {
					// Preserve the exact historical Shoemake forward-direction sequence;
					// only reconstruct its orientation with world-up zero roll.
					const Vector3 direction = Vector3::normalize(
						rotation * Vector3(0.0f, 0.0f, -1.0f));
					rotation = make_world_up_camera_rotation(direction);
				}
			} else {
				// Uniformly propose a direction on S^2, then accept the wall-biased
				// component with probability sin(theta)^power, where theta is measured
				// from world +Y. The resulting solid-angle density is therefore largest
				// around the world-space equator and vanishes at the poles for power > 0.
				const bool horizon_component = uniform(random) < horizon_mix;
				Vector3 direction;
				while (true) {
					const float y = 1.0f - 2.0f * uniform(random);
					const float azimuth = TWO_PI * uniform(random);
					const float radial_squared = Math::max(1.0f - y * y, 0.0f);
					const float radial = sqrtf(radial_squared);
					direction = Vector3(radial * cosf(azimuth), y, radial * sinf(azimuth));
					if (!horizon_component || horizon_power == 0.0f ||
						uniform(random) <= powf(radial, horizon_power)) break;
				}

				// A direction alone leaves camera roll unspecified. Draw it uniformly
				// so the camera and single-ray modes share directions without giving
				// the full-frame camera an artificial world-up preference.
				Quaternion look_rotation = make_world_up_camera_rotation(direction);
				// Consume the roll variate in both modes so subsequent directions remain
				// identical and the experiment changes roll alone.
				const float roll_angle = TWO_PI * uniform(random);
				if (roll_mode == CounterQuaternionRollMode::UNIFORM) {
					Quaternion roll_rotation = Quaternion::axis_angle(direction, roll_angle);
					rotation = Quaternion::normalize(roll_rotation * look_rotation);
				} else {
					rotation = look_rotation;
				}
			}
			// q and -q encode the same rotation. Keep one S^3 representative.
			if (rotation.w < 0.0f) {
				rotation = Quaternion(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
			}
			const Vector3 sampled_direction = Vector3::normalize(
				rotation * Vector3(0.0f, 0.0f, -1.0f));
			const Vector3 reference_up = fabsf(sampled_direction.y) < 0.999f
				? Vector3(0.0f, 1.0f, 0.0f) : Vector3(1.0f, 0.0f, 0.0f);
			const Vector3 projected_world_up = Vector3::normalize(
				reference_up - Vector3::dot(reference_up, sampled_direction) * sampled_direction);
			const Vector3 camera_up = Vector3::normalize(
				rotation * Vector3(0.0f, 1.0f, 0.0f));
			sum_abs_world_y += fabs(double(sampled_direction.y));
			sum_squared_world_y += double(sampled_direction.y) * double(sampled_direction.y);
			sum_world_up_alignment += double(Vector3::dot(camera_up, projected_world_up));
			uniform_counter_quaternions.push_back(rotation);
		}
		const char * roll_label = roll_mode == CounterQuaternionRollMode::UNIFORM
			? "uniform roll" : "world-up zero roll";
		if (distribution == CounterQuaternionDistribution::UNIFORM) {
			IO::print("Prepared {} global uniform-direction camera quaternions (Shoemake forward directions, fixed seed, {}, w >= 0).\n"_sv,
				quaternion_count, roll_label);
		} else {
			IO::print("Prepared {} global world-horizon-sine camera quaternions (world +Y pole, sin(theta)^{}, horizon mix {}, fixed seed, {}).\n"_sv,
				quaternion_count, horizon_power, horizon_mix, roll_label);
		}
		IO::print("Counter quaternion direction check: mean |world Y| {}, RMS world Y {} (smaller means stronger XZ-equator concentration).\n"_sv,
			float(sum_abs_world_y / double(quaternion_count)),
			float(sqrt(sum_squared_world_y / double(quaternion_count))));
		std::printf("Counter quaternion roll check: mean camera-up alignment with projected world-up %.9f (world-up mode should be 1).\n",
			double(sum_world_up_alignment / double(quaternion_count)));
	}

	const uint64_t position_index = render_index / quaternion_count;
	const uint64_t quaternion_index = render_index % quaternion_count;
	if (position_index >= position_count) return false;
	if (!set_camera_keyframe_sample(position_index, position_count)) return false;
	camera.rotation = uniform_counter_quaternions[quaternion_index];
	return true;
}

bool Scene::set_uniform_quaternion_fps_sample(
	uint64_t quaternion_index, uint64_t position_index, uint64_t position_count,
	uint64_t quaternion_count, CounterQuaternionRollMode roll_mode) {
	if (position_count == 0 || quaternion_count == 0 || quaternion_index >= quaternion_count || camera_keyframes.size() == 0) return false;

	if (uniform_fps_quaternions.size() != quaternion_count || cached_fps_quaternion_roll_mode != roll_mode) {
		uniform_fps_quaternions.clear();
		uniform_fps_quaternions.reserve(quaternion_count);
		cached_fps_quaternion_roll_mode = roll_mode;
		// Deliberately independent from the counter-training quaternion seed.
		std::mt19937 random(0xB17E5EEDu);
		std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
		// The fourth raw sample clustered with the second and third samples on
		// the world -X side. Use raw sample 7 as B3, then retain every displaced
		// sample in deterministic order if a run requests more than four.
		static constexpr uint64_t first_sample_order[8] = { 0, 1, 2, 7, 3, 4, 5, 6 };
		const uint64_t candidate_count = Math::max(quaternion_count, uint64_t(8));
		std::vector<Quaternion> candidates;
		candidates.reserve(candidate_count);
		for (uint64_t i = 0; i < candidate_count; ++i) {
			const float u1 = uniform(random);
			const float u2 = uniform(random);
			const float u3 = uniform(random);
			const float root_one_minus_u1 = sqrtf(1.0f - u1);
			const float root_u1 = sqrtf(u1);
			const float angle_2 = TWO_PI * u2;
			const float angle_3 = TWO_PI * u3;
			Quaternion rotation(
				root_one_minus_u1 * sinf(angle_2),
				root_one_minus_u1 * cosf(angle_2),
				root_u1 * sinf(angle_3),
				root_u1 * cosf(angle_3));
			if (roll_mode == CounterQuaternionRollMode::WORLD_UP) {
				const Vector3 direction = Vector3::normalize(
					rotation * Vector3(0.0f, 0.0f, -1.0f));
				rotation = make_world_up_camera_rotation(direction);
			}
			if (rotation.w < 0.0f) {
				rotation = Quaternion(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
			}
			candidates.push_back(rotation);
		}
		double sum_fps_world_up_alignment = 0.0;
		for (uint64_t i = 0; i < quaternion_count; ++i) {
			const uint64_t candidate_index = i < 8 ? first_sample_order[i] : i;
			const Quaternion rotation = candidates[candidate_index];
			uniform_fps_quaternions.push_back(rotation);
			const Vector3 direction = Vector3::normalize(
				rotation * Vector3(0.0f, 0.0f, -1.0f));
			const Vector3 reference_up = fabsf(direction.y) < 0.999f
				? Vector3(0.0f, 1.0f, 0.0f) : Vector3(1.0f, 0.0f, 0.0f);
			const Vector3 projected_world_up = Vector3::normalize(
				reference_up - Vector3::dot(reference_up, direction) * direction);
			const Vector3 camera_up = Vector3::normalize(
				rotation * Vector3(0.0f, 1.0f, 0.0f));
			sum_fps_world_up_alignment += double(Vector3::dot(camera_up, projected_world_up));
			std::printf("FPS test rotation B%llu = Quaternion(%.9g, %.9g, %.9g, %.9g)\n",
				static_cast<unsigned long long>(i),
				double(rotation.x), double(rotation.y), double(rotation.z), double(rotation.w));
		}
		IO::print("Prepared {} independent global FPS rotations (Shoemake forward directions, fixed seed, {}, w >= 0).\n"_sv,
			quaternion_count,
			roll_mode == CounterQuaternionRollMode::UNIFORM ? "uniform roll" : "world-up zero roll");
		std::printf("FPS quaternion roll check: mean camera-up alignment with projected world-up %.9f (world-up mode should be 1).\n",
			sum_fps_world_up_alignment / double(quaternion_count));
	}

	if (!set_camera_keyframe_sample(position_index, position_count)) return false;
	camera.rotation = uniform_fps_quaternions[quaternion_index];
	return true;
}

bool Scene::set_camera_keyframe_time(float time) {
	int first = -1;
	int previous = -1;
	for (int i = 0; i < camera_keyframes.size(); i++) {
		if (camera_keyframes[i].phantom) continue;
		if (first < 0) first = i;

		if (previous >= 0 && time <= camera_keyframes[i].time) {
			const CameraKeyframe & a = camera_keyframes[previous];
			const CameraKeyframe & b = camera_keyframes[i];
			const float duration = b.time - a.time;
			const float t = duration > 0.0f ? Math::clamp((time - a.time) / duration, 0.0f, 1.0f) : 1.0f;
			const Vector3 position = a.position * (1.0f - t) + b.position * t;
			const Vector3 lookat = a.lookat * (1.0f - t) + b.lookat * t;
			const Vector3 forward = lookat - position;
			if (Vector3::length_squared(forward) == 0.0f) return false;

			const Vector3 up = fabsf(Vector3::normalize(forward).y) > 0.9999f ? Vector3(0.0f, 0.0f, 1.0f) : Vector3(0.0f, 1.0f, 0.0f);
			camera.position = position;
			camera.rotation = Quaternion::look_rotation(-forward, up);
			return true;
		}

		previous = i;
	}

	if (first < 0) return false;
	const CameraKeyframe & last = camera_keyframes[previous];
	const Vector3 forward = last.lookat - last.position;
	if (Vector3::length_squared(forward) == 0.0f) return false;
	const Vector3 up = fabsf(Vector3::normalize(forward).y) > 0.9999f ? Vector3(0.0f, 0.0f, 1.0f) : Vector3(0.0f, 1.0f, 0.0f);
	camera.position = last.position;
	camera.rotation = Quaternion::look_rotation(-forward, up);
	return true;
}

bool Scene::set_camera_keyframe_sample(uint64_t sample_index, uint64_t sample_count) {
	int first = -1;
	int last = -1;
	for (int i = 0; i < camera_keyframes.size(); i++) {
		if (!camera_keyframes[i].phantom) {
			if (first < 0) first = i;
			last = i;
		}
	}
	if (first < 0) return false;

	const float progress = sample_count > 1 ? float(sample_index) / float(sample_count - 1) : 0.0f;
	const float time = camera_keyframes[first].time + (camera_keyframes[last].time - camera_keyframes[first].time) * Math::clamp(progress, 0.0f, 1.0f);
	return set_camera_keyframe_time(time);
}

void Scene::check_materials() {
	has_diffuse    = false;
	has_plastic    = false;
	has_pbr_metallic_roughness = false;
	has_dielectric = false;
	has_conductor  = false;
	has_lights     = false;

	// Check properties of the Scene, so we know which kernels are required
	for (int i = 0; i < asset_manager.materials.size(); i++) {
		const Material & material = asset_manager.materials[i];
		switch (material.type) {
			case Material::Type::DIFFUSE:    has_diffuse    = true; break;
			case Material::Type::PLASTIC:    has_plastic    = true; break;
			case Material::Type::PBR_METALLIC_ROUGHNESS: has_pbr_metallic_roughness = true; break;
			case Material::Type::DIELECTRIC: has_dielectric = true; break;
			case Material::Type::CONDUCTOR:  has_conductor  = true; break;
			case Material::Type::LIGHT:      has_lights    |= material.is_light(); break;
			default: ASSERT_UNREACHABLE();
		}
		// Textured emission is not represented by Material::Type::LIGHT, but it
		// still needs a shadow-ray buffer and area-light sampling.
		has_lights |= material.emissive_texture_handle.handle != INVALID &&
			(material.emission.x > 0.0f || material.emission.y > 0.0f || material.emission.z > 0.0f);
	}
	has_lights |= punctual_lights.size() > 0 || directional_lights.size() > 0;
}

void Scene::update(float delta) {
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].update();
	}
}
