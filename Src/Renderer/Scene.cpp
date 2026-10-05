#include "Scene.h"

#include <ctype.h>
#include <filesystem>
#include <fstream>
#include <stdio.h>

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

} // namespace

Scene::Scene(Allocator * allocator) : allocator(allocator), asset_manager(allocator), camera(Math::deg_to_rad(85.0f)), meshes(allocator), punctual_lights(allocator), directional_lights(allocator), camera_presets(allocator), camera_keyframes(allocator) {
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
	has_dielectric = false;
	has_conductor  = false;
	has_lights     = false;

	// Check properties of the Scene, so we know which kernels are required
	for (int i = 0; i < asset_manager.materials.size(); i++) {
		const Material & material = asset_manager.materials[i];
		switch (material.type) {
			case Material::Type::DIFFUSE:    has_diffuse    = true; break;
			case Material::Type::PLASTIC:    has_plastic    = true; break;
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
