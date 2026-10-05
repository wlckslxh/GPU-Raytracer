#pragma once
#include "Core/Allocators/LinearAllocator.h"

#include "Assets/AssetManager.h"

#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

// Analytic lights imported from KHR_lights_punctual. `range == 0` denotes the
// glTF default: no finite cutoff range.
struct PunctualLight {
	Vector3 position;
	Vector3 color;
	float   intensity = 1.0f;
	float   range = 0.0f;
};

struct DirectionalLight {
	// Direction in which light travels (the glTF node's local -Z axis).
	Vector3 direction;
	Vector3 color;
	float   intensity = 1.0f;
};

struct Scene {
	Allocator * allocator = nullptr;

	AssetManager asset_manager;

	Camera      camera;
	Array<Mesh> meshes;
	Array<PunctualLight> punctual_lights;
	Array<DirectionalLight> directional_lights;
	Sky         sky;

	bool has_diffuse    = false;
	bool has_plastic    = false;
	bool has_dielectric = false;
	bool has_conductor  = false;
	bool has_lights     = false;

	Scene(Allocator * allocator);

	Mesh & add_mesh(String name, Handle<MeshData> mesh_data_handle, Handle<Material> material_handle = Handle<Material>::get_default());

	void check_materials();

	void update(float delta);
};
