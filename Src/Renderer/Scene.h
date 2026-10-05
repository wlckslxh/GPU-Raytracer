#pragma once
#include "Core/Allocators/LinearAllocator.h"

#include "Assets/AssetManager.h"

#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

// Analytic point lights imported from KHR_lights_punctual.  `radius` is the
// project-specific attenuation range used by the Vulkan renderer, not a mesh
// radius.
struct PunctualLight {
	Vector3 position;
	Vector3 color;
	float   radius = 1.0f;
};

struct Scene {
	Allocator * allocator = nullptr;

	AssetManager asset_manager;

	Camera      camera;
	Array<Mesh> meshes;
	Array<PunctualLight> punctual_lights;
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
