#pragma once
#include "Core/String.h"

#include "Math/Vector3.h"

#include "Handle.h"

#include "Medium.h"
#include "Texture.h"

struct Material {
	String name;

	enum struct Type : char {
		LIGHT,
		DIFFUSE,
		PLASTIC,
		PBR_METALLIC_ROUGHNESS,
		DIELECTRIC,
		CONDUCTOR
	};

	Type type = Type::DIFFUSE;

	Vector3 emission;
	Handle<Texture> emissive_texture_handle;

	Vector3         diffuse = Vector3(1.0f, 1.0f, 1.0f);
	Handle<Texture> texture_handle;
	// glTF 2.0 metallic-roughness workflow. The texture stores roughness in G
	// and metallic in B; diffuse/texture_handle remain the base-color inputs.
	float           metallic = 0.0f;
	Handle<Texture> metallic_roughness_texture_handle;

	// glTF alpha coverage. BLEND falls back to deterministic cutout.
	float alpha        = 1.0f;
	float alpha_cutoff = 0.5f;
	bool  alpha_masked = false;

	Handle<Medium> medium_handle;
	float          index_of_refraction = 1.33f;

	Vector3 eta = Vector3(1.33f);
	Vector3 k   = Vector3(1.0f);

	float linear_roughness = 0.5f;

	bool is_light() const {
		return type == Type::LIGHT && Vector3::length_squared(emission) > 0.0f;
	}
};
