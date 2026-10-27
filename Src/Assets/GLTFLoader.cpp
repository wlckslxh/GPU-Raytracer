#include "GLTFLoader.h"

#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "Core/IO.h"
#include "Math/Matrix4.h"
#include "Math/Quaternion.h"
#include "Renderer/Scene.h"

// tinygltf is deliberately not committed as part of this change.  Put
// tiny_gltf.h, json.hpp and stb_image.h in ThirdParty/tinygltf/.
#if __has_include("../../ThirdParty/tinygltf/tiny_gltf.h")
	#define TINYGLTF_IMPLEMENTATION
	#define TINYGLTF_NO_STB_IMAGE_WRITE
	#include "../../ThirdParty/tinygltf/tiny_gltf.h"
	#define GPU_RAYTRACER_HAS_TINYGLTF 1
#elif __has_include("tiny_gltf.h")
	// Also allow a build system to supply tinygltf through its include path.
	#define TINYGLTF_IMPLEMENTATION
	#define TINYGLTF_NO_STB_IMAGE_WRITE
	#include "tiny_gltf.h"
	#define GPU_RAYTRACER_HAS_TINYGLTF 1
#else
	#define GPU_RAYTRACER_HAS_TINYGLTF 0
#endif

#if GPU_RAYTRACER_HAS_TINYGLTF

// tinygltf normally tries to decode every external image itself. Keep KTX/KTX2
// URIs intact instead; libktx transcodes them asynchronously through
// AssetManager after the glTF material has been imported.
static bool load_gltf_image(
	tinygltf::Image * image,
	const int image_index,
	std::string * error,
	std::string * warning,
	int requested_width,
	int requested_height,
	const unsigned char * bytes,
	int size,
	void * user_data)
{
	const size_t dot = image->uri.find_last_of('.');
	if (dot != std::string::npos) {
		std::string extension = image->uri.substr(dot + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
		if (extension == "ktx" || extension == "ktx2") return true;
	}

	return tinygltf::LoadImageData(
		image, image_index, error, warning, requested_width, requested_height, bytes, size, user_data);
}

static Matrix4 get_node_transform(const tinygltf::Node & node) {
	if (node.matrix.size() == 16) {
		Matrix4 result;
		// glTF stores matrices in column-major order; Matrix4(row, col) is used here.
		for (int row = 0; row < 4; row++) {
			for (int col = 0; col < 4; col++) {
				result(row, col) = float(node.matrix[col * 4 + row]);
			}
		}
		return result;
	}

	Vector3 translation(0.0f);
	Vector3 scale(1.0f);
	Quaternion rotation;

	if (node.translation.size() == 3) {
		translation = Vector3(float(node.translation[0]), float(node.translation[1]), float(node.translation[2]));
	}
	if (node.rotation.size() == 4) {
		rotation = Quaternion(
			float(node.rotation[0]), float(node.rotation[1]),
			float(node.rotation[2]), float(node.rotation[3])
		);
	}
	if (node.scale.size() == 3) {
		scale = Vector3(float(node.scale[0]), float(node.scale[1]), float(node.scale[2]));
	}

	return
		Matrix4::create_translation(translation) *
		Matrix4::create_rotation(rotation) *
		Matrix4::create_scale(scale.x, scale.y, scale.z);
}

static const unsigned char * get_accessor_data(
	const tinygltf::Model & model, const tinygltf::Accessor & accessor, size_t element_size, size_t * stride_out)
{
	if (accessor.bufferView < 0 || size_t(accessor.bufferView) >= model.bufferViews.size()) return nullptr;

	const tinygltf::BufferView & view = model.bufferViews[accessor.bufferView];
	if (view.buffer < 0 || size_t(view.buffer) >= model.buffers.size()) return nullptr;

	const tinygltf::Buffer & buffer = model.buffers[view.buffer];
	size_t stride = view.byteStride ? size_t(view.byteStride) : element_size;
	size_t offset = size_t(view.byteOffset) + size_t(accessor.byteOffset);

	if (stride < element_size || offset > buffer.data.size()) return nullptr;
	if (accessor.count > 0) {
		if (element_size > buffer.data.size() - offset) return nullptr;
		if ((accessor.count - 1) > (buffer.data.size() - offset - element_size) / stride) return nullptr;
	}

	*stride_out = stride;
	return buffer.data.data() + offset;
}

static float read_float(const unsigned char * data, size_t stride, size_t index, int component) {
	float result;
	memcpy(&result, data + index * stride + size_t(component) * sizeof(float), sizeof(result));
	return result;
}

static Vector3 read_vec3(const unsigned char * data, size_t stride, size_t index) {
	return Vector3(read_float(data, stride, index, 0), read_float(data, stride, index, 1), read_float(data, stride, index, 2));
}

static Vector2 read_vec2(const unsigned char * data, size_t stride, size_t index) {
	return Vector2(read_float(data, stride, index, 0), read_float(data, stride, index, 1));
}

static bool append_primitive(const tinygltf::Model & model, const tinygltf::Primitive & primitive, const Matrix4 & transform, Array<Triangle> & triangles) {
	if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
		IO::print("Ignoring non-triangle glTF primitive\n"_sv);
		return true;
	}

	auto position_it = primitive.attributes.find("POSITION");
	if (position_it == primitive.attributes.end()) {
		IO::print("Ignoring glTF primitive without POSITION\n"_sv);
		return true;
	}
	if (position_it->second < 0 || size_t(position_it->second) >= model.accessors.size()) {
		IO::print("Ignoring glTF primitive with invalid POSITION accessor index\n"_sv);
		return true;
	}

	const tinygltf::Accessor & positions = model.accessors[position_it->second];
	if (positions.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || positions.type != TINYGLTF_TYPE_VEC3) {
		IO::print("Ignoring glTF primitive with unsupported POSITION accessor\n"_sv);
		return true;
	}

	size_t position_stride;
	const unsigned char * position_data = get_accessor_data(model, positions, 3 * sizeof(float), &position_stride);
	if (!position_data) {
		IO::print("Ignoring glTF primitive with invalid POSITION accessor\n"_sv);
		return true;
	}

	const unsigned char * normal_data = nullptr;
	size_t normal_stride = 0;
	auto normal_it = primitive.attributes.find("NORMAL");
	if (normal_it != primitive.attributes.end() && normal_it->second >= 0 && size_t(normal_it->second) < model.accessors.size()) {
		const tinygltf::Accessor & normals = model.accessors[normal_it->second];
		if (normals.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && normals.type == TINYGLTF_TYPE_VEC3 && normals.count == positions.count) {
			normal_data = get_accessor_data(model, normals, 3 * sizeof(float), &normal_stride);
		}
	}

	const unsigned char * uv_data = nullptr;
	size_t uv_stride = 0;
	auto uv_it = primitive.attributes.find("TEXCOORD_0");
	if (uv_it != primitive.attributes.end() && uv_it->second >= 0 && size_t(uv_it->second) < model.accessors.size()) {
		const tinygltf::Accessor & uvs = model.accessors[uv_it->second];
		if (uvs.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && uvs.type == TINYGLTF_TYPE_VEC2 && uvs.count == positions.count) {
			uv_data = get_accessor_data(model, uvs, 2 * sizeof(float), &uv_stride);
		}
	}

	const unsigned char * index_data = nullptr;
	size_t index_stride = 0;
	int index_component_type = 0;
	size_t index_count = positions.count;
	if (primitive.indices >= 0) {
		if (size_t(primitive.indices) >= model.accessors.size()) {
			IO::print("Ignoring glTF primitive with invalid index accessor index\n"_sv);
			return true;
		}
		const tinygltf::Accessor & indices = model.accessors[primitive.indices];
		if (indices.type != TINYGLTF_TYPE_SCALAR) {
			IO::print("Ignoring glTF primitive with non-scalar indices\n"_sv);
			return true;
		}
		size_t element_size = 0;
		switch (indices.componentType) {
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  element_size = sizeof(uint8_t);  break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: element_size = sizeof(uint16_t); break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   element_size = sizeof(uint32_t); break;
			default:
				IO::print("Ignoring glTF primitive with unsupported index type\n"_sv);
				return true;
		}
		index_data = get_accessor_data(model, indices, element_size, &index_stride);
		if (!index_data) {
			IO::print("Ignoring glTF primitive with invalid index accessor\n"_sv);
			return true;
		}
		index_component_type = indices.componentType;
		index_count = indices.count;
	}

	if (index_count % 3 != 0) {
		IO::print("Ignoring glTF primitive whose index count is not divisible by three\n"_sv);
		return true;
	}

	auto read_index = [&](size_t index) {
		if (!index_data) return uint32_t(index);
		const unsigned char * src = index_data + index * index_stride;
		uint32_t value = 0;
		switch (index_component_type) {
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  value = *src; break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: { uint16_t v; memcpy(&v, src, sizeof(v)); value = v; break; }
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   { uint32_t v; memcpy(&v, src, sizeof(v)); value = v; break; }
		}
		return value;
	};

	Matrix4 normal_transform = Matrix4::cofactor(transform);
	for (size_t i = 0; i < index_count; i += 3) {
		uint32_t indices[3] = { read_index(i), read_index(i + 1), read_index(i + 2) };
		if (indices[0] >= positions.count || indices[1] >= positions.count || indices[2] >= positions.count) {
			IO::print("Ignoring glTF triangle with an out-of-range index\n"_sv);
			continue;
		}

		Vector3 position[3];
		Vector3 normal[3] = { };
		Vector2 uv[3] = { };
		for (int v = 0; v < 3; v++) {
			position[v] = Matrix4::transform_position(transform, read_vec3(position_data, position_stride, indices[v]));
			if (normal_data) normal[v] = Vector3::normalize(Matrix4::transform_direction(normal_transform, read_vec3(normal_data, normal_stride, indices[v])));
			if (uv_data) uv[v] = read_vec2(uv_data, uv_stride, indices[v]);
		}

		triangles.emplace_back(position[0], position[1], position[2], normal[0], normal[1], normal[2], uv[0], uv[1], uv[2]);
	}
	return true;
}

static void append_node(const tinygltf::Model & model, int node_index, const Matrix4 & parent_transform, Array<Triangle> & triangles) {
	if (node_index < 0 || size_t(node_index) >= model.nodes.size()) {
		IO::print("Ignoring invalid glTF node index\n"_sv);
		return;
	}

	const tinygltf::Node & node = model.nodes[node_index];
	Matrix4 transform = parent_transform * get_node_transform(node);
	if (node.mesh >= 0 && size_t(node.mesh) < model.meshes.size()) {
		for (const tinygltf::Primitive & primitive : model.meshes[node.mesh].primitives) {
			append_primitive(model, primitive, transform, triangles);
		}
	}
	for (int child_index : node.children) {
		append_node(model, child_index, transform, triangles);
	}
}

static String make_string(const std::string & value, const char * fallback) {
	return value.empty() ? String(fallback) : String(value.c_str(), value.size());
}

static Handle<Texture> import_texture(
	const tinygltf::Model & model,
	int texture_index,
	Scene & scene,
	std::vector<Handle<Texture>> & texture_handles,
	const std::filesystem::path & gltf_directory)
{
	if (texture_index < 0 || size_t(texture_index) >= model.textures.size())
		return { INVALID };
	if (texture_handles[texture_index].handle != INVALID)
		return texture_handles[texture_index];

	const tinygltf::Texture & gltf_texture = model.textures[texture_index];
	if (gltf_texture.source < 0 || size_t(gltf_texture.source) >= model.images.size()) {
		IO::print("Ignoring glTF texture with an invalid image source\n"_sv);
		return { INVALID };
	}

	const tinygltf::Image & image = model.images[gltf_texture.source];
	const size_t uri_dot = image.uri.find_last_of('.');
	if (uri_dot != std::string::npos) {
		std::string extension = image.uri.substr(uri_dot + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
		if (extension == "ktx2") {
			const std::string texture_filename = (gltf_directory / image.uri).string();
			texture_handles[texture_index] = scene.asset_manager.add_texture(
				String(texture_filename.c_str(), scene.allocator),
				make_string(image.name, "glTF KTX2 Texture"));
			return texture_handles[texture_index];
		}
	}

	if (image.width <= 0 || image.height <= 0 || image.component < 1 || image.component > 4 || image.bits != 8) {
		IO::print("Ignoring glTF image with unsupported pixel format\n"_sv);
		return { INVALID };
	}

	const size_t pixel_count = size_t(image.width) * size_t(image.height);
	const size_t source_size = pixel_count * size_t(image.component);
	if (image.image.size() < source_size) {
		IO::print("Ignoring glTF image with truncated pixel data\n"_sv);
		return { INVALID };
	}

	Texture texture = { };
	texture.name = make_string(image.name, "glTF Texture");
	texture.format = Texture::Format::RGBA;
	texture.channels = 4;
	texture.width = image.width;
	texture.height = image.height;
	texture.mip_offsets = { 0 };
	texture.data.resize(pixel_count * 4);

	for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
		const unsigned char * source = image.image.data() + pixel * image.component;
		unsigned char * destination = texture.data.data() + pixel * 4;
		switch (image.component) {
			case 1: destination[0] = destination[1] = destination[2] = source[0]; destination[3] = 255; break;
			case 2: destination[0] = destination[1] = destination[2] = source[0]; destination[3] = source[1]; break;
			case 3: destination[0] = source[0]; destination[1] = source[1]; destination[2] = source[2]; destination[3] = 255; break;
			case 4: memcpy(destination, source, 4); break;
		}
	}

	texture_handles[texture_index] = scene.asset_manager.add_texture(std::move(texture));
	return texture_handles[texture_index];
}

static Handle<Material> import_material(
	const tinygltf::Model & model,
	int material_index,
	Scene & scene,
	std::vector<Handle<Material>> & material_handles,
	std::vector<Handle<Texture>> & texture_handles,
	const std::filesystem::path & gltf_directory)
{
	if (material_index < 0 || size_t(material_index) >= model.materials.size())
		return Handle<Material>::get_default();
	if (material_handles[material_index].handle != INVALID)
		return material_handles[material_index];

	const tinygltf::Material & gltf_material = model.materials[material_index];
	Material material = { };
	material.name = make_string(gltf_material.name, "glTF Material");

	const std::vector<double> & base_color = gltf_material.pbrMetallicRoughness.baseColorFactor;
	if (base_color.size() >= 3) {
		material.diffuse = Vector3(
			float(base_color[0]), float(base_color[1]), float(base_color[2]));
	}
	if (base_color.size() >= 4) {
		material.alpha = float(base_color[3]);
	}
	material.alpha_masked = gltf_material.alphaMode == "MASK" || gltf_material.alphaMode == "BLEND";
	material.alpha_cutoff = gltf_material.alphaMode == "MASK" ? float(gltf_material.alphaCutoff) : 0.5f;
	material.linear_roughness = float(gltf_material.pbrMetallicRoughness.roughnessFactor);
	material.metallic = float(gltf_material.pbrMetallicRoughness.metallicFactor);
	material.texture_handle = import_texture(
		model,
		gltf_material.pbrMetallicRoughness.baseColorTexture.index,
		scene,
		texture_handles,
		gltf_directory);
	material.metallic_roughness_texture_handle = import_texture(
		model,
		gltf_material.pbrMetallicRoughness.metallicRoughnessTexture.index,
		scene,
		texture_handles,
		gltf_directory);
	material.emissive_texture_handle = import_texture(
		model,
		gltf_material.emissiveTexture.index,
		scene,
		texture_handles,
		gltf_directory);

	const std::vector<double> & emission = gltf_material.emissiveFactor;
	const auto get_extra_number = [&gltf_material](const char * name, float * value) {
		if (!gltf_material.extras.IsObject() || !gltf_material.extras.Has(name)) return false;
		const tinygltf::Value & extra = gltf_material.extras.Get(name);
		if (!extra.IsNumber()) return false;
		*value = float(extra.GetNumberAsDouble());
		return true;
	};

	float reflectance = 0.0f;
	float refractance = 0.0f;
	get_extra_number("Kr", &reflectance);
	get_extra_number("Kt", &refractance);

	if (reflectance > 0.0f) {
		material.type = Material::Type::CONDUCTOR;
		material.linear_roughness = 0.001f;
		material.eta = Vector3(0.155f, 0.116f, 0.138f);
		material.k   = Vector3(4.828f, 3.122f, 2.146f);
	} else if (refractance > 0.0f) {
		material.type = Material::Type::DIELECTRIC;
		material.index_of_refraction = 1.5f;
		material.linear_roughness = 0.001f;
	} else if (emission.size() >= 3 && (emission[0] != 0.0 || emission[1] != 0.0 || emission[2] != 0.0)) {
		material.emission = Vector3(float(emission[0]), float(emission[1]), float(emission[2]));
		if (gltf_material.extensions.find("KHR_materials_emissive_strength") != gltf_material.extensions.end()) {
			const tinygltf::Value & extension = gltf_material.extensions.at("KHR_materials_emissive_strength");
			if (extension.IsObject() && extension.Has("emissiveStrength")) {
				const tinygltf::Value & strength = extension.Get("emissiveStrength");
				if (strength.IsNumber()) material.emission *= float(strength.GetNumberAsDouble());
			}
		}

		// An emissive texture is a per-UV self-emission term. Keep the surface
		// material so its base-color/alpha texture still shades normally.
		if (gltf_material.emissiveTexture.index < 0) {
			material.type = Material::Type::LIGHT;
		}
	} else {
		material.type = Material::Type::PBR_METALLIC_ROUGHNESS;
	}

	material_handles[material_index] = scene.asset_manager.add_material(std::move(material));
	return material_handles[material_index];
}

// Import glTF punctual lights as analytic lights instead of synthesizing
// emissive geometry.
static void append_punctual_point_light(
	const tinygltf::Model & model,
	const tinygltf::Node & node,
	const Matrix4 & world_transform,
	Scene & scene)
{
	auto extension_it = node.extensions.find("KHR_lights_punctual");
	if (extension_it == node.extensions.end()) return;

	const tinygltf::Value & extension = extension_it->second;
	if (!extension.IsObject() || !extension.Has("light")) {
		IO::print("Ignoring glTF node with malformed KHR_lights_punctual extension\n"_sv);
		return;
	}

	const tinygltf::Value & light_index_value = extension.Get("light");
	if (!light_index_value.IsNumber()) {
		IO::print("Ignoring glTF punctual light with a non-numeric light index\n"_sv);
		return;
	}

	const int light_index = int(light_index_value.GetNumberAsInt());
	if (light_index < 0 || size_t(light_index) >= model.lights.size()) {
		IO::print("Ignoring glTF punctual light with an invalid light index\n"_sv);
		return;
	}

	const tinygltf::Light & gltf_light = model.lights[light_index];
	if (gltf_light.type != "point" && gltf_light.type != "directional") {
		IO::print("Ignoring unsupported glTF {} light '{}'\n"_sv, gltf_light.type.c_str(), gltf_light.name.c_str());
		return;
	}
	Vector3 colour(1.0f);
	if (gltf_light.color.size() == 3) {
		colour = Vector3(
			float(gltf_light.color[0]),
			float(gltf_light.color[1]),
			float(gltf_light.color[2])
		);
	}

	// The models_0930 assets retain Blender/glTF's physical intensity in the
	// standard field, but their Vulkan renderer uses the artist-authored
	// normalized value in extras.intensity.  Prefer that project-specific value
	// when it is present; ordinary glTF files still use the standard field.
	float intensity = float(gltf_light.intensity);
	if (gltf_light.extras.IsObject() && gltf_light.extras.Has("intensity")) {
		const tinygltf::Value & intensity_value = gltf_light.extras.Get("intensity");
		if (intensity_value.IsNumber()) {
			const float custom_intensity = float(intensity_value.GetNumberAsDouble());
			if (std::isfinite(custom_intensity)) intensity = custom_intensity;
		}
	}
	if (!std::isfinite(intensity) || intensity <= 0.0f) return;

	if (gltf_light.type == "directional") {
		const Vector3 direction = Vector3::normalize(Matrix4::transform_direction(world_transform, Vector3(0.0f, 0.0f, -1.0f)));
		scene.directional_lights.emplace_back(DirectionalLight { direction, colour, intensity });
		return;
	}

	// glTF's range is optional. A missing range must not become an arbitrary
	// one-unit cutoff; zero represents an unbounded point light. Preserve the
	// custom extras.radius only as a fallback for the older Sponza asset.
	float range = 0.0f;
	if (std::isfinite(float(gltf_light.range)) && gltf_light.range > 0.0) {
		range = float(gltf_light.range);
	} else if (gltf_light.extras.IsObject() && gltf_light.extras.Has("radius")) {
		const tinygltf::Value & radius_value = gltf_light.extras.Get("radius");
		if (radius_value.IsNumber()) {
			const float parsed_radius = float(radius_value.GetNumberAsDouble());
			if (std::isfinite(parsed_radius) && parsed_radius > 0.0f) range = parsed_radius;
		}
	}

	scene.punctual_lights.emplace_back(PunctualLight {
		Matrix4::transform_position(world_transform, Vector3(0.0f)), colour, intensity, range });
}

static void append_node_to_single_blas(
	const tinygltf::Model & model,
	int node_index,
	const Matrix4 & parent_transform,
	Scene & scene,
	std::vector<Handle<Material>> & material_handles,
	std::vector<Handle<Texture>> & texture_handles,
	const std::filesystem::path & gltf_directory,
	Array<Triangle> & triangles,
	Array<int> & material_ids)
{
	if (node_index < 0 || size_t(node_index) >= model.nodes.size()) return;
	const tinygltf::Node & node = model.nodes[node_index];
	const Matrix4 transform = parent_transform * get_node_transform(node);
	append_punctual_point_light(model, node, transform, scene);

	if (node.mesh >= 0 && size_t(node.mesh) < model.meshes.size()) {
		for (const tinygltf::Primitive & primitive : model.meshes[node.mesh].primitives) {
			const size_t first_triangle = triangles.size();
			append_primitive(model, primitive, transform, triangles);

			Handle<Material> material_handle = import_material(
				model, primitive.material, scene, material_handles, texture_handles, gltf_directory);
			for (size_t triangle_index = first_triangle; triangle_index < triangles.size(); ++triangle_index) {
				material_ids.push_back(material_handle.handle);
			}
		}
	}

	for (int child_index : node.children) {
		append_node_to_single_blas(
			model, child_index, transform, scene, material_handles, texture_handles, gltf_directory,
			triangles, material_ids);
	}
}

Array<Triangle> GLTFLoader::load(const String & filename, Allocator * allocator) {
	tinygltf::TinyGLTF context;
	context.SetImageLoader(load_gltf_image, nullptr);

	tinygltf::Model model;
	std::string error, warning;
	std::string path(filename.c_str(), filename.size());
	bool is_binary = path.size() >= 4 && path.substr(path.size() - 4) == ".glb";
	bool loaded = is_binary
		? context.LoadBinaryFromFile(&model, &error, &warning, path)
		: context.LoadASCIIFromFile (&model, &error, &warning, path);

	if (!warning.empty()) IO::print("glTF warning: {}\n"_sv, warning.c_str());
	if (!loaded) {
		IO::print("Failed to load glTF '{}': {}\n"_sv, filename, error.c_str());
		return Array<Triangle>(allocator);
	}

	int scene_index = model.defaultScene >= 0 ? model.defaultScene : 0;
	if (scene_index < 0 || size_t(scene_index) >= model.scenes.size()) {
		IO::print("glTF '{}' has no valid scene\n"_sv, filename);
		return Array<Triangle>(allocator);
	}

	Array<Triangle> triangles(allocator);
	for (int node_index : model.scenes[scene_index].nodes) {
		append_node(model, node_index, Matrix4(), triangles);
	}

	IO::print("Loaded glTF '{}' from disk ({} triangles)\n"_sv, filename, triangles.size());
	return triangles;
}

void GLTFLoader::load_scene(const String & filename, Scene & scene) {
	tinygltf::TinyGLTF context;
	context.SetImageLoader(load_gltf_image, nullptr);
	tinygltf::Model model;
	std::string error, warning;
	std::string path(filename.c_str(), filename.size());
	const bool is_binary = path.size() >= 4 && path.substr(path.size() - 4) == ".glb";
	const bool loaded = is_binary
		? context.LoadBinaryFromFile(&model, &error, &warning, path)
		: context.LoadASCIIFromFile (&model, &error, &warning, path);

	if (!warning.empty()) IO::print("glTF warning: {}\n"_sv, warning.c_str());
	if (!loaded) {
		IO::print("Failed to load glTF '{}': {}\n"_sv, filename, error.c_str());
		return;
	}

	const int scene_index = model.defaultScene >= 0 ? model.defaultScene : 0;
	if (scene_index < 0 || size_t(scene_index) >= model.scenes.size()) {
		IO::print("glTF '{}' has no valid scene\n"_sv, filename);
		return;
	}

	std::vector<Handle<Texture>> texture_handles(model.textures.size());
	std::vector<Handle<Material>> material_handles(model.materials.size());
	for (Handle<Texture> & handle : texture_handles) handle.handle = INVALID;
	for (Handle<Material> & handle : material_handles) handle.handle = INVALID;

	const size_t mesh_count_before = scene.meshes.size();
	Array<Triangle> triangles(scene.allocator);
	Array<int> material_ids(scene.allocator);
	const std::filesystem::path gltf_directory = std::filesystem::path(path).parent_path();
	for (int node_index : model.scenes[scene_index].nodes) {
		append_node_to_single_blas(
			model, node_index, Matrix4(), scene, material_handles, texture_handles, gltf_directory,
			triangles, material_ids);
	}

	if (triangles.size() > 0) {
		scene.add_mesh(
			String(filename),
			scene.asset_manager.add_mesh_data(filename, std::move(triangles), std::move(material_ids)),
			Handle<Material>::get_default());
	}

	IO::print(
		"Loaded glTF '{}' from disk ({} single-BLAS mesh, {} materials, {} textures)\n"_sv,
		filename,
		scene.meshes.size() - mesh_count_before,
		model.materials.size(),
		model.textures.size());
}

#else

Array<Triangle> GLTFLoader::load(const String & filename, Allocator * allocator) {
	IO::print("Cannot load glTF '{}': tinygltf is missing. Copy tiny_gltf.h, json.hpp and stb_image.h to ThirdParty/tinygltf/.\n"_sv, filename);
	return Array<Triangle>(allocator);
}

void GLTFLoader::load_scene(const String & filename, Scene &) {
	IO::print("Cannot load glTF '{}': tinygltf is missing. Copy tiny_gltf.h, json.hpp and stb_image.h to ThirdParty/tinygltf/.\n"_sv, filename);
}

#endif
