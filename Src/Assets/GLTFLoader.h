#pragma once

#include "Core/Array.h"
#include "Renderer/Triangle.h"

#include "Core/String.h"

struct Scene;

namespace GLTFLoader {
	// Loads static triangle geometry from a glTF 2.0 .gltf or .glb file.
	// Node transforms are baked into the returned triangles.
	Array<Triangle> load(const String & filename, Allocator * allocator);

	// Imports all glTF primitives into one renderer MeshData/BLAS while retaining
	// material IDs per triangle for texture and emissive-light lookup.
	void load_scene(const String & filename, Scene & scene);
}
