#include "Sky.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <vector>

#include <stb_image.h>

#include "Core/IO.h"
#include "Core/String.h"

namespace {

constexpr uint8_t KTX1_IDENTIFIER[] = {
	0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
};

constexpr uint32_t KTX_ENDIANNESS_LE = 0x04030201;
constexpr uint32_t GL_UNSIGNED_BYTE  = 0x1401;
constexpr uint32_t GL_RGBA           = 0x1908;
constexpr uint32_t GL_RGBA8          = 0x8058;
constexpr uint32_t GL_SRGB8_ALPHA8   = 0x8C43;
constexpr float    PI                = 3.14159265358979323846f;

uint32_t read_u32_le(const uint8_t * data) {
	return uint32_t(data[0]) | (uint32_t(data[1]) << 8) | (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
}

float srgb_to_linear(float value) {
	return value <= 0.04045f ? value / 12.92f : powf((value + 0.055f) / 1.055f, 2.4f);
}

Vector4 sample_ktx1_face_rgba8(const uint8_t * const faces[6], int face_size, int face, float u, float v, bool is_srgb) {
	u = fminf(fmaxf(u, 0.0f), 1.0f);
	v = fminf(fmaxf(v, 0.0f), 1.0f);

	float x = u * float(face_size - 1);
	float y = v * float(face_size - 1);
	int   x0 = int(x);
	int   y0 = int(y);
	int   x1 = x0 < face_size - 1 ? x0 + 1 : x0;
	int   y1 = y0 < face_size - 1 ? y0 + 1 : y0;
	float tx = x - float(x0);
	float ty = y - float(y0);

	auto pixel = [&](int px, int py) {
		const uint8_t * src = faces[face] + 4 * (px + py * face_size);
		float r = src[0] * (1.0f / 255.0f);
		float g = src[1] * (1.0f / 255.0f);
		float b = src[2] * (1.0f / 255.0f);
		return Vector4(
			is_srgb ? srgb_to_linear(r) : r,
			is_srgb ? srgb_to_linear(g) : g,
			is_srgb ? srgb_to_linear(b) : b,
			src[3] * (1.0f / 255.0f)
		);
	};

	Vector4 x00 = pixel(x0, y0);
	Vector4 x10 = pixel(x1, y0);
	Vector4 x01 = pixel(x0, y1);
	Vector4 x11 = pixel(x1, y1);
	return (x00 * (1.0f - tx) + x10 * tx) * (1.0f - ty) + (x01 * (1.0f - tx) + x11 * tx) * ty;
}

Vector4 sample_ktx1_cubemap_rgba8(const uint8_t * const faces[6], int face_size, float x, float y, float z, bool is_srgb) {
	float abs_x = fabsf(x);
	float abs_y = fabsf(y);
	float abs_z = fabsf(z);

	int   face;
	float s;
	float t;
	if (abs_x >= abs_y && abs_x >= abs_z) {
		if (x >= 0.0f) {
			face = 0; // +X
			s = -z / abs_x;
			t = -y / abs_x;
		} else {
			face = 1; // -X
			s =  z / abs_x;
			t = -y / abs_x;
		}
	} else if (abs_y >= abs_z) {
		if (y >= 0.0f) {
			face = 2; // +Y
			s =  x / abs_y;
			t =  z / abs_y;
		} else {
			face = 3; // -Y
			s =  x / abs_y;
			t = -z / abs_y;
		}
	} else {
		if (z >= 0.0f) {
			face = 4; // +Z
			s =  x / abs_z;
			t = -y / abs_z;
		} else {
			face = 5; // -Z
			s = -x / abs_z;
			t = -y / abs_z;
		}
	}

	return sample_ktx1_face_rgba8(faces, face_size, face, 0.5f * (s + 1.0f), 0.5f * (t + 1.0f), is_srgb);
}

bool load_ktx1_cubemap(Sky & sky, const String & filename) {
	std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
	if (!file) return false;

	std::streamsize file_size = file.tellg();
	if (file_size < 64) return false;
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> bytes((size_t(file_size)));
	if (!file.read(reinterpret_cast<char *>(bytes.data()), file_size)) return false;
	if (memcmp(bytes.data(), KTX1_IDENTIFIER, sizeof(KTX1_IDENTIFIER)) != 0) return false;

	const uint8_t * header = bytes.data() + sizeof(KTX1_IDENTIFIER);
	uint32_t endianness            = read_u32_le(header +  0);
	uint32_t gl_type               = read_u32_le(header +  4);
	uint32_t gl_type_size          = read_u32_le(header +  8);
	uint32_t gl_format             = read_u32_le(header + 12);
	uint32_t gl_internal_format    = read_u32_le(header + 16);
	uint32_t gl_base_internal      = read_u32_le(header + 20);
	uint32_t face_width            = read_u32_le(header + 24);
	uint32_t face_height           = read_u32_le(header + 28);
	uint32_t face_depth            = read_u32_le(header + 32);
	uint32_t array_elements        = read_u32_le(header + 36);
	uint32_t faces_count           = read_u32_le(header + 40);
	uint32_t mip_levels            = read_u32_le(header + 44);
	uint32_t key_value_data_bytes  = read_u32_le(header + 48);

	bool supported_format =
		endianness == KTX_ENDIANNESS_LE &&
		gl_type == GL_UNSIGNED_BYTE && gl_type_size == 1 &&
		gl_format == GL_RGBA && gl_base_internal == GL_RGBA &&
		(gl_internal_format == GL_RGBA8 || gl_internal_format == GL_SRGB8_ALPHA8) &&
		face_width > 0 && face_width == face_height && face_depth == 0 &&
		array_elements == 0 && faces_count == 6 && mip_levels >= 1;
	if (!supported_format) return false;

	size_t offset = 64ull + key_value_data_bytes;
	if (offset + 4 > bytes.size()) return false;
	uint32_t image_size = read_u32_le(bytes.data() + offset);
	size_t expected_image_size = size_t(face_width) * size_t(face_height) * 4;
	if (image_size != expected_image_size) return false;
	offset += 4;

	const uint8_t * faces[6];
	for (int face = 0; face < 6; face++) {
		if (offset + image_size > bytes.size()) return false;
		faces[face] = bytes.data() + offset;
		offset += (size_t(image_size) + 3ull) & ~3ull;
	}

	// An equirectangular map with four pixels per cube-face pixel around the equator
	// preserves roughly the cubemap's source detail while staying compatible with sample_sky().
	if (face_width > uint32_t(INT32_MAX / 4)) return false;
	sky.width  = int(face_width * 4);
	sky.height = int(face_width * 2);
	sky.data.resize(size_t(sky.width) * size_t(sky.height));

	bool is_srgb = gl_internal_format == GL_SRGB8_ALPHA8;
	for (int y = 0; y < sky.height; y++) {
		float theta = (float(y) + 0.5f) * PI / float(sky.height);
		float sin_theta = sinf(theta);
		float cos_theta = cosf(theta);
		for (int x = 0; x < sky.width; x++) {
			float phi = ((float(x) + 0.5f) / float(sky.width) - 0.5f) * 2.0f * PI;
			// Inverse of Sky.h: phi = atan2(-z, x), theta = acos(y).
			float dir_x = sin_theta * cosf(phi);
			float dir_y = cos_theta;
			float dir_z = -sin_theta * sinf(phi);
			sky.data[x + y * sky.width] = sample_ktx1_cubemap_rgba8(faces, int(face_width), dir_x, dir_y, dir_z, is_srgb);
		}
	}

	return true;
}

bool has_ktx_extension(const String & filename) {
	return filename.size() >= 4 && strcmp(filename.data() + filename.size() - 4, ".ktx") == 0;
}

} // namespace

void Sky::load(const String & filename) {
	if (has_ktx_extension(filename)) {
		if (load_ktx1_cubemap(*this, filename)) {
			IO::print("Loaded KTX1 cubemap Sky '{}' as {}x{} equirectangular linear float data.\n"_sv, filename, width, height);
			return;
		}

		IO::print("Unable to load KTX1 Sky '{}'. Only uncompressed RGBA8/SRGB8_ALPHA8 cubemaps are supported.\n"_sv, filename);
		IO::exit(1);
	}

	int channels;
	float * hdr = stbi_loadf(filename.data(), &width, &height, &channels, STBI_rgb);

	if (!hdr || width == 0 || height == 0) {
		IO::print("Unable to load hdr Sky from file '{}'!\n"_sv, filename);
		IO::exit(1);
	}

	// Allocate data and copy it over from the file
	data.resize(width * height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = x + y * width;
			data[index] = Vector4(
				hdr[3*index + 0],
				hdr[3*index + 1],
				hdr[3*index + 2],
				0.0f
			);
		}
	}

	stbi_image_free(hdr);
}
