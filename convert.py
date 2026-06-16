# Convert Gaussian Splatting .ply files, or approximate ordinary .glb meshes,
# into the compact .splat format used by this viewer.
#
# GLB conversion is a mesh surface sampling fallback. It does not recover the
# trained Gaussian parameters that are present in a real 3DGS .ply file.

import argparse
import json
import struct
from io import BytesIO
from pathlib import Path

import numpy as np


COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

ACCESSOR_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def write_splat(buffer, position, scales, color, rotation):
    buffer.write(np.asarray(position, dtype=np.float32).tobytes())
    buffer.write(np.asarray(scales, dtype=np.float32).tobytes())
    buffer.write((np.asarray(color) * 255).clip(0, 255).astype(np.uint8).tobytes())
    buffer.write(
        ((np.asarray(rotation) / np.linalg.norm(rotation)) * 128 + 128)
        .clip(0, 255)
        .astype(np.uint8)
        .tobytes()
    )


def process_ply_to_splat(ply_file_path):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise SystemExit(
            "PLY conversion requires the 'plyfile' package. Install it with: "
            "python3 -m pip install plyfile"
        ) from exc

    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        write_splat(buffer, position, scales, color, rot)

    return buffer.getvalue()


def read_glb(glb_file_path):
    with open(glb_file_path, "rb") as f:
        data = f.read()

    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"{glb_file_path} is not a binary glTF/GLB file")

    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError("Only GLB version 2 is supported")
    if length != len(data):
        raise ValueError("GLB length header does not match file size")

    offset = 12
    gltf = None
    bin_chunk = b""
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk

    if gltf is None:
        raise ValueError("GLB is missing its JSON chunk")

    return gltf, bin_chunk


def accessor_array(gltf, bin_chunk, accessor_index):
    accessor = gltf["accessors"][accessor_index]
    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    component_dtype = np.dtype(COMPONENT_DTYPE[accessor["componentType"]])
    components = ACCESSOR_COMPONENTS[accessor["type"]]
    count = accessor["count"]

    byte_offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    byte_stride = buffer_view.get("byteStride", component_dtype.itemsize * components)
    item_size = component_dtype.itemsize * components

    if byte_stride == item_size:
        arr = np.frombuffer(
            bin_chunk,
            dtype=component_dtype,
            count=count * components,
            offset=byte_offset,
        ).reshape(count, components)
    else:
        arr = np.empty((count, components), dtype=component_dtype)
        for i in range(count):
            start = byte_offset + i * byte_stride
            arr[i] = np.frombuffer(
                bin_chunk,
                dtype=component_dtype,
                count=components,
                offset=start,
            )

    if accessor.get("normalized", False):
        arr = normalize_accessor_values(arr, component_dtype)

    return arr


def normalize_accessor_values(arr, dtype):
    if dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if dtype == np.int8:
        return np.maximum(arr.astype(np.float32) / 127.0, -1.0)
    if dtype == np.int16:
        return np.maximum(arr.astype(np.float32) / 32767.0, -1.0)
    return arr


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T

    translation = np.array(node.get("translation", [0, 0, 0]), dtype=np.float32)
    scale = np.array(node.get("scale", [1, 1, 1]), dtype=np.float32)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rotation = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    transform = rotation @ np.diag([scale[0], scale[1], scale[2], 1]).astype(np.float32)
    transform[:3, 3] = translation
    return transform


def mesh_nodes(gltf):
    scene_index = gltf.get("scene", 0)
    scene = gltf.get("scenes", [{}])[scene_index]
    roots = scene.get("nodes", list(range(len(gltf.get("nodes", [])))))

    results = []

    def visit(node_index, parent_transform):
        node = gltf["nodes"][node_index]
        transform = parent_transform @ node_matrix(node)
        if "mesh" in node:
            results.append((node["mesh"], transform))
        for child in node.get("children", []):
            visit(child, transform)

    for root in roots:
        visit(root, np.eye(4, dtype=np.float32))

    return results


def material_color(gltf, material_index):
    if material_index is None:
        return np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
    material = gltf.get("materials", [])[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    return np.array(pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0]), dtype=np.float32)


def primitive_triangles(gltf, bin_chunk, primitive, transform):
    if primitive.get("mode", 4) != 4:
        return None

    attributes = primitive.get("attributes", {})
    if "POSITION" not in attributes:
        return None

    positions = accessor_array(gltf, bin_chunk, attributes["POSITION"]).astype(np.float32)
    positions_h = np.c_[positions, np.ones(len(positions), dtype=np.float32)]
    positions = (positions_h @ transform.T)[:, :3]

    if "COLOR_0" in attributes:
        colors = accessor_array(gltf, bin_chunk, attributes["COLOR_0"]).astype(np.float32)
        if colors.shape[1] == 3:
            colors = np.c_[colors, np.ones(len(colors), dtype=np.float32)]
    else:
        colors = np.tile(
            material_color(gltf, primitive.get("material")),
            (len(positions), 1),
        )

    if "indices" in primitive:
        indices = accessor_array(gltf, bin_chunk, primitive["indices"]).reshape(-1)
    else:
        indices = np.arange(len(positions), dtype=np.uint32)

    triangles = indices.reshape(-1, 3)
    return positions[triangles], colors[triangles]


def collect_glb_triangles(gltf, bin_chunk):
    triangle_positions = []
    triangle_colors = []

    for mesh_index, transform in mesh_nodes(gltf):
        mesh = gltf["meshes"][mesh_index]
        for primitive in mesh.get("primitives", []):
            result = primitive_triangles(gltf, bin_chunk, primitive, transform)
            if result is None:
                continue
            positions, colors = result
            triangle_positions.append(positions)
            triangle_colors.append(colors)

    if not triangle_positions:
        raise ValueError("No triangle mesh primitives found in GLB")

    return np.concatenate(triangle_positions), np.concatenate(triangle_colors)


def sample_triangle_surface(triangles, colors, sample_count, rng):
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edges_a, edges_b), axis=1) * 0.5
    valid = areas > 0
    triangles = triangles[valid]
    colors = colors[valid]
    areas = areas[valid]

    if len(triangles) == 0:
        raise ValueError("GLB mesh has no non-degenerate triangles")

    probabilities = areas / areas.sum()
    triangle_indices = rng.choice(len(triangles), size=sample_count, p=probabilities)
    chosen = triangles[triangle_indices]
    chosen_colors = colors[triangle_indices]

    u = rng.random(sample_count, dtype=np.float32)
    v = rng.random(sample_count, dtype=np.float32)
    flip = u + v > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    w = 1 - u - v

    points = (
        chosen[:, 0] * w[:, None]
        + chosen[:, 1] * u[:, None]
        + chosen[:, 2] * v[:, None]
    )
    sampled_colors = (
        chosen_colors[:, 0] * w[:, None]
        + chosen_colors[:, 1] * u[:, None]
        + chosen_colors[:, 2] * v[:, None]
    )

    return points, sampled_colors, areas.mean()


def process_glb_to_splat(
    glb_file_path,
    sample_count=200000,
    splat_scale=1.0,
    opacity=0.85,
    seed=1,
):
    gltf, bin_chunk = read_glb(glb_file_path)
    triangles, colors = collect_glb_triangles(gltf, bin_chunk)
    rng = np.random.default_rng(seed)
    points, colors, mean_area = sample_triangle_surface(
        triangles,
        colors,
        sample_count,
        rng,
    )

    radius = np.sqrt(mean_area / max(sample_count / len(triangles), 1)) * splat_scale
    scales = np.array([radius, radius, radius], dtype=np.float32)
    rotation = np.array([1, 0, 0, 0], dtype=np.float32)

    # Draw farther splats first for the viewer's initial load order.
    sorted_indices = np.argsort(points[:, 2])
    buffer = BytesIO()
    for idx in sorted_indices:
        color = np.array(
            [colors[idx, 0], colors[idx, 1], colors[idx, 2], opacity],
            dtype=np.float32,
        )
        write_splat(buffer, points[idx], scales, color, rotation)

    return buffer.getvalue()


def process_file_to_splat(input_file, args):
    suffix = Path(input_file).suffix.lower()
    if suffix == ".ply":
        return process_ply_to_splat(input_file)
    if suffix == ".glb":
        return process_glb_to_splat(
            input_file,
            sample_count=args.samples,
            splat_scale=args.splat_scale,
            opacity=args.opacity,
            seed=args.seed,
        )
    raise ValueError(f"Unsupported input extension '{suffix}'. Use .ply or .glb.")


def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)


def default_output_path(input_file):
    path = Path(input_file)
    return str(path.with_suffix(path.suffix + ".splat"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gaussian .ply files or ordinary .glb meshes to .splat."
    )
    parser.add_argument("input_files", nargs="+", help="Input .ply or .glb files.")
    parser.add_argument(
        "--output",
        "-o",
        default="output.splat",
        help="The output .splat file when converting one input.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200000,
        help="Number of mesh surface samples to generate for each .glb input.",
    )
    parser.add_argument(
        "--splat-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for generated .glb splat radii.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.85,
        help="Opacity for generated .glb splats, from 0 to 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used for .glb surface sampling.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("--samples must be greater than 0")
    if not 0 <= args.opacity <= 1:
        raise SystemExit("--opacity must be between 0 and 1")

    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        splat_data = process_file_to_splat(input_file, args)
        output_file = args.output if len(args.input_files) == 1 else default_output_path(input_file)
        save_splat_file(splat_data, output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
