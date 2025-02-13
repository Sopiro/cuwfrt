#include "loader.cuh"

#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

namespace cuwfrt
{
static MaterialIndex fallback_material = { Material::TypeIndexOf<DiffuseMaterial>(), 0 };

void SetFallbackMaterial(MaterialIndex material_index)
{
    fallback_material = material_index;
}

static bool LoadMesh(Scene& scene, tinygltf::Model& model, tinygltf::Mesh& mesh, const Mat4& transform)
{
    for (int32 prim = 0; prim < mesh.primitives.size(); ++prim)
    {
        tinygltf::Primitive& primitive = mesh.primitives[prim];

        if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
        {
            continue;
        }

        int32 position_index = primitive.attributes.contains("POSITION") ? primitive.attributes["POSITION"] : -1;
        int32 normal_index = primitive.attributes.contains("NORMAL") ? primitive.attributes["NORMAL"] : -1;
        int32 tangent_index = primitive.attributes.contains("TANGENT") ? primitive.attributes["TANGENT"] : -1;
        int32 texcoord_index = primitive.attributes.contains("TEXCOORD_0") ? primitive.attributes["TEXCOORD_0"] : -1;
        int32 indices_index = int32(primitive.indices);

        if (position_index == -1 || normal_index == -1 || texcoord_index == -1)
        {
            return false;
        }

        // position
        tinygltf::Accessor& position_accessor = model.accessors[position_index];
        tinygltf::BufferView& position_buffer_view = model.bufferViews[position_accessor.bufferView];
        tinygltf::Buffer& position_buffer = model.buffers[position_buffer_view.buffer];
        // normal
        tinygltf::Accessor& normal_accessor = model.accessors[normal_index];
        tinygltf::BufferView& normal_buffer_view = model.bufferViews[normal_accessor.bufferView];
        tinygltf::Buffer& normal_buffer = model.buffers[normal_buffer_view.buffer];
        // texcoord
        tinygltf::Accessor& texcoord_accessor = model.accessors[texcoord_index];
        tinygltf::BufferView& texcoord_buffer_view = model.bufferViews[texcoord_accessor.bufferView];
        tinygltf::Buffer& texcoord_buffer = model.buffers[texcoord_buffer_view.buffer];
        // indices
        tinygltf::Accessor& indices_accessor = model.accessors[indices_index];
        tinygltf::BufferView& indices_buffer_view = model.bufferViews[indices_accessor.bufferView];
        tinygltf::Buffer& indices_buffer = model.buffers[indices_buffer_view.buffer];

        std::vector<Vec3> positions(position_accessor.count);
        std::vector<Vec3> normals(normal_accessor.count);
        std::vector<Vec3> tangents(normal_accessor.count);
        std::vector<Vec2> texcoords(texcoord_accessor.count);
        std::vector<int32> indices(indices_accessor.count);

        memcpy(positions.data(), position_buffer.data.data() + position_buffer_view.byteOffset, position_buffer_view.byteLength);
        memcpy(normals.data(), normal_buffer.data.data() + normal_buffer_view.byteOffset, normal_buffer_view.byteLength);
        memcpy(texcoords.data(), texcoord_buffer.data.data() + texcoord_buffer_view.byteOffset, texcoord_buffer_view.byteLength);

        if (tangent_index != -1)
        {
            // tangent
            tinygltf::Accessor& tangent_accessor = model.accessors[tangent_index];
            tinygltf::BufferView& tangent_buffer_view = model.bufferViews[tangent_accessor.bufferView];
            tinygltf::Buffer& tangent_buffer = model.buffers[tangent_buffer_view.buffer];
            memcpy(tangents.data(), tangent_buffer.data.data() + tangent_buffer_view.byteOffset, tangent_buffer_view.byteLength);
        }
        else
        {
            for (size_t i = 0; i < normals.size(); ++i)
            {
                Vec3 bitangent;
                CoordinateSystem(normals[i], &tangents[i], &bitangent);
            }
        }

        if (tinygltf::GetNumComponentsInType(indices_accessor.type) != 1)
        {
            return false;
        }

        switch (tinygltf::GetComponentSizeInBytes(indices_accessor.componentType))
        {
        case sizeof(uint8):
        {
            std::vector<uint8> temp;
            temp.resize(indices_accessor.count);
            memcpy(temp.data(), indices_buffer.data.data() + indices_buffer_view.byteOffset, indices_buffer_view.byteLength);
            for (size_t i = 0; i < indices_accessor.count; ++i)
            {
                indices[i] = int32(temp[i]);
            }
        }
        break;
        case sizeof(uint16):
        {
            std::vector<uint16> temp;
            temp.resize(indices_accessor.count);
            memcpy(temp.data(), indices_buffer.data.data() + indices_buffer_view.byteOffset, indices_buffer_view.byteLength);
            for (size_t i = 0; i < indices_accessor.count; ++i)
            {
                indices[i] = int32(temp[i]);
            }
        }
        break;
        case sizeof(uint32):
        {
            std::vector<uint32> temp;
            temp.resize(indices_accessor.count);
            memcpy(temp.data(), indices_buffer.data.data() + indices_buffer_view.byteOffset, indices_buffer_view.byteLength);
            for (size_t i = 0; i < indices_accessor.count; ++i)
            {
                indices[i] = int32(temp[i]);
            }
        }
        break;

        default:
            return false;
        }

        TriangleMesh tri_mesh(positions, normals, tangents, texcoords, indices, transform);
        scene.AddTriangleMesh(tri_mesh, fallback_material);
    }

    return true;
}

static void ProcessNode(Scene& my_scene, tinygltf::Model& model, tinygltf::Node& node, Mat4 parent)
{
    Mat4 transform =
        node.matrix.size() == 0
            ? identity
            : Mat4(
                  Vec4(float(node.matrix[0]), float(node.matrix[1]), float(node.matrix[2]), float(node.matrix[3])),
                  Vec4(float(node.matrix[4]), float(node.matrix[5]), float(node.matrix[6]), float(node.matrix[7])),
                  Vec4(float(node.matrix[8]), float(node.matrix[9]), float(node.matrix[10]), float(node.matrix[11])),
                  Vec4(float(node.matrix[12]), float(node.matrix[13]), float(node.matrix[14]), float(node.matrix[15]))
              );

    transform = Mul(parent, transform);

    if ((node.mesh >= 0) && (node.mesh < model.meshes.size()))
    {
        LoadMesh(my_scene, model, model.meshes[node.mesh], transform);
    }

    // Recursively process child nodes
    for (size_t i = 0; i < node.children.size(); i++)
    {
        WakAssert((node.children[i] >= 0) && (node.children[i] < model.nodes.size()));
        ProcessNode(my_scene, model, model.nodes[node.children[i]], transform);
    }
}

static void LoadScene(Scene& my_scene, tinygltf::Model& model, const Transform& transform)
{
    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    for (size_t i = 0; i < scene.nodes.size(); ++i)
    {
        WakAssert((scene.nodes[i] >= 0) && (scene.nodes[i] < model.nodes.size()));
        ProcessNode(my_scene, model, model.nodes[scene.nodes[i]], Mat4(transform));
    }
}

void LoadGLTF(Scene& scene, std::filesystem::path filename, const Transform& transform)
{
    tinygltf::TinyGLTF gltf;
    tinygltf::Model model;

    std::string err, warn;

    bool success;
    if (filename.extension() == ".gltf")
    {
        success = gltf.LoadASCIIFromFile(&model, &err, &warn, filename.string());
    }
    else if (filename.extension() == ".glb")
    {
        success = gltf.LoadBinaryFromFile(&model, &err, &warn, filename.string());
    }
    else
    {
        std::cout << "Faild to load model: " << filename.string() << std::endl;
        return;
    }

    if (!success)
    {
        std::cout << "gltf warning: " << warn << std::endl;
        std::cout << "gltf error: " << err << std::endl;
        return;
    }

    LoadScene(scene, model, transform);
}

} // namespace cuwfrt
