#include "cuda_api.h"
#include "cuda_error.cuh"

#include "gpu_scene.cuh"
#include "scene.h"

namespace cuwfrt
{

void GPUScene::Init(const Scene* scene)
{
    auto vectors = scene->materials.get_vectors();

    int32 offsets[Materials::count];
    int32 total_size = 0;

    for (int32 i = 0; i < vectors.size(); ++i)
    {
        offsets[i] = total_size;
        total_size += int32(vectors[i].size());
    }

    size_t offsets_size = sizeof(int32) * Materials::count;
    cudaCheck(cudaMalloc(&data.offsets, offsets_size));
    cudaCheck(cudaMemcpyAsync(data.offsets, offsets, offsets_size, cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&data.materials, total_size));
    for (int32 i = 0; i < Materials::count; ++i)
    {
        size_t size = vectors[i].size();
        if (size > 0)
        {
            cudaCheck(cudaMemcpyAsync(data.materials + offsets[i], vectors[i].data(), size, cudaMemcpyHostToDevice));
        }
    }

    size_t position_size = sizeof(Vec3) * scene->positions.size();
    cudaCheck(cudaMalloc(&data.positions, position_size));
    cudaCheck(cudaMemcpyAsync(data.positions, scene->positions.data(), position_size, cudaMemcpyHostToDevice));

    size_t normal_size = sizeof(Vec3) * scene->normals.size();
    cudaCheck(cudaMalloc(&data.normals, normal_size));
    cudaCheck(cudaMemcpyAsync(data.normals, scene->normals.data(), normal_size, cudaMemcpyHostToDevice));

    size_t tangent_size = sizeof(Vec3) * scene->tangents.size();
    cudaCheck(cudaMalloc(&data.tangents, tangent_size));
    cudaCheck(cudaMemcpyAsync(data.tangents, scene->tangents.data(), tangent_size, cudaMemcpyHostToDevice));

    size_t texcoord_size = sizeof(Vec2) * scene->texcoords.size();
    cudaCheck(cudaMalloc(&data.texcoords, texcoord_size));
    cudaCheck(cudaMemcpyAsync(data.texcoords, scene->texcoords.data(), texcoord_size, cudaMemcpyHostToDevice));

    size_t material_indices_size = sizeof(MaterialIndex) * scene->material_indices.size();
    cudaCheck(cudaMalloc(&data.material_indices, material_indices_size));
    cudaCheck(
        cudaMemcpyAsync(data.material_indices, scene->material_indices.data(), material_indices_size, cudaMemcpyHostToDevice)
    );

    size_t indices_size = sizeof(Vec3i) * scene->indices.size();
    cudaCheck(cudaMalloc(&data.indices, indices_size));
    cudaCheck(cudaMemcpyAsync(data.indices, scene->indices.data(), indices_size, cudaMemcpyHostToDevice));

    size_t light_indices_size = sizeof(int32) * scene->light_indices.size();
    cudaCheck(cudaMalloc(&data.light_indices, light_indices_size));
    cudaCheck(cudaMemcpyAsync(data.light_indices, scene->light_indices.data(), light_indices_size, cudaMemcpyHostToDevice));

    // Create textures on GPU memory
    std::vector<cudaTextureObject_t> temp_tex_objs;
    temp_tex_objs.reserve(scene->textures.size());
    textures.reserve(scene->textures.size());

    for (const TextureDesc& td : scene->textures)
    {
        Texture& t = textures.emplace_back(td);
        temp_tex_objs.emplace_back(t.tex_obj);
    }

    size_t textures_size = sizeof(cudaTextureObject_t) * scene->textures.size();
    cudaCheck(cudaMalloc(&data.tex_objs, textures_size));
    cudaCheck(cudaMemcpyAsync(data.tex_objs, temp_tex_objs.data(), textures_size, cudaMemcpyHostToDevice));

    // Build BVH
    BVH bvh(scene);

    size_t bvh_primitives_size = sizeof(PrimitiveIndex) * bvh.primitives.size();
    cudaCheck(cudaMalloc(&data.bvh_primitives, bvh_primitives_size));
    cudaCheck(cudaMemcpyAsync(data.bvh_primitives, bvh.primitives.data(), bvh_primitives_size, cudaMemcpyHostToDevice));

    size_t bvh_nodes_size = sizeof(LinearBVHNode) * bvh.node_count;
    cudaCheck(cudaMalloc(&data.bvh_nodes, bvh_nodes_size));
    cudaCheck(cudaMemcpyAsync(data.bvh_nodes, bvh.nodes, bvh_nodes_size, cudaMemcpyHostToDevice));

    cudaCheck(cudaDeviceSynchronize());
}

void GPUScene::Free()
{
    cudaCheck(cudaFree(data.materials));
    cudaCheck(cudaFree(data.offsets));

    cudaCheck(cudaFree(data.positions));
    cudaCheck(cudaFree(data.normals));
    cudaCheck(cudaFree(data.tangents));
    cudaCheck(cudaFree(data.texcoords));
    cudaCheck(cudaFree(data.material_indices));
    cudaCheck(cudaFree(data.indices));
    cudaCheck(cudaFree(data.light_indices));

    cudaCheck(cudaFree(data.bvh_primitives));
    cudaCheck(cudaFree(data.bvh_nodes));

    cudaCheck(cudaFree(data.tex_objs));
}

} // namespace cuwfrt
