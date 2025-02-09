#include "cuwfrt/cuda_api.h"
#include "cuwfrt/cuda_error.cuh"
#include "cuwfrt/util/async_job.h"

#include "gpu_scene.cuh"
#include "scene.cuh"

namespace cuwfrt
{

void GPUData::Init(const Scene* cpu_scene)
{
    // Build BVH asynchronously
    std::unique_ptr<BVH> bvh;
    auto j = RunAsync([cpu_scene, &bvh]() {
        bvh = std::make_unique<BVH>(cpu_scene);
        return true;
    });

    auto vectors = cpu_scene->materials.get_vectors();

    int32 offsets[Materials::count];
    int32 total_size = 0;

    for (int32 i = 0; i < vectors.size(); ++i)
    {
        offsets[i] = total_size;
        total_size += int32(vectors[i].size());
    }

    size_t offsets_size = sizeof(int32) * Materials::count;
    cudaCheck(cudaMalloc(&scene.offsets, offsets_size));
    cudaCheck(cudaMemcpyAsync(scene.offsets, offsets, offsets_size, cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&scene.materials, total_size));
    for (int32 i = 0; i < Materials::count; ++i)
    {
        size_t size = vectors[i].size();
        if (size > 0)
        {
            cudaCheck(cudaMemcpyAsync(scene.materials + offsets[i], vectors[i].data(), size, cudaMemcpyHostToDevice));
        }
    }

    size_t position_size = sizeof(Point3) * cpu_scene->positions.size();
    cudaCheck(cudaMalloc(&scene.positions, position_size));
    cudaCheck(cudaMemcpyAsync(scene.positions, cpu_scene->positions.data(), position_size, cudaMemcpyHostToDevice));

    size_t normal_size = sizeof(Vec3) * cpu_scene->normals.size();
    cudaCheck(cudaMalloc(&scene.normals, normal_size));
    cudaCheck(cudaMemcpyAsync(scene.normals, cpu_scene->normals.data(), normal_size, cudaMemcpyHostToDevice));

    size_t tangent_size = sizeof(Vec3) * cpu_scene->tangents.size();
    cudaCheck(cudaMalloc(&scene.tangents, tangent_size));
    cudaCheck(cudaMemcpyAsync(scene.tangents, cpu_scene->tangents.data(), tangent_size, cudaMemcpyHostToDevice));

    size_t texcoord_size = sizeof(Vec2) * cpu_scene->texcoords.size();
    cudaCheck(cudaMalloc(&scene.texcoords, texcoord_size));
    cudaCheck(cudaMemcpyAsync(scene.texcoords, cpu_scene->texcoords.data(), texcoord_size, cudaMemcpyHostToDevice));

    size_t material_indices_size = sizeof(MaterialIndex) * cpu_scene->material_indices.size();
    cudaCheck(cudaMalloc(&scene.material_indices, material_indices_size));
    cudaCheck(
        cudaMemcpyAsync(scene.material_indices, cpu_scene->material_indices.data(), material_indices_size, cudaMemcpyHostToDevice)
    );

    size_t indices_size = sizeof(Vec3i) * cpu_scene->indices.size();
    cudaCheck(cudaMalloc(&scene.indices, indices_size));
    cudaCheck(cudaMemcpyAsync(scene.indices, cpu_scene->indices.data(), indices_size, cudaMemcpyHostToDevice));

    size_t light_indices_size = sizeof(PrimitiveIndex) * cpu_scene->light_indices.size();
    cudaCheck(cudaMalloc(&scene.light_indices, light_indices_size));
    cudaCheck(cudaMemcpyAsync(scene.light_indices, cpu_scene->light_indices.data(), light_indices_size, cudaMemcpyHostToDevice));

    // Create textures on GPU memory
    std::vector<cudaTextureObject_t> temp_tex_objs;
    temp_tex_objs.reserve(cpu_scene->textures.size());
    textures.reserve(cpu_scene->textures.size());

    for (const TextureDesc& td : cpu_scene->textures)
    {
        Texture& t = textures.emplace_back(td);
        temp_tex_objs.emplace_back(t.tex_obj);
    }

    size_t textures_size = sizeof(cudaTextureObject_t) * cpu_scene->textures.size();
    cudaCheck(cudaMalloc(&scene.tex_objs, textures_size));
    cudaCheck(cudaMemcpyAsync(scene.tex_objs, temp_tex_objs.data(), textures_size, cudaMemcpyHostToDevice));

    j->Wait();

    size_t bvh_primitives_size = sizeof(PrimitiveIndex) * bvh->primitives.size();
    cudaCheck(cudaMalloc(&scene.bvh_primitives, bvh_primitives_size));
    cudaCheck(cudaMemcpyAsync(scene.bvh_primitives, bvh->primitives.data(), bvh_primitives_size, cudaMemcpyHostToDevice));

    size_t bvh_nodes_size = sizeof(LinearBVHNode) * bvh->node_count;
    cudaCheck(cudaMalloc(&scene.bvh_nodes, bvh_nodes_size));
    cudaCheck(cudaMemcpyAsync(scene.bvh_nodes, bvh->nodes, bvh_nodes_size, cudaMemcpyHostToDevice));

    cudaCheck(cudaDeviceSynchronize());
}

void GPUData::Free()
{
    cudaCheck(cudaFree(scene.materials));
    cudaCheck(cudaFree(scene.offsets));

    cudaCheck(cudaFree(scene.positions));
    cudaCheck(cudaFree(scene.normals));
    cudaCheck(cudaFree(scene.tangents));
    cudaCheck(cudaFree(scene.texcoords));
    cudaCheck(cudaFree(scene.material_indices));
    cudaCheck(cudaFree(scene.indices));
    cudaCheck(cudaFree(scene.light_indices));

    cudaCheck(cudaFree(scene.bvh_primitives));
    cudaCheck(cudaFree(scene.bvh_nodes));

    cudaCheck(cudaFree(scene.tex_objs));
}

} // namespace cuwfrt
