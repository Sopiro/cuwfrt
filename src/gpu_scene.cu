#include "cuda_api.h"
#include "cuda_error.cuh"

#include "gpu_scene.cuh"
#include "scene.h"

namespace cuwfrt
{

void GPUScene::Init(const Scene* scene)
{
    // Build BVH
    BVH bvh(scene);

    size_t material_size = sizeof(Material) * scene->materials.size();
    cudaCheck(cudaMalloc(&materials, material_size));
    cudaCheck(cudaMemcpyAsync(materials, scene->materials.data(), material_size, cudaMemcpyHostToDevice));

    size_t position_size = sizeof(Vec3) * scene->positions.size();
    cudaCheck(cudaMalloc(&positions, position_size));
    cudaCheck(cudaMemcpyAsync(positions, scene->positions.data(), position_size, cudaMemcpyHostToDevice));

    size_t normal_size = sizeof(Vec3) * scene->normals.size();
    cudaCheck(cudaMalloc(&normals, normal_size));
    cudaCheck(cudaMemcpyAsync(normals, scene->normals.data(), normal_size, cudaMemcpyHostToDevice));

    size_t tangent_size = sizeof(Vec3) * scene->tangents.size();
    cudaCheck(cudaMalloc(&tangents, tangent_size));
    cudaCheck(cudaMemcpyAsync(tangents, scene->tangents.data(), tangent_size, cudaMemcpyHostToDevice));

    size_t texcoord_size = sizeof(Vec2) * scene->texcoords.size();
    cudaCheck(cudaMalloc(&texcoords, texcoord_size));
    cudaCheck(cudaMemcpyAsync(texcoords, scene->texcoords.data(), texcoord_size, cudaMemcpyHostToDevice));

    size_t material_indices_size = sizeof(MaterialIndex) * scene->material_indices.size();
    cudaCheck(cudaMalloc(&material_indices, material_indices_size));
    cudaCheck(cudaMemcpyAsync(material_indices, scene->material_indices.data(), material_indices_size, cudaMemcpyHostToDevice));

    size_t indices_size = sizeof(Vec3i) * scene->indices.size();
    cudaCheck(cudaMalloc(&indices, indices_size));
    cudaCheck(cudaMemcpyAsync(indices, scene->indices.data(), indices_size, cudaMemcpyHostToDevice));

    size_t light_indices_size = sizeof(int32) * scene->light_indices.size();
    cudaCheck(cudaMalloc(&light_indices, light_indices_size));
    cudaCheck(cudaMemcpyAsync(light_indices, scene->light_indices.data(), light_indices_size, cudaMemcpyHostToDevice));

    size_t bvh_primitives_size = sizeof(PrimitiveIndex) * bvh.primitives.size();
    cudaCheck(cudaMalloc(&bvh_primitives, bvh_primitives_size));
    cudaCheck(cudaMemcpyAsync(bvh_primitives, bvh.primitives.data(), bvh_primitives_size, cudaMemcpyHostToDevice));

    size_t bvh_nodes_size = sizeof(LinearBVHNode) * bvh.node_count;
    cudaCheck(cudaMalloc(&bvh_nodes, bvh_nodes_size));
    cudaCheck(cudaMemcpyAsync(bvh_nodes, bvh.nodes, bvh_nodes_size, cudaMemcpyHostToDevice));

    cudaCheck(cudaDeviceSynchronize());
}

void GPUScene::Free()
{
    cudaCheck(cudaFree(materials));
    cudaCheck(cudaFree(positions));
    cudaCheck(cudaFree(normals));
    cudaCheck(cudaFree(tangents));
    cudaCheck(cudaFree(texcoords));
    cudaCheck(cudaFree(material_indices));
    cudaCheck(cudaFree(indices));
    cudaCheck(cudaFree(light_indices));

    cudaCheck(cudaFree(bvh_primitives));
    cudaCheck(cudaFree(bvh_nodes));
}

} // namespace cuwfrt
