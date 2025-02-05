#pragma once

#include "gpu_scene.cuh"
#include "quad_renderer.h"
#include "scene.h"

namespace cuwfrt
{

class Scene;

class RayTracer
{
public:
    RayTracer(Scene* scene);
    ~RayTracer();

    void Update();

private:
    void InitGPUResources();
    void FreeGPUResources();

    void RenderGPU();
    void UpdateTexture();
    void RenderQuad();

    Point2i res;

    GLuint pbo, texture;
    cudaGraphicsResource* cuda_pbo;

    QuadRenderer qr;

    Scene* scene;
    GPUScene gpu_scene;
};

} // namespace cuwfrt
