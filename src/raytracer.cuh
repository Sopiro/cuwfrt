#pragma once

#include "alzartak/window.h"

#include "camera.h"
#include "gpu_scene.cuh"
#include "quad_renderer.h"
#include "scene.h"

namespace cuwfrt
{

using namespace alzartak;

class Scene;

struct Options
{
    int32 max_bounces = 5;
};

class RayTracer
{
public:
    RayTracer(Window* window, Scene* scene, Camera* camera, Options* options);
    ~RayTracer();

    void RayTrace(int32 time);
    void DrawFrame();

private:
    void InitGPUResources();
    void FreeGPUResources();

    void CreateFrameBuffer();
    void DeleteFrameBuffer();

    void RenderGPU();
    void UpdateTexture();
    void RenderQuad();

    Window* window;
    Point2i res;

    GLuint pbo, texture;
    cudaGraphicsResource* cuda_pbo;
    Vec4* d_sample_buffer;
    Vec4* d_frame_buffer;

    QuadRenderer qr;

    Scene* scene;
    Camera* camera;
    Options* options;

    GPUScene gpu_scene;

    int32 time;
};

} // namespace cuwfrt
