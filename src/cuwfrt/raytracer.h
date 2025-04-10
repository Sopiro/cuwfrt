#pragma once

#include "alzartak/window.h"

#include "scene/gpu_scene.h"
#include "scene/scene.cuh"

#include "wavefront.h"

#include "camera/camera.h"
#include "quad_renderer.h"

namespace cuwfrt
{

using namespace alzartak;

class Scene;

struct Options
{
    int32 max_bounces = 5;
    bool render_sky = true;
};

using Kernel = void(Vec4*, Vec4*, Point2i, GPUScene, Camera, Options, int32);

class RayTracer
{
public:
    static inline const int32 num_kernels = 6;
    static inline const char* kernel_name[num_kernels] = { "Gradient",        "Normal",        "AO",
                                                           "Pathtrace Naive", "Pathtrace NEE", "Wavefront" };

    RayTracer(Window* window, const Scene* scene, const Camera* camera, const Options* options);
    ~RayTracer();

    void RayTrace(int32 kernel_index, int32 time);
    void RayTraceWavefront(int32 time);
    void DrawFrame();

private:
    void InitGPUResources();
    void FreeGPUResources();

    void CreateFrameBuffer();
    void DeleteFrameBuffer();

    void UpdateTexture();
    void RenderQuad();

    void Resize(int32 width, int32 height);

    Window* window;
    Point2i res;

    GLuint pbo, texture;
    cudaGraphicsResource* cuda_pbo;
    Vec4* d_sample_buffer;
    Vec4* d_frame_buffer;

    QuadRenderer qr;

    const Scene* scene;
    const Camera* camera;
    const Options* options;

    cudaStream_t streams[2];
    GPUResources gpu_res;
    WavefrontResources wf;

    int32 time;
};

} // namespace cuwfrt
