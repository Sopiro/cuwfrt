#pragma once

#include "alzartak/window.h"

#include "scene/gpu_scene.h"
#include "scene/scene.h"

#include "denoise.h"
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

using Kernel = void(Vec4*, Point2i, GPUScene, Camera, GBuffer, Options, int32);

class RayTracer
{
public:
    static const int32 num_kernels;
    static const char* kernel_names[];

    RayTracer(Window* window, const Scene* scene, const Camera* camera, const Options* options);
    ~RayTracer();

    void RayTrace(int32 kernel_index);
    void RayTraceWavefront();

    void ClearSamples();
    void AccumulateSamples();

    void Denoise();

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
    Vec4* frame_buffer;

    int32 frame_index, output_index;
    Vec4* sample_buffer[2];

    Camera g_camera[2];
    GBuffer g_buffer[2];
    HistoryBuffer h_buffer[2];

    int32 spp;
    Vec4* accumulation_buffer;

    QuadRenderer qr;

    const Scene* scene;
    const Camera* camera;
    const Options* options;

    std::array<cudaStream_t, 2> streams;
    std::array<cudaStream_t, Materials::count> ray_queue_streams;
    GPUResources gpu_res;
    WavefrontResources wf;
};

} // namespace cuwfrt
