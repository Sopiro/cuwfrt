#include "api.h"

#include "alzartak/mesh.h"
#include "alzartak/mesh_shader.h"
#include "alzartak/window.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

using namespace alzartak;

Window* window;
const Point2i resolution(1280, 720);

GLuint pbo, texture;
cudaGraphicsResource* cuda_pbo;

const Vertex quad_vertices[4] = { { { -1.0f, -1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 0 } },
                                  { { 1.0f, -1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 0 } },
                                  { { -1.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1 } },
                                  { { 1.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 1 } } };
const int32 quad_indices[6] = { 0, 1, 2, 2, 1, 3 };

// CUDA Kernel: Writes colors to PBO
__kernel__ void KernelRender(Point3* pixels, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y)
    {
        return;
    }

    int index = y * res.x + x;
    pixels[index] = Point3(x / (float)res.x, y / (float)res.y, 128 / 255.0f); // Simple gradient
}

// Render to the PBO using CUDA
void RenderGPU()
{
    Point3* device_ptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo);
    cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &size, cuda_pbo);

    const dim3 threads(8, 8);
    const dim3 blocks((resolution.x + threads.x - 1) / threads.x, (resolution.y + threads.y - 1) / threads.y);

    KernelRender<<<blocks, threads>>>(device_ptr, resolution);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cuda_pbo);
}

// Copy PBO data to texture
void UpdateTexture()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, resolution.x, resolution.y, GL_RGB, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// OpenGL Rendering: Use PBO texture on a fullscreen quad
void RenderQuad(Mesh& quad, MeshShader& shader)
{
    shader.Use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    quad.Draw();
}

std::unique_ptr<Mesh> quad;
std::unique_ptr<MeshShader> shader;

void Update()
{
    window->BeginFrame(GL_COLOR_BUFFER_BIT);

    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos({ 4, 4 }, ImGuiCond_Once, { 0.0f, 0.0f });
    if (ImGui::Begin("alzartak", NULL))
    {
        ImGui::Text("%d fps", int32(io.Framerate));
    }
    ImGui::End();

    RenderGPU();
    UpdateTexture();

    RenderQuad(*quad, *shader);

    window->EndFrame();
}

// Initialize PBO & CUDA Interop
void Init()
{
    window = Window::Init(resolution.x, resolution.y, "cuda RTRT");

    // Enable culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // Enable blend
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    window->SetFramebufferSizeChangeCallback([&](int32 width, int32 height) -> void { glViewport(0, 0, width, height); });

    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, resolution.x * resolution.y * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);

    // Create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, resolution.x, resolution.y, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create fullscreen quad mesh & shader
    quad = std::make_unique<Mesh>(quad_vertices, quad_indices);

    shader = MeshShader::Create();
    shader->SetModelMatrix(identity);
    shader->SetViewMatrix(identity);
    shader->SetProjectionMatrix(identity);
}

void Terminate()
{
    cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    quad = nullptr;
    shader = nullptr;
}

int main()
{
#if defined(_WIN32) && defined(_DEBUG)
    // Enable memory-leak reports
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    Init();

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(Update, 0, 1);
#else
    auto last_time = std::chrono::steady_clock::now();
    const float target_frame_time = 1.0f / window->GetRefreshRate();
    float delta_time = target_frame_time;

    while (!window->ShouldClose())
    {
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> duration = current_time - last_time;
        float elapsed_time = duration.count();
        last_time = current_time;

        delta_time += elapsed_time;
        if (delta_time > target_frame_time)
        {
            Update();
            delta_time -= target_frame_time;
        }
    }
#endif

    Terminate();

    return 0;
}
