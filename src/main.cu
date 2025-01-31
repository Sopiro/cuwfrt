#include "api.h"

#include "alzartak/mesh.h"
#include "alzartak/mesh_shader.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

using namespace alzartak;

GLFWwindow* window;
const int WIDTH = 1280;
const int HEIGHT = 720;

GLuint pbo, texture;
cudaGraphicsResource* cuda_pbo;

// Fullscreen quad vertices
const Vertex quad_vertices[4] = { { { -1.0f, -1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 0 } },
                                  { { 1.0f, -1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 0 } },
                                  { { -1.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1 } },
                                  { { 1.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 1 } } };
const int32 quad_indices[6] = { 0, 1, 2, 2, 1, 3 };

// CUDA Kernel: Writes colors to PBO
__kernel__ void KernelRender(float3* pixels, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
    {
        return;
    }

    int index = y * width + x;
    pixels[index] = make_float3(x / (float)width, y / (float)height, 128 / 255.0f); // Simple gradient
}

// Render to the PBO using CUDA
void RenderGPU()
{
    float3* device_ptr;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo);
    cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &size, cuda_pbo);

    const dim3 threads(8, 8);
    const dim3 blocks((WIDTH + threads.x - 1) / threads.x, (HEIGHT + threads.y - 1) / threads.y);

    KernelRender<<<blocks, threads>>>(device_ptr, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cuda_pbo);
}

// Copy PBO data to texture
void UpdateTexture()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_FLOAT, nullptr);
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

// Initialize PBO & CUDA Interop
void Init()
{
    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);

    // Create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Terminate()
{
    cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main()
{
    if (!glfwInit())
    {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "cuda RTRT", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        return -1;
    }

    Init();

    // Create fullscreen quad mesh & shader
    Mesh quad(quad_vertices, quad_indices);

    auto shader = MeshShader::Create();
    shader->SetModelMatrix(identity);
    shader->SetViewMatrix(identity);
    shader->SetProjectionMatrix(identity);

    // Main Loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Update PBO using CUDA
        RenderGPU();
        UpdateTexture();

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        RenderQuad(quad, *shader);
        glfwSwapBuffers(window);
    }

    // Cleanup
    Terminate();

    return 0;
}
