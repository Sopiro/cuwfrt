#include "raytracer.h"

#include "alzartak/window.h"

static Window* window;
static RayTracer* raytracer;

void Update()
{
    window->BeginFrame(GL_COLOR_BUFFER_BIT);

    raytracer->Update();

    window->EndFrame();
}

// Initialize PBO & CUDA Interop
void Init()
{
    window = Window::Init(1280, 720, "cuda RTRT");

    window->SetFramebufferSizeChangeCallback([&](int32 width, int32 height) -> void { glViewport(0, 0, width, height); });

    // Enable culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // Enable blend
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    raytracer = new RayTracer();
}

void Terminate()
{
    delete raytracer;
}

int main()
{
#if defined(_WIN32) && defined(_DEBUG)
    // Enable memory-leak reports
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    Init();

    auto last_time = std::chrono::steady_clock::now();
    const float target_frame_time = 1.0f / window->GetRefreshRate();
    float delta_time = 0;

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

    Terminate();

    return 0;
}
