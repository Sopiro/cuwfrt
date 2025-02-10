#include "alzartak/camera.h"
#include "alzartak/window.h"

#include "cuwfrt/raytracer.cuh"

#include "cuwfrt/scene/builder.cuh"
#include "cuwfrt/util/parallel.h"

#include "cuwfrt/kernel/kernel_ao.cuh"
#include "cuwfrt/kernel/kernel_debug.cuh"
#include "cuwfrt/kernel/kernel_pt_naive.cuh"
#include "cuwfrt/kernel/kernel_pt_nee.cuh"

using namespace alzartak;

namespace cuwfrt
{

static Window* window;
static RayTracer* raytracer;

static Scene scene;
static Camera camera;
static Options options;

static Camera3D player;

static int32 time = 0;
static int32 max_samples = 64;
static Float vfov = 71;
static Float aperture = 0;
static Float focus_dist = 1;

static const int32 num_kernels = 5;
static const char* name[num_kernels] = { "Gradient", "Normal", "AO", "Pathtrace Naive", "Pathtrace NEE" };
static Kernel* kernels[num_kernels] = { RenderGradient, RenderNormal, RaytraceAO, PathTraceNaive, PathTraceNEE };
static int32 selection = 4;

static Vec3 GetForward()
{
    float pitch = player.rotation.x;
    float yaw = player.rotation.y;

    Vec3 forward;
    forward.x = std::cos(pitch) * std::sin(yaw);
    forward.y = -std::sin(pitch);
    forward.z = std::cos(pitch) * std::cos(yaw);

    return Normalize(forward);
}

static void Update(Float dt)
{
    ++time;

    window->BeginFrame(GL_COLOR_BUFFER_BIT);

    ImGuiIO& io = ImGui::GetIO();

    // ImGui::ShowDemoWindow();

    ImGui::SetNextWindowPos({ 4, 4 }, ImGuiCond_Once, { 0.0f, 0.0f });
    if (ImGui::Begin("cuwfrt", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("%d fps", int32(io.Framerate));
        ImGui::Text("%d samples", std::min(time, max_samples));
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderInt("max bounces", &options.max_bounces, 0, 64)) time = 0;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderInt("max samples", &max_samples, 1, 1024)) time = 0;
        ImGui::Separator();
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("vfov", &vfov, 1.0f, 130.0f)) time = 0;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("aperture", &aperture, 0.0f, 0.1f)) time = 0;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("focus", &focus_dist, 0.0f, 10.0f)) time = 0;
        if (ImGui::Button("Reset camera", { 100, 0 }))
        {
            vfov = 71;
            aperture = 0;
            focus_dist = 1;
            time = 0;
        }
        ImGui::Separator();
        if (ImGui::Checkbox("Render sky", &options.render_sky)) time = 0;
        if (ImGui::Combo("", &selection, name, num_kernels)) time = 0;
    }
    ImGui::End();

    if (player.UpdateInput(dt))
    {
        if (Length2(player.velocity) < 0.5f)
        {
            player.velocity.SetZero();
        }
        time = 0;
    }

    camera = Camera(player.position, GetForward(), y_axis, vfov, aperture, focus_dist, window->GetWindowSize());
    if (time < max_samples)
    {
        raytracer->RayTrace(kernels[selection], time);
    }

    raytracer->DrawFrame();

    window->EndFrame();
}

static void BuildScene()
{
    for (int32 j = 0; j < 1; ++j)
    {
        for (int32 i = 0; i < 1; ++i)
        {
            CreateCornellBox(scene, Transform{ Point3(i * 1.1f, j * 1.1f, 0) });
        }
    }
}

// Initialize PBO & CUDA interop capability
static void Init()
{
    ThreadPool::global_thread_pool.reset(new ThreadPool(std::thread::hardware_concurrency()));

    window = Window::Init(1024, 768, "cuda RTRT");

    // Enable culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // Enable blend
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    BuildScene();

    player.position.Set(0.5, 0.5, 1.0f);
    player.speed = 1.5f;
    player.damping = 100.0f;

    options.max_bounces = 3;

    raytracer = new RayTracer(window, &scene, &camera, &options);

    scene.Clear();
}

static void Terminate()
{
    delete raytracer;
}

} // namespace cuwfrt

int main()
{
#if defined(_WIN32) && defined(_DEBUG)
    // Enable memory-leak reports
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    using namespace cuwfrt;

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
            Update(delta_time);
            delta_time = 0;
        }
    }

    Terminate();

    return 0;
}
