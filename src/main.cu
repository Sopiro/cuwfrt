#include "raytracer.cuh"

#include "alzartak/camera.h"
#include "alzartak/window.h"
#include "builder.h"
#include "parallel.h"

using namespace alzartak;

namespace cuwfrt
{

static Window* window;
static RayTracer* raytracer;

static Scene scene;
static Camera camera;

static Camera3D player;

static int32 time = 0;
static int32 max_samples = 64;

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

void Update(Float dt)
{
    ++time;

    window->BeginFrame(GL_COLOR_BUFFER_BIT);

    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos({ 4, 4 }, ImGuiCond_Once, { 0.0f, 0.0f });
    if (ImGui::Begin("cuwfrt", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("%d fps", int32(io.Framerate));
        ImGui::Text("%d samples", std::min(time, max_samples));
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderInt("max samples", &max_samples, 1, 1024))
        {
            time = 0;
        }
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

    camera = Camera(player.position, GetForward(), y_axis, 71, 0.0f, 1.0f, window->GetWindowSize());
    if (time < max_samples)
    {
        raytracer->RayTrace(time);
    }

    raytracer->DrawFrame();

    window->EndFrame();
}

// Initialize PBO & CUDA interop capability
void Init()
{
    ThreadPool::global_thread_pool.reset(new ThreadPool(std::thread::hardware_concurrency()));

    window = Window::Init(1280, 720, "cuda RTRT");

    // Enable culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // Enable blend
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    MaterialIndex white = scene.AddMaterial({ .reflectance{ .73f, .73f, .73f } });
    MaterialIndex red = scene.AddMaterial({ .reflectance{ .65f, .05f, .05f } });
    MaterialIndex green = scene.AddMaterial({ .reflectance{ .12f, .45f, .15f } });
    MaterialIndex light = scene.AddMaterial({ .is_light{ true }, .reflectance{ 15.0f, 15.0f, 15.0f } });

    // The Cornell box
    {
        // front
        auto tf = Transform{ Vec3(0.5f, 0.5f, -1.0f), identity, Vec3(1.0f) };
        CreateRectXY(scene, tf, white);

        // left
        tf = Transform{ Vec3(0.0f, 0.5f, -0.5f), identity, Vec3(1.0f) };
        CreateRectYZ(scene, tf, red);

        // right
        tf = Transform{ Vec3(1.0f, 0.5f, -0.5f), Quat(pi, y_axis), Vec3(1.0f) };
        CreateRectYZ(scene, tf, green);

        // bottom
        tf = Transform{ Vec3(0.5f, 0.0f, -0.5f), identity, Vec3(1.0f) };
        CreateRectXZ(scene, tf, white);

        // top
        tf = Transform{ Vec3(0.5f, 1.0f, -0.5f), Quat(pi, x_axis), Vec3(1.0f) };
        CreateRectXZ(scene, tf, white);

        // Left block
        {
            Float hx = 0.14f;
            Float hy = 0.28f;
            Float hz = 0.14f;

            tf = Transform{ 0.33f, hy, -0.66f, Quat(DegToRad(18.0f), y_axis), Vec3(hx * 2.0f, hy * 2.0f, hz * 2.0f) };
            CreateBox(scene, tf, white);
        }

        // Right block
        {
            Float hx = 0.14f;
            Float hy = 0.14f;
            Float hz = 0.14f;

            tf = Transform{ 0.66f, hy, -0.33f, Quat(DegToRad(-18.0f), y_axis), Vec3(hx * 2.0f, hy * 2.0f, hz * 2.0f) };
            CreateBox(scene, tf, white);
        }

        // Lights
        {
            tf = Transform{ 0.5f, 0.995f, -0.5f, Quat(pi, x_axis), Vec3(0.25f) };
            CreateRectXZ(scene, tf, light);
        }
    }

    player.position.Set(0.5f, 0.5f, 1.0f);
    player.speed = 1.5f;
    player.damping = 100.0f;
    raytracer = new RayTracer(window, &scene, &camera);
}

void Terminate()
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
