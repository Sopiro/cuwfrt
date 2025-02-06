#include "raytracer.h"

#include "alzartak/camera.h"
#include "alzartak/window.h"
#include "builder.h"

using namespace alzartak;

namespace cuwfrt
{

static Window* window;
static RayTracer* raytracer;

static Scene scene;
static Camera camera;

void Update(Float dt)
{
    window->BeginFrame(GL_COLOR_BUFFER_BIT);

    raytracer->Update();

    window->EndFrame();
}

// Initialize PBO & CUDA interop capability
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

    MaterialIndex white = scene.AddMaterial({ .reflectance{ .73f, .73f, .73f } });
    MaterialIndex red = scene.AddMaterial({ .reflectance{ .65f, .05f, .05f } });
    MaterialIndex green = scene.AddMaterial({ .reflectance{ .12f, .45f, .15f } });

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

            // auto mat = scene.CreateMaterial<ThinDielectricMaterial>(1.5f);

            tf = Transform{ 0.66f, hy, -0.33f, Quat(DegToRad(-18.0f), y_axis), Vec3(hx * 2.0f, hy * 2.0f, hz * 2.0f) };
            CreateBox(scene, tf, white);
        }
    }

    Point3 lookfrom{ 0.5f, 0.5f, 2.05f };
    Point3 lookat{ 0.5f, 0.5f, 0.0f };

    Float dist_to_focus = Dist(lookfrom, lookat);
    Float aperture = 0.0f;
    Float vFov = 71.0f;

    camera = Camera(lookfrom, lookat, y_axis, vFov, aperture, dist_to_focus, window->GetWindowSize());

    raytracer = new RayTracer(&scene, &camera);
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
            delta_time -= target_frame_time;
        }
    }

    Terminate();

    return 0;
}
