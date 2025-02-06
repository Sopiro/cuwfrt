#include "raytracer.h"

#include "alzartak/window.h"

using namespace alzartak;

namespace cuwfrt
{

static Window* window;
static RayTracer* raytracer;

static Scene scene;
static Camera camera;

void Update()
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

    Material lambertian{};
    lambertian.is_light = false;
    lambertian.reflectance = Vec3(1);
    lambertian.metallic = 0;
    lambertian.roughness = 0;

    MaterialIndex mi = scene.AddMaterial(lambertian);

    Point3 p0 = { -0.5, -0.5, 0.0 };
    Point3 p1 = { 0.5, -0.5, 0.0 };
    Point3 p2 = { 0.5, 0.5, 0.0 };
    Point3 p3 = { -0.5, 0.5, 0.0 };

    Vertex v0{ p0, z_axis, x_axis, Point2(0.0, 0.0) };
    Vertex v1{ p1, z_axis, x_axis, Point2(1.0, 0.0) };
    Vertex v2{ p2, z_axis, x_axis, Point2(1.0, 1.0) };
    Vertex v3{ p3, z_axis, x_axis, Point2(0.0, 1.0) };

    auto vertices = std::vector<Vertex>{ v0, v1, v2, v3 };
    auto indices = std::vector<int32>{ 0, 1, 2, 0, 2, 3 };

    Mesh mesh(vertices, indices, identity);

    scene.AddMesh(mesh, mi);

    Point3 lookfrom{ 0.5f, 0.5f, 2.05f };
    Point3 lookat{ 0.5f, 0.5f, 0.0f };

    Float dist_to_focus = Dist(lookfrom, lookat);
    Float aperture = 0.0f;
    Float vFov = 28.0f;

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
            Update();
            delta_time -= target_frame_time;
        }
    }

    Terminate();

    return 0;
}
