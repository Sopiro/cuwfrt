#include "alzartak/camera.h"

#include "cuwfrt/raytracer.h"

#include "cuwfrt/scene/builder.cuh"
#include "cuwfrt/util/parallel.h"

#include "cuwfrt/loader/loader.cuh"

#include "cuwfrt/cuda_error.cuh"

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

static int32 selection = 5;

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
    window->PollEvents();

    if (player.UpdateInput(dt))
    {
        if (Length2(player.velocity) < 0.5f)
        {
            player.velocity.SetZero();
        }
        time = 0;
    }
}

static void Render()
{
    window->BeginFrame(GL_COLOR_BUFFER_BIT);

    ImGuiIO& io = ImGui::GetIO();

    // ImGui::ShowDemoWindow();

    ImGui::SetNextWindowPos({ 4, 4 }, ImGuiCond_Once, { 0.0f, 0.0f });
    if (ImGui::Begin("cuwfrt", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("%d fps %.6fms", int32(io.Framerate), io.DeltaTime);
        ImGui::Text("%d samples", std::min(time + 1, max_samples));
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
        if (ImGui::Combo("##", &selection, RayTracer::kernel_name, RayTracer::num_kernels)) time = 0;
    }
    ImGui::End();

    camera = Camera(player.position, GetForward(), y_axis, vfov, aperture, focus_dist, window->GetWindowSize());
    if (time < max_samples)
    {
        if (selection > 4)
        {
            raytracer->RayTraceWavefront(time);
        }
        else
        {
            raytracer->RayTrace(selection, time);
        }
    }

    raytracer->DrawFrame();

    window->EndFrame();

    ++time;
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

    // static MaterialIndex mirror = scene.AddMaterial<MirrorMaterial>(Vec3{ .73f, .73f, .73f });
    // SetLoaderFallbackMaterial(mirror);

    // static MaterialIndex white = scene.AddMaterial<DiffuseMaterial>(Vec3{ .73f, .73f, .73f });
    // SetLoaderFallbackMaterial(white);

    static MaterialIndex glass = scene.AddMaterial<DielectricMaterial>(1.5f, Vec3(1.0f));
    SetLoaderFallbackMaterial(glass);

    LoadModel(
        scene, "Z:/dev/cpp_workspace/Bulbit/res/stanford/lucy.obj", Transform(Vec3(0.66f, 0.28f, -0.33f), identity, Vec3(0.5f))
    );

    // LoadModel(scene, "Z:/dev/cpp_workspace/Bulbit/res/sponza/glTF/Sponza.gltf", Transform(Vec3(0, 0, 0), identity,
    // Vec3(0.01f)));

    // LoadModel(scene, "C:/Users/sopir/Desktop/untitled.gltf", identity);
    // LoadModel(scene, "Z:/dev/cpp_workspace/Bulbit/res/stanford/bunny.obj", { Vec3(0), identity, Vec3(0.5f) });
}

static void SetImGuiStyle()
{
    // auto& io = ImGui::GetIO();
    // io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/arialbd.ttf", 16.0f, NULL, io.Fonts->GetGlyphRangesKorean());

    // Got from: https://github.com/ocornut/imgui/issues/707#issuecomment-468798935
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    /// 0 = FLAT APPEARENCE
    /// 1 = MORE "3D" LOOK
    int is3D = 0;

    colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_Border] = ImVec4(0.12f, 0.12f, 0.12f, 0.71f);
    colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.42f, 0.42f, 0.42f, 0.54f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.42f, 0.42f, 0.42f, 0.40f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.67f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.19f, 0.19f, 0.19f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.17f, 0.17f, 0.17f, 0.90f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.335f, 0.335f, 0.335f, 1.000f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.24f, 0.24f, 0.24f, 0.53f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.65f, 0.65f, 0.65f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.64f, 0.64f, 0.64f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.54f, 0.54f, 0.54f, 0.35f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.52f, 0.52f, 0.52f, 0.59f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.47f, 0.47f, 0.47f, 1.00f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.76f, 0.76f, 0.76f, 0.77f);
    colors[ImGuiCol_Separator] = ImVec4(0.000f, 0.000f, 0.000f, 0.137f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.700f, 0.671f, 0.600f, 0.290f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.702f, 0.671f, 0.600f, 0.674f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.73f, 0.73f, 0.73f, 0.35f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

    style.PopupRounding = 3;

    style.WindowPadding = ImVec2(4, 4);
    style.FramePadding = ImVec2(6, 4);
    style.ItemSpacing = ImVec2(6, 2);

    style.ScrollbarSize = 18;

    style.WindowBorderSize = 1;
    style.ChildBorderSize = 1;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = float(is3D);

    style.WindowRounding = 3;
    style.ChildRounding = 3;
    style.FrameRounding = 3;
    style.ScrollbarRounding = 2;
    style.GrabRounding = 3;

#ifdef IMGUI_HAS_DOCK
    style.TabBorderSize = is3D;
    style.TabRounding = 3;

    colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    colors[ImGuiCol_Tab] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_TabActive] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.85f, 0.85f, 0.85f, 0.28f);

    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
#endif
}

static void InitCudaDevice()
{
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(1);
    }

    std::cout << "Available CUDA devices: " << device_count << std::endl;

    int best_device = 0;
    int best_score = -1;

    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp props;
        cudaCheck(cudaGetDeviceProperties(&props, i));

        int score = 0;
        score += props.multiProcessorCount * 1000;
        score += int(props.totalGlobalMem) / (1024 * 1024 * 128);

        std::cout << i << ": " << props.name << " (SMs: " << props.multiProcessorCount
                  << ", Mem: " << (props.totalGlobalMem >> 20) << " MB)" << std::endl;

        if (score > best_score)
        {
            best_score = score;
            best_device = i;
        }
    }

    std::cout << "CUDA device set to " << best_device << std::endl;
    cudaCheck(cudaSetDevice(best_device));
}

static void Init()
{
    ThreadPool::global_thread_pool.reset(new ThreadPool(std::thread::hardware_concurrency()));

    InitCudaDevice();

    window = Window::Init(1280, 720, "cuda RTRT");

    SetImGuiStyle();

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

    options.max_bounces = 5;

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
    float passed_time = 0;

    while (!window->ShouldClose())
    {
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> duration = current_time - last_time;
        float elapsed_time = duration.count();
        passed_time += elapsed_time;
        last_time = current_time;

        if (passed_time > target_frame_time)
        {
            while (passed_time > target_frame_time)
            {
                Update(target_frame_time);
                passed_time -= target_frame_time;
            }
            Render();
        }
    }

    Terminate();

    return 0;
}
