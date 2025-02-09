#include "builder.cuh"

namespace cuwfrt
{

void CreateRectXY(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc)
{
    Point3 p0 = { -0.5, -0.5, 0.0 };
    Point3 p1 = { 0.5, -0.5, 0.0 };
    Point3 p2 = { 0.5, 0.5, 0.0 };
    Point3 p3 = { -0.5, 0.5, 0.0 };

    Vertex v0{ p0, z_axis, x_axis, Point2(0.0, 0.0) };
    Vertex v1{ p1, z_axis, x_axis, Point2(tc.x, 0.0) };
    Vertex v2{ p2, z_axis, x_axis, Point2(tc.x, tc.y) };
    Vertex v3{ p3, z_axis, x_axis, Point2(0.0, tc.y) };

    auto vertices = std::vector<Vertex>{ v0, v1, v2, v3 };
    auto indices = std::vector<int32>{ 0, 1, 2, 0, 2, 3 };

    Mesh m(vertices, indices, transform);
    scene.AddMesh(m, material);
}

void CreateRectXZ(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc)
{

    Point3 p0 = { -0.5, 0.0, 0.5 };
    Point3 p1 = { 0.5, 0.0, 0.5 };
    Point3 p2 = { 0.5, 0.0, -0.5 };
    Point3 p3 = { -0.5, 0.0, -0.5 };

    Vertex v0{ p0, y_axis, x_axis, Point2(0.0, 0.0) };
    Vertex v1{ p1, y_axis, x_axis, Point2(tc.x, 0.0) };
    Vertex v2{ p2, y_axis, x_axis, Point2(tc.x, tc.y) };
    Vertex v3{ p3, y_axis, x_axis, Point2(0.0, tc.y) };

    auto vertices = std::vector<Vertex>{ v0, v1, v2, v3 };
    auto indices = std::vector<int32>{ 0, 1, 2, 0, 2, 3 };

    Mesh m(vertices, indices, transform);
    scene.AddMesh(m, material);
}

void CreateRectYZ(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc)
{
    Point3 p0 = { 0.0, -0.5, 0.5 };
    Point3 p1 = { 0.0, -0.5, -0.5 };
    Point3 p2 = { 0.0, 0.5, -0.5 };
    Point3 p3 = { 0.0, 0.5, 0.5 };

    Vertex v0{ p0, x_axis, -z_axis, Point2(0.0, 0.0) };
    Vertex v1{ p1, x_axis, -z_axis, Point2(tc.x, 0.0) };
    Vertex v2{ p2, x_axis, -z_axis, Point2(tc.x, tc.y) };
    Vertex v3{ p3, x_axis, -z_axis, Point2(0.0, tc.y) };

    auto vertices = std::vector<Vertex>{ v0, v1, v2, v3 };
    auto indices = std::vector<int32>{ 0, 1, 2, 0, 2, 3 };

    Mesh m(vertices, indices, transform);
    scene.AddMesh(m, material);
}

void CreateBox(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc)
{
    /*
          7--------6
         /|       /|
        3--------2 |
        | 4------|-5
        |/       |/
        0--------1
    */
    Point3 p0 = { -0.5, -0.5, 0.5 };
    Point3 p1 = { 0.5, -0.5, 0.5 };
    Point3 p2 = { 0.5, 0.5, 0.5 };
    Point3 p3 = { -0.5, 0.5, 0.5 };

    Point3 p4 = { -0.5, -0.5, -0.5 };
    Point3 p5 = { 0.5, -0.5, -0.5 };
    Point3 p6 = { 0.5, 0.5, -0.5 };
    Point3 p7 = { -0.5, 0.5, -0.5 };

    Vertex v00 = { p0, z_axis, x_axis, Point2(0.0, 0.0) };
    Vertex v01 = { p1, z_axis, x_axis, Point2(tc.x, 0.0) };
    Vertex v02 = { p2, z_axis, x_axis, Point2(tc.x, tc.y) };
    Vertex v03 = { p3, z_axis, x_axis, Point2(0.0, tc.y) };

    Vertex v04 = { p1, x_axis, -z_axis, Point2(0.0, 0.0) };
    Vertex v05 = { p5, x_axis, -z_axis, Point2(tc.x, 0.0) };
    Vertex v06 = { p6, x_axis, -z_axis, Point2(tc.x, tc.y) };
    Vertex v07 = { p2, x_axis, -z_axis, Point2(0.0, tc.y) };

    Vertex v08 = { p5, -z_axis, -x_axis, Point2(0.0, 0.0) };
    Vertex v09 = { p4, -z_axis, -x_axis, Point2(tc.x, 0.0) };
    Vertex v10 = { p7, -z_axis, -x_axis, Point2(tc.x, tc.y) };
    Vertex v11 = { p6, -z_axis, -x_axis, Point2(0.0, tc.y) };

    Vertex v12 = { p4, -x_axis, z_axis, Point2(0.0, 0.0) };
    Vertex v13 = { p0, -x_axis, z_axis, Point2(tc.x, 0.0) };
    Vertex v14 = { p3, -x_axis, z_axis, Point2(tc.x, tc.y) };
    Vertex v15 = { p7, -x_axis, z_axis, Point2(0.0, tc.y) };

    Vertex v16 = { p3, y_axis, x_axis, Point2(0.0, 0.0) };
    Vertex v17 = { p2, y_axis, x_axis, Point2(tc.x, 0.0) };
    Vertex v18 = { p6, y_axis, x_axis, Point2(tc.x, tc.y) };
    Vertex v19 = { p7, y_axis, x_axis, Point2(0.0, tc.y) };

    Vertex v20 = { p1, -y_axis, -x_axis, Point2(0.0, 0.0) };
    Vertex v21 = { p0, -y_axis, -x_axis, Point2(tc.x, 0.0) };
    Vertex v22 = { p4, -y_axis, -x_axis, Point2(tc.x, tc.y) };
    Vertex v23 = { p5, -y_axis, -x_axis, Point2(0.0, tc.y) };

    auto vertices = std::vector<Vertex>{ v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11,
                                         v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23 };

    // clang-format off
    auto indices = std::vector<int32>{
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        8, 9, 10, 8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23
    };
    // clang-format on

    Mesh m(vertices, indices, transform);
    scene.AddMesh(m, material);
}

void CreateCornellBox(Scene& scene, const Transform& o)
{
    static TextureIndex wak_tex = scene.AddTexture({ .filename = "C:/Users/sopir/Desktop/assets/wakdu.jpg", .non_color = false });
    static MaterialIndex wak = scene.AddMaterial<DiffuseMaterial>(wak_tex);
    static MaterialIndex white = scene.AddMaterial<DiffuseMaterial>(Vec3{ .73f, .73f, .73f });
    static MaterialIndex red = scene.AddMaterial<DiffuseMaterial>(Vec3{ .65f, .05f, .05f });
    static MaterialIndex green = scene.AddMaterial<DiffuseMaterial>(Vec3{ .12f, .45f, .15f });
    static MaterialIndex light = scene.AddMaterial<DiffuseLightMaterial>(Vec3{ 15.0f, 15.0f, 15.0f });
    // static MaterialIndex mirror = scene.AddMaterial<MirrorMaterial>(Vec3{ 0.8f });

    // The Cornell box
    {
        // front
        auto tf = Transform{ Vec3(0.5f, 0.5f, -1.0f), identity, Vec3(1.0f) };
        CreateRectXY(scene, o * tf, wak);

        // left
        tf = Transform{ Vec3(0.0f, 0.5f, -0.5f), identity, Vec3(1.0f) };
        CreateRectYZ(scene, o * tf, red);

        // right
        tf = Transform{ Vec3(1.0f, 0.5f, -0.5f), Quat(pi, y_axis), Vec3(1.0f) };
        CreateRectYZ(scene, o * tf, green);

        // bottom
        tf = Transform{ Vec3(0.5f, 0.0f, -0.5f), identity, Vec3(1.0f) };
        CreateRectXZ(scene, o * tf, white);

        // top
        tf = Transform{ Vec3(0.5f, 1.0f, -0.5f), Quat(pi, x_axis), Vec3(1.0f) };
        CreateRectXZ(scene, o * tf, white);

        // Left block
        {
            Float hx = 0.14f;
            Float hy = 0.28f;
            Float hz = 0.14f;

            tf = Transform{ 0.33f, hy, -0.66f, Quat(DegToRad(18.0f), y_axis), Vec3(hx * 2.0f, hy * 2.0f, hz * 2.0f) };
            CreateBox(scene, o * tf, white);
        }

        // Right block
        {
            Float hx = 0.14f;
            Float hy = 0.14f;
            Float hz = 0.14f;

            tf = Transform{ 0.66f, hy, -0.33f, Quat(DegToRad(-18.0f), y_axis), Vec3(hx * 2.0f, hy * 2.0f, hz * 2.0f) };
            CreateBox(scene, o * tf, white);
        }

        // Lights
        {
            tf = Transform{ 0.5f, 0.995f, -0.5f, Quat(pi, x_axis), Vec3(0.25f) };
            CreateRectXZ(scene, o * tf, light);
        }
    }
}

} // namespace cuwfrt
