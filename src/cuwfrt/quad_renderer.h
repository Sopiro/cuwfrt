#pragma once

#include "alzartak/mesh.h"
#include "alzartak/mesh_shader.h"

namespace cuwfrt
{

class QuadRenderer
{
public:
    QuadRenderer()
        : quad{ quad_vertices, quad_indices }
        , shader{}
    {
        shader.SetModelMatrix(identity);
        shader.SetViewMatrix(identity);
        shader.SetProjectionMatrix(identity);
    }

    void Draw() const
    {
        shader.Use();
        quad.Draw();
    }

private:
    alzartak::Mesh quad;
    alzartak::MeshShader shader;

    static inline alzartak::Vertex quad_vertices[4] = { { { -1.0f, -1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 0 } },
                                                        { { 1.0f, -1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 0 } },
                                                        { { -1.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1 } },
                                                        { { 1.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 1 } } };
    static inline int32 quad_indices[6] = { 0, 1, 2, 2, 1, 3 };
};

} // namespace cuwfrt
