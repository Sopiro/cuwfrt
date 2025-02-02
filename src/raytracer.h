#pragma once

#include "quad_renderer.h"

class RayTracer
{
public:
    RayTracer();
    ~RayTracer();

    void Update();

private:
    void RenderGPU();
    void UpdateTexture();
    void RenderQuad();

    Point2i res;

    GLuint pbo, texture;
    cudaGraphicsResource* cuda_pbo;

    QuadRenderer qr;
};