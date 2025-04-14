#include "wavefront.h"

#include "cuwfrt/cuda_error.cuh"

namespace cuwfrt
{

void WavefrontResources::Init(Point2i res)
{
    ray_capacity = res.x * res.y;

    active.Init(ray_capacity);
    next.Init(ray_capacity);
    closest.Init(ray_capacity);
    miss.Init(ray_capacity);
    shadow.Init(ray_capacity);
}

void WavefrontResources::Free()
{
    active.Free();
    next.Free();
    closest.Free();
    miss.Free();
    shadow.Free();
}

void WavefrontResources::Resize(Point2i res)
{
    ray_capacity = res.x * res.y;

    active.Resize(ray_capacity);
    next.Resize(ray_capacity);
    closest.Resize(ray_capacity);
    miss.Resize(ray_capacity);
    shadow.Resize(ray_capacity);
}

} // namespace cuwfrt
