#pragma once

#include "scene.cuh"

namespace cuwfrt
{

void CreateRectXY(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));
void CreateRectXZ(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));
void CreateRectYZ(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));
void CreateBox(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));
void CreateCornellBox(Scene& scene, const Transform& o);

} // namespace cuwfrt
