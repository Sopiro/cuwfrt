#pragma once

#include "indices.h"
#include "scene.h"

namespace cuwfrt
{

void CreateRectXY(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));
void CreateRectXZ(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));
void CreateRectYZ(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));
void CreateBox(Scene& scene, const Transform& transform, MaterialIndex material, const Point2& tc = Point2(1, 1));

} // namespace cuwfrt
