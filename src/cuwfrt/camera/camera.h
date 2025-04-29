#pragma once

#include "filters.h"

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"
#include "cuwfrt/shading/sampling.h"

namespace cuwfrt
{

// Pinhole camera model
class Camera
{
public:
    Camera() = default;
    Camera(
        const Point3& position,
        const Point3& forward,
        const Vec3& up,
        Float vfov, // vertical field-of-view. in degrees.
        Float aperture,
        Float focus_dist,
        const Point2i& resolution,
        Float sigma
    )
        : origin{ position }
        , focus_dist{ focus_dist }
        , resolution{ resolution }
        , sigma{ sigma }
    {
        Float theta = DegToRad(vfov);
        Float h = std::tan(theta / 2);
        Float aspect_ratio = (Float)resolution.x / (Float)resolution.y;
        Float viewport_height = 2 * h;
        Float viewport_width = aspect_ratio * viewport_height;

        w = -Normalize(forward);
        u = Normalize(Cross(up, w));
        v = Cross(w, u);

        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    __GPU__ Float SampleRay(Ray* ray, const Point2i& pixel, const Point2& u0, const Point2& u1) const
    {
        Point2 pixel_offset = SampleGaussianFilter(sigma, u0) + Point2(Float(0.5), Float(0.5));
        Point3 pixel_center = lower_left + horizontal * (pixel.x + pixel_offset.x) / resolution.x +
                              vertical * (pixel.y + pixel_offset.y) / resolution.y;

        Point3 aperture_sample = lens_radius * SampleUniformUnitDiskXY(u1);
        Point3 camera_offset = u * aperture_sample.x + v * aperture_sample.y;
        Point3 camera_center = origin + camera_offset;

        ray->o = camera_center;
        ray->d = Normalize(pixel_center - camera_center);

        return 1;
    }

    __GPU__ Point2i GetRasterPos(const Point3& world_pos)
    {
        Vec3 o2p = Normalize(world_pos - origin);

        Float l = focus_dist / Dot(o2p, -w);
        Point3 p_focus = origin + o2p * l;

        Vec3 ll2p = p_focus - lower_left;

        Float w2 = Length2(horizontal);
        Float h2 = Length2(vertical);

        int32 px = int32(resolution.x * Dot(horizontal, ll2p) / w2);
        int32 py = int32(resolution.y * Dot(vertical, ll2p) / h2);

        return { px, py };
    }

    bool operator==(const Camera&) const = default;
    bool operator!=(const Camera&) const = default;

private:
    Float focus_dist;
    Float sigma;
    Point2i resolution;

    Point3 origin;
    Point3 lower_left;
    Vec3 horizontal, vertical;

    Float lens_radius;

    // Local coordinate frame
    Vec3 u, v, w;
};

} // namespace cuwfrt
