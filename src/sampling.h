#pragma once

#include "common.h"

namespace cuwfrt
{

// Heuristic functions for MIS
inline __CPU_GPU__ Float BalanceHeuristic(Float pdf_f, Float pdf_g)
{
    return pdf_f / (pdf_f + pdf_g);
}

inline __CPU_GPU__ Float BalanceHeuristic(int32 nf, Float pdf_f, int32 ng, Float pdf_g)
{
    return (nf * pdf_f) / (nf * pdf_f + ng * pdf_g);
}

inline __CPU_GPU__ Float PowerHeuristic(Float pdf_f, Float pdf_g)
{
    Float f2 = pdf_f * pdf_f;
    Float g2 = pdf_g * pdf_g;

    return f2 / (f2 + g2);
}

inline __CPU_GPU__ Float PowerHeuristic(int32 nf, Float pdf_f, int32 ng, Float pdf_g)
{
    Float f = nf * pdf_f;
    Float g = ng * pdf_g;

    return (f * f) / (f * f + g * g);
}

inline __CPU_GPU__ Vec3 SampleUniformHemisphere(const Point2& u)
{
    Float z = u[0];
    Float r = std::sqrt(std::fmax(0.0f, 1 - z * z));
    Float phi = two_pi * u[1];

    return Vec3(r * std::cos(phi), r * std::sin(phi), z);
}

inline __CPU_GPU__ Float UniformHemispherePDF()
{
    return inv_two_pi;
}

inline __CPU_GPU__ Vec3 SampleUniformSphere(const Point2& u)
{
    Float z = 1 - 2 * u[0];
    Float r = std::sqrt(std::fmax(0.0f, 1 - z * z));
    Float phi = two_pi * u[1];

    Float x = r * std::cos(phi);
    Float y = r * std::sin(phi);

    return Vec3(x, y, z);
}

inline __CPU_GPU__ Float UniformSpherePDF()
{
    return inv_four_pi;
}

// z > 0
inline __CPU_GPU__ Vec3 SampleCosineHemisphere(const Point2& u)
{
    Float z = std::sqrt(1 - u[1]);

    Float phi = two_pi * u[0];
    Float su2 = std::sqrt(u[1]);
    Float x = std::cos(phi) * su2;
    Float y = std::sin(phi) * su2;

    return Vec3(x, y, z);
}

inline __CPU_GPU__ Float CosineHemispherePDF(Float cos_theta)
{
    return cos_theta * inv_pi;
}

inline __CPU_GPU__ Vec3 SampleInsideUnitSphere(const Point2& u)
{
#if 1
    Float theta = two_pi * u[0];
    Float phi = std::acos(2 * u[1] - 1);

    Float sin_phi = std::sin(phi);

    Float x = sin_phi * std::cos(theta);
    Float y = sin_phi * std::sin(theta);
    Float z = std::cos(phi);

    return Vec3(x, y, z);
#else
    // Rejection sampling
    Vec3 p;
    do
    {
        p = RandVec3(-1.0f, 1.0f);
    } while (Length2(p) >= 1);

    return p;
#endif
}

inline __CPU_GPU__ Vec3 SampleUniformUnitDiskXY(const Point2& u)
{
    Float r = std::sqrt(u.x);
    Float theta = two_pi * u.y;
    return Vec3(r * std::cos(theta), r * std::sin(theta), 0);
}

inline __CPU_GPU__ Float SampleExponential(Float u, Float a)
{
    return -std::log(1 - u) / a;
}

inline __CPU_GPU__ Float ExponentialPDF(Float x, Float a)
{
    return a * std::exp(-a * x);
}

inline __CPU_GPU__ Vec3 Sample_GGX(Vec3 wo, Float alpha2, Point2 u)
{
    WakNotUsed(wo);

    Float theta = std::acos(std::sqrt((1 - u.x) / ((alpha2 - 1) * u.x + 1)));
    Float phi = two_pi * u.y;

    Float sin_thetha = std::sin(theta);
    Float x = std::cos(phi) * sin_thetha;
    Float y = std::sin(phi) * sin_thetha;
    Float z = std::cos(theta);

    Vec3 h{ x, y, z }; // Sampled half vector

    return h;
}

// "Sampling Visible GGX Normals with Spherical Caps" by Dupuy & Benyoub
// https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
// https://cdrdv2-public.intel.com/782052/sampling-visible-ggx-normals.pdf
inline __CPU_GPU__ Vec3 SampleVNDFHemisphere(Vec3 wo, Point2 u)
{
    // sample a spherical cap in (-wo.z, 1]
    Float phi = two_pi * u.x;
    Float z = std::fma((1 - u.y), (1 + wo.z), -wo.z);
    Float sin_theta = std::sqrt(Clamp(1 - z * z, 0, 1));
    Float x = sin_theta * std::cos(phi);
    Float y = sin_theta * std::sin(phi);
    Vec3 c = Vec3(x, y, z);
    // compute halfway direction;
    Vec3 h = c + wo;

    // return without normalization as this is done later (see line 25)
    return h;
}

inline __CPU_GPU__ Vec3 Sample_GGX_VNDF_Dupuy_Benyoub(Vec3 wo, Float alpha_x, Float alpha_y, Point2 u)
{
    // warp to the hemisphere configuration
    Vec3 woStd = Normalize(Vec3(wo.x * alpha_x, wo.y * alpha_y, wo.z));
    // sample the hemisphere
    Vec3 wmStd = SampleVNDFHemisphere(woStd, u);
    // warp back to the ellipsoid configuration
    Vec3 wm = Normalize(Vec3(wmStd.x * alpha_x, wmStd.y * alpha_y, wmStd.z));
    // return final normal
    return wm;
}

// Source: "Sampling the GGX Distribution of Visible Normals" by Heitz
// https://jcgt.org/published/0007/04/01/
inline __CPU_GPU__ Vec3 Sample_GGX_VNDF_Heitz(Vec3 wo, Float alpha_x, Float alpha_y, Point2 u)
{
    // Section 3.2: transforming the view direction to the hemisphere configuration
    Vec3 Vh{ alpha_x * wo.x, alpha_y * wo.y, wo.z };
    Vh.Normalize();

    // Build an orthonormal basis with v, t1, and t2
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    Vec3 T1 = (Vh.z < 0.999f) ? Normalize(Cross(Vh, z_axis)) : x_axis;
    Vec3 T2 = Cross(T1, Vh);

    // Section 4.2: parameterization of the projected area
    Float r = std::sqrt(u.x);
    Float phi = two_pi * u.y;
    Float t1 = r * std::cos(phi);
    Float t2 = r * std::sin(phi);
    Float s = 0.5f * (1 + Vh.z);
    t2 = Lerp(std::sqrt(1 - t1 * t1), t2, s);

    // Section 4.3: reprojection onto hemisphere
    Vec3 Nh = t1 * T1 + t2 * T2 + std::sqrt(std::fmax(0.0f, 1 - t1 * t1 - t2 * t2)) * Vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    Vec3 h = Normalize(Vec3(alpha_x * Nh.x, alpha_y * Nh.y, std::fmax(0.0f, Nh.z))); // Sampled half vector

    return h;
}

} // namespace cuwfrt
