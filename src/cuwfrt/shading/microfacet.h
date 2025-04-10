#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"

#include "cuwfrt/shading/frame.h"
#include "cuwfrt/shading/sampling.h"
#include "cuwfrt/shading/scattering.h"

#define mf_default_reflectance Vec3(0.04f)
#define mf_min_alpha (0.003f)

namespace cuwfrt
{

namespace mf
{

inline __GPU__ Float RoughnessToAlpha(Float roughness)
{
    return std::fmax(roughness * roughness, mf_min_alpha);
}

inline __GPU__ Vec3 F0(Vec3 basecolor, Float metallic)
{
    return Lerp(mf_default_reflectance, basecolor, metallic);
}

inline __GPU__ Vec3 F_Schlick(Vec3 f0, Float cosine_theta)
{
    return f0 + (/*f90*/ Vec3(1) - f0) * std::pow(1 - cosine_theta, 5.0f);
}

inline __GPU__ Float Lambda(const Vec3& w, Float alpha_x, Float alpha_y)
{
    Float tan2_theta = Tan2Theta(w);
    if (isinf(tan2_theta))
    {
        return 0;
    }

    Float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);
    return (std::sqrt(1 + alpha2 * tan2_theta) - 1) / 2;
}

inline __GPU__ Float G1(const Vec3& w, Float alpha_x, Float alpha_y)
{
    return 1 / (1 + Lambda(w, alpha_x, alpha_y));
}

inline __GPU__ Float G(const Vec3& wo, const Vec3& wi, Float alpha_x, Float alpha_y)
{
    return 1 / (1 + Lambda(wo, alpha_x, alpha_y) + Lambda(wi, alpha_x, alpha_y));
}

inline __GPU__ Float D(const Vec3& wm, Float alpha_x, Float alpha_y)
{
    Float tan2_theta = Tan2Theta(wm);
    if (isinf(tan2_theta))
    {
        return 0;
    }

    Float cos4_theta = Sqr(Cos2Theta(wm));

    if (cos4_theta < 1e-16f)
    {
        return 0;
    }
    Float e = tan2_theta * (Sqr(CosPhi(wm) / alpha_x) + Sqr(SinPhi(wm) / alpha_y));

    return 1 / (pi * alpha_x * alpha_y * cos4_theta * Sqr(1 + e));
}

inline __GPU__ Float D(const Vec3& w, const Vec3& wm, Float alpha_x, Float alpha_y)
{
    return G1(w, alpha_x, alpha_y) / AbsCosTheta(w) * D(wm, alpha_x, alpha_y) * AbsDot(w, wm);
}

inline __GPU__ Float PDF(const Vec3& w, const Vec3& wm, Float alpha_x, Float alpha_y)
{
    return D(w, wm, alpha_x, alpha_y);
}

inline __GPU__ Vec3 Sample_Wm(const Vec3& w, Float alpha_x, Float alpha_y, Point2 u12)
{
    Vec3 wm = Sample_GGX_VNDF_Dupuy_Benyoub(w, alpha_x, alpha_y, u12);
    return wm;
}

} // namespace mf

} // namespace cuwfrt
