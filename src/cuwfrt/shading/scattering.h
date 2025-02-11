#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"

namespace cuwfrt
{

inline __GPU__ Float FresnelSchlick(Float cos_theta_i, Float eta)
{
    cos_theta_i = Clamp(cos_theta_i, -1, 1);

    // Potentially flip interface orientation for Fresnel equations
    if (cos_theta_i < 0)
    {
        eta = 1 / eta;
        cos_theta_i = -cos_theta_i;
    }

    // Compute cos_theta_t for Fresnel equations using Snell's law
    Float sin2_theta_i = 1 - Sqr(cos_theta_i);
    Float sin2_theta_t = sin2_theta_i / Sqr(eta);

    if (sin2_theta_t >= 1)
    {
        // Total internal reflection
        return 1;
    }

    // Compute the reflectance at normal incidence (R0)
    Float R0 = Sqr((1 - eta) / (1 + eta));

    // Schlick's approximation for Fresnel reflectance
    return R0 + (1 - R0) * std::pow(1 - cos_theta_i, 5);
}

inline __GPU__ Float FresnelDielectric(Float cos_theta_i, Float eta)
{
    cos_theta_i = Clamp(cos_theta_i, -1, 1);

    // Potentially flip interface orientation for Fresnel equations
    if (cos_theta_i < 0)
    {
        eta = 1 / eta;
        cos_theta_i = -cos_theta_i;
    }

    // Compute cos_theta_t for Fresnel equations using Snell's law
    Float sin2_theta_i = 1 - Sqr(cos_theta_i);
    Float sin2_theta_t = sin2_theta_i / Sqr(eta);
    if (sin2_theta_t >= 1)
    {
        // Total internal reflection
        return 1;
    }

    Float cos_theta_t = std::sqrt(std::max<Float>(0, 1 - sin2_theta_t));

    Float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    Float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}

inline __GPU__ bool Refract(Vec3* wt, Vec3 wi, Vec3 n, Float eta, Float* eta_p)
{
    Float cos_theta_i = Dot(n, wi);

    // Potentially flip interface orientation
    if (cos_theta_i < 0)
    {
        eta = 1 / eta;
        cos_theta_i = -cos_theta_i;
        n = -n;
    }

    Float sin2_theta_i = std::max<Float>(0, 1 - Sqr(cos_theta_i));
    Float sin2_theta_t = sin2_theta_i / Sqr(eta);

    // Case of total internal reflection
    if (sin2_theta_t >= 1)
    {
        return false;
    }

    Float cos_theta_t = std::sqrt(1 - sin2_theta_t);

    Float inv_eta = 1 / eta;
    *wt = -wi * inv_eta + (cos_theta_i * inv_eta - cos_theta_t) * n;

    if (eta_p)
    {
        *eta_p = eta;
    }

    return true;
}

} // namespace cuwfrt
