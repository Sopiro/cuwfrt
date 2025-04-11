#pragma once

#include "cuwfrt/material/materials.cuh"

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
    return R0 + (1 - R0) * std::powf(1 - cos_theta_i, 5);
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

inline __GPU__ Vec3 rho(const Material* mat, const GPUScene* scene, const Intersection& isect, const Vec3& wo)
{
    // Precomputed Halton samples from pbrt4
    constexpr int rho_samples = 16;
    const Float uc_rho[rho_samples] = { 0.75741637f, 0.37870818f, 0.7083487f, 0.18935409f, 0.9149363f, 0.35417435f,
                                        0.5990858f,  0.09467703f, 0.8578725f, 0.45746812f, 0.686759f,  0.17708716f,
                                        0.9674518f,  0.2995429f,  0.5083201f, 0.047338516f };
    const Point2 u_rho[rho_samples] = { Point2(0.855985f, 0.570367f), Point2(0.381823f, 0.851844f), Point2(0.285328f, 0.764262f),
                                        Point2(0.733380f, 0.114073f), Point2(0.542663f, 0.344465f), Point2(0.127274f, 0.414848f),
                                        Point2(0.964700f, 0.947162f), Point2(0.594089f, 0.643463f), Point2(0.095109f, 0.170369f),
                                        Point2(0.825444f, 0.263359f), Point2(0.429467f, 0.454469f), Point2(0.244460f, 0.816459f),
                                        Point2(0.756135f, 0.731258f), Point2(0.516165f, 0.152852f), Point2(0.180888f, 0.214174f),
                                        Point2(0.898579f, 0.503897f) };

    Vec3 r(0);
    for (size_t i = 0; i < rho_samples; ++i)
    {
        Scattering ss;
        if (mat->SampleBSDF(&ss, scene, isect, wo, uc_rho[i], u_rho[i]))
        {
            r += ss.s * AbsCosTheta(ss.wi) / ss.pdf;
        }
    }
    return r / rho_samples;
}

} // namespace cuwfrt
