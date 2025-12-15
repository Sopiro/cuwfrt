__GPU__ inline Vec3 DielectricMaterial::Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return Vec3(0);
}

__GPU__ inline bool DielectricMaterial::SampleBSDF(
    Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
) const
{
    Frame f(isect.shading_normal);
    Vec3 wo_local = f.ToLocal(wo);

    // Sample perfect specular dielectric BSDF
    Float R = FresnelSchlick(CosTheta(wo_local), eta);
    Float T = 1 - R;

    // Compute sampling probabilities for reflection and transmission
    Float pr = R;
    Float pt = T;

    ss->s = Vec3(__half2float(r[0]), __half2float(r[1]), __half2float(r[2]));

    if (u0 < pr / (pr + pt))
    {
        // Sample perfect specular dielectric BRDF
        Vec3 wi(-wo_local.x, -wo_local.y, wo_local.z);

        Vec3 fr(R / AbsCosTheta(wi));
        ss->s *= fr;
        ss->is_specular = true;
        ss->wi = f.FromLocal(wi);
        ss->pdf = pr / (pr + pt);
    }
    else
    {
        // Sample perfect specular dielectric BTDF
        // Compute ray direction for specular transmission
        Vec3 wi;
        Float eta_p;
        if (!Refract(&wi, wo_local, z_axis, eta, &eta_p))
        {
            return false;
        }

        Vec3 ft(T / AbsCosTheta(wi));

        ss->s *= ft;
        ss->is_specular = true;
        ss->wi = f.FromLocal(wi);
        ss->pdf = pt / (pr + pt);
    }

    return true;
}

__GPU__ inline Float DielectricMaterial::PDF(
    const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi
) const
{
    return 0;
}

__GPU__ inline Vec3 DielectricMaterial::BSDF(
    const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi
) const
{
    return Vec3(0);
}

__GPU__ inline Vec4 DielectricMaterial::Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return { rho(this, scene, isect, wo), 1 };
}