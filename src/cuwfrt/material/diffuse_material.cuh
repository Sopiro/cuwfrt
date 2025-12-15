__GPU__ inline Vec3 DiffuseMaterial::Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return Vec3(0);
}

__GPU__ inline bool DiffuseMaterial::SampleBSDF(
    Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
) const
{
    Frame f(isect.shading_normal);
    Vec3 wi = SampleCosineHemisphere(u12);
    ss->pdf = CosineHemispherePDF(wi.z);
    ss->wi = f.FromLocal(wi);
    ss->is_specular = false;
    ss->s = Lambertian(scene, isect);

    return true;
}

__GPU__ inline Float DiffuseMaterial::PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    Frame f(isect.shading_normal);
    Vec3 wi_local = f.ToLocal(wi);
    Vec3 wo_local = f.ToLocal(wo);
    if (!SameHemisphere(wi_local, wo_local))
    {
        return 0;
    }

    return CosineHemispherePDF(AbsCosTheta(wi_local));
}

__GPU__ inline Vec3 DiffuseMaterial::BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    Frame f(isect.shading_normal);
    Vec3 wi_local = f.ToLocal(wi);
    Vec3 wo_local = f.ToLocal(wo);
    if (!SameHemisphere(wi_local, wo_local))
    {
        return Vec3(0);
    }

    return Lambertian(scene, isect);
}

__GPU__ inline Vec4 DiffuseMaterial::Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    if (r.x < 0)
    {
        Point2 uv = triangle::GetTexcoord(scene, isect);
        Vec3 albedo = SampleTexture(scene, TextureIndex(r.z), uv);
        return { albedo + Vec3(1e-2f), 1 };
    }
    else
    {
        return { r + Vec3(1e-2f), 1 };
    }
}

__GPU__ inline Vec3 DiffuseMaterial::Lambertian(const GPUScene* scene, const Intersection& isect) const
{
    if (r.x < 0)
    {
        Point2 uv = triangle::GetTexcoord(scene, isect);
        Vec3 tex = SampleTexture(scene, TextureIndex(r.z), uv);
        return tex * inv_pi;
    }
    else
    {
        return r * inv_pi;
    }
}