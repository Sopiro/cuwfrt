__GPU__ inline Vec3 DiffuseLightMaterial::Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    if (isect.front_face)
    {
        return emission;
    }
    else
    {
        return Vec3(0);
    }
}

__GPU__ inline bool DiffuseLightMaterial::SampleBSDF(
    Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
) const
{
    return false;
}

__GPU__ inline Float DiffuseLightMaterial::PDF(
    const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi
) const
{
    return 0;
}

__GPU__ inline Vec3 DiffuseLightMaterial::BSDF(
    const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi
) const
{
    return Vec3(0);
}

__GPU__ inline Vec4 DiffuseLightMaterial::Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return Vec4(0);
}