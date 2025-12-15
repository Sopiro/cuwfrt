__GPU__ inline Vec3 MirrorMaterial::Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return Vec3(0);
}

__GPU__ inline bool MirrorMaterial::SampleBSDF(
    Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
) const
{
    ss->s = reflectance;
    ss->wi = Reflect(wo, isect.shading_normal);
    ss->pdf = 1;
    ss->is_specular = true;

    return true;
}

__GPU__ inline Float MirrorMaterial::PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    return 0;
}

__GPU__ inline Vec3 MirrorMaterial::BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    return Vec3(0);
}

__GPU__ inline Vec4 MirrorMaterial::Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return { reflectance, 1 };
}