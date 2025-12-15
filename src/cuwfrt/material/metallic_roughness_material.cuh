__GPU__ inline Vec3 MetallicRoughnessMaterial::Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    if (tex_emissive > 0)
    {
        Point2 uv = triangle::GetTexcoord(scene, isect);
        return SampleTexture(scene, tex_emissive, uv);
    }
    else
    {
        return Vec3(0);
    }
}

__GPU__ inline bool MetallicRoughnessMaterial::SampleBSDF(
    Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
) const
{
    Frame f(isect.shading_normal);

    Vec3 wo_local = f.ToLocal(wo);
    if (wo_local.z == 0)
    {
        return false;
    }

    Point2 uv = triangle::GetTexcoord(scene, isect);
    Vec3 basecolor = SampleTexture(scene, tex_basecolor, uv);
    Float metallic = SampleTexture(scene, tex_metallic, uv).z;
    Float roughness = SampleTexture(scene, tex_roughness, uv).y;
    Float alpha = mf::RoughnessToAlpha(roughness);

    constexpr Vec3 coefficient(0.2126f, 0.7152f, 0.0722f);

    Vec3 f0 = mf::F0(basecolor, metallic);
    Vec3 F = mf::F_Schlick(f0, Dot(wo, isect.shading_normal));
    Float diff_weight = (1 - metallic);
    Float spec_weight = Dot(coefficient, F);
    // Float spec_weight = std::fmax(F.x, std::fmax(F.y, F.z));
    Float t = Clamp(spec_weight / (diff_weight + spec_weight), 0.15f, 0.9f);

    Vec3 wm, wi;
    if (u0 < t)
    {
        // Sample glossy
        wm = mf::Sample_Wm(wo_local, alpha, alpha, u12);
        wi = Reflect(wo_local, wm);

        if (!SameHemisphere(wo_local, wi))
        {
            return false;
        }
    }
    else
    {
        // Sample diffuse
        wi = SampleCosineHemisphere(u12);
        wm = Normalize(wi + wo_local);
    }

    Float cos_theta_o = AbsCosTheta(wo_local);
    Float cos_theta_i = AbsCosTheta(wi);
    if (cos_theta_i == 0 || cos_theta_o == 0)
    {
        return false;
    }

    Vec3 f_s = F * mf::D(wm, alpha, alpha) * mf::G(wo_local, wi, alpha, alpha) / (4 * cos_theta_i * cos_theta_o);
    Vec3 f_d = (Vec3(1) - F) * (1 - metallic) * (basecolor * inv_pi);

    Float p_s = mf::D(wo_local, wm, alpha, alpha) / (4 * AbsDot(wo_local, wm));
    Float p_d = cos_theta_i * inv_pi;

    ss->is_specular = false;
    ss->s = f_s + f_d;
    ss->wi = f.FromLocal(wi);
    ss->pdf = t * p_s + (1 - t) * p_d;
    return true;
}

__GPU__ inline Float MetallicRoughnessMaterial::PDF(
    const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi
) const
{
    Frame f(isect.shading_normal);
    Vec3 wi_local = f.ToLocal(wi);
    Vec3 wo_local = f.ToLocal(wo);
    if (!SameHemisphere(wi_local, wo_local))
    {
        return 0;
    }

    Vec3 wm = wo_local + wi_local;
    if (Length2(wm) == 0)
    {
        return 0;
    }
    wm.Normalize();

    if (Dot(wm, Vec3(0, 0, 1)) < 0)
    {
        wm.Negate();
    }

    Point2 uv = triangle::GetTexcoord(scene, isect);
    Vec3 basecolor = SampleTexture(scene, tex_basecolor, uv);
    Float metallic = SampleTexture(scene, tex_metallic, uv).z;
    Float roughness = SampleTexture(scene, tex_roughness, uv).y;
    Float alpha = mf::RoughnessToAlpha(roughness);

    constexpr Vec3 coefficient(0.2126f, 0.7152f, 0.0722f);

    Vec3 f0 = mf::F0(basecolor, metallic);
    Vec3 F = mf::F_Schlick(f0, Dot(wo, isect.shading_normal));
    Float diff_weight = (1 - metallic);
    Float spec_weight = Dot(coefficient, F);
    // Float spec_weight = std::fmax(F.x, std::fmax(F.y, F.z));
    Float t = Clamp(spec_weight / (diff_weight + spec_weight), 0.15f, 0.9f);

    Float p_s = mf::PDF(wo_local, wm, alpha, alpha) / (4 * AbsDot(wo_local, wm));
    Float p_d = AbsCosTheta(wi_local) * inv_pi;

    return t * p_s + (1 - t) * p_d;
}

__GPU__ inline Vec3 MetallicRoughnessMaterial::BSDF(
    const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi
) const
{
    Frame f(isect.shading_normal);
    Vec3 wi_local = f.ToLocal(wi);
    Vec3 wo_local = f.ToLocal(wo);
    if (!SameHemisphere(wi_local, wo_local))
    {
        return Vec3(0);
    }

    Float cos_theta_o = AbsCosTheta(wo_local);
    Float cos_theta_i = AbsCosTheta(wi_local);
    if (cos_theta_i == 0 || cos_theta_o == 0)
    {
        return Vec3(0);
    }

    Vec3 wm = wo_local + wi_local;
    if (Length2(wm) == 0)
    {
        return Vec3(0);
    }
    wm.Normalize();

    Point2 uv = triangle::GetTexcoord(scene, isect);
    Vec3 color = SampleTexture(scene, tex_basecolor, uv);
    Float metallic = SampleTexture(scene, tex_metallic, uv).z;
    Float roughness = SampleTexture(scene, tex_roughness, uv).y;
    Float alpha = mf::RoughnessToAlpha(roughness);

    Vec3 f0 = mf::F0(color, metallic);
    Vec3 F = mf::F_Schlick(f0, Dot(wi_local, wm));

    Vec3 f_s = F * mf::D(wm, alpha, alpha) * mf::G(wo_local, wi_local, alpha, alpha) / (4 * cos_theta_i * cos_theta_o);
    Vec3 f_d = (Vec3(1) - F) * (1 - metallic) * (color * inv_pi);

    return f_d + f_s;
}

__GPU__ inline Vec4 MetallicRoughnessMaterial::Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return { rho(this, scene, isect, wo), 1 };
}