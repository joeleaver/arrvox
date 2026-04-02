// Grass shading model — PBR with subsurface translucency for grass blades.
//
// Delegates to shade_pbr for the heavy lifting (lights, shadows, GI, AO).
// The grass-specific visual qualities come from the material properties:
// albedo (green), subsurface scattering (backlit translucency), roughness.
// The normal comes from the G-buffer (computed by the march using the
// gradient of the procedural opacity field).

fn shade_grass(ctx: ShadingContext) -> vec3<f32> {
    return shade_pbr(ctx);
}
