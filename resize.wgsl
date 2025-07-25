override workgroupxy_size = 8;

@group(0) @binding(0) var input_sampler: sampler;
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;


@compute @workgroup_size(workgroupxy_size, workgroupxy_size, 1)
fn resize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_pos = global_id.xy;
    let texture_size = textureDimensions(output_texture);
    if all(pixel_pos < texture_size) {
        let texel = textureSampleLevel(input_texture, input_sampler, (vec2f(pixel_pos) + vec2f(0.5)) / vec2f(texture_size), 0);
        textureStore(output_texture, pixel_pos, texel);
    }
}
