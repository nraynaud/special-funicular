override workgroup_size = 64;
override workgroupxy_size = 8;

struct Params {
    from_mip: u32,
    extrema_threshold: f32,
    extrema_border: i32
};

@group(0) @binding(0) var<uniform> parameters: Params;
@group(0) @binding(1) var diff_stack: texture_2d_array<i32>;
// 4D position (x, y, scale, mip) of found extrema
@group(0) @binding(10) var<storage, read_write> extrema_storage: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> extrema_count: atomic<u32>;

const MAX_2_31 = pow(2, 31) - 1.0;
const X = vec3i(1, 0, 0);
const Y = vec3i(0, 1, 0);
const Z = vec3i(0, 0, 1);

fn get_pix(mip: i32, pos: vec3i) -> f32 {
    // TODO: normalization is sketchy, MAX_2_31 doesn't fit in an f32
    return f32(textureLoad(diff_stack, pos.xy, pos.z, mip).r) / MAX_2_31;
}

fn d_axis(mip: i32, pos: vec3i, dir: vec3i) -> f32 {
    return 0.5 * (get_pix(mip, pos + dir) - get_pix(mip, pos - dir));
}

fn get_gradient(mip: i32, pos: vec3i) -> vec3f {
    let dx = d_axis(mip, pos, X);
    let dy = d_axis(mip, pos, Y);
    let dz = d_axis(mip, pos, Z);
    return vec3f(dx, dy, dz);
}

fn dd_axis(mip: i32, pos: vec3i, dir: vec3i, center_value: f32) -> f32 {
    return get_pix(mip, pos - dir) + get_pix(mip, pos + dir) + 2.0 * center_value;
}

fn dxdy_axis(mip: i32, pos: vec3i, dir1: vec3i, dir2: vec3i) -> f32 {
    var sum = get_pix(mip, pos + dir1 + dir2);
    sum += get_pix(mip, pos - dir1 - dir2);
    sum -= get_pix(mip, pos + dir1 - dir2);
    sum -= get_pix(mip, pos - dir1 + dir2);
    return sum / 4.0;
}

fn get_hessian(mip: i32, pos: vec3i) -> mat3x3f {
    // hessian is a symmetric matrix, no stress on col/row issues
    let center_value = get_pix(mip, pos);
    let dxx = dd_axis(mip, pos, X, center_value);
    let dyy = dd_axis(mip, pos, Y, center_value);
    let dzz = dd_axis(mip, pos, Z, center_value);
    let dxy = dxdy_axis(mip, pos, X, Y);
    let dxz = dxdy_axis(mip, pos, X, Z);
    let dyz = dxdy_axis(mip, pos, Y, Z);
    let xcol = vec3f(dxx, dxy, dxz);
    let ycol = vec3f(dxy, dyy, dyz);
    let zcol = vec3f(dxz, dyz, dzz);
    return mat3x3f(xcol, ycol, zcol);
}

fn compute_least_squares(hessian: mat3x3f, gradient: vec3f) -> vec4f {
    //generated with AI
    // return the pos correction in the first 3 values and last position is the value correction.
    // Solve Hx = -g for x, where H is the Hessian and g is the gradient
    let det = determinant(hessian);
    // Check if matrix is invertible
    if (abs(det) < 1e-10) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    // Each row of the cofactor matrix is the cross product of two columns
    let cof_row0 = cross(hessian[1], hessian[2]);
    let cof_row1 = cross(hessian[2], hessian[0]);
    let cof_row2 = cross(hessian[0], hessian[1]);
    let cofactor_transpose = mat3x3f(cof_row0, cof_row1, cof_row2);
    let inv_hessian = cofactor_transpose * (1.0 / det);
    // Solve for x = -H^(-1) * g
    let correction = -(inv_hessian * gradient);
    return vec4f(correction, 0.5 * dot(gradient, correction));
}

@compute @workgroup_size(workgroup_size, 1, 1)
// one invocation per extremum
fn extrema_refine(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_size = vec3f(vec2f(textureDimensions(diff_stack)), f32(textureNumLayers(diff_stack)));
    var extremum = extrema_storage[global_id.x];
    var allowed_refinements = 5;
    var discard_extremum = false;
    while allowed_refinements > 0 {
        allowed_refinements--;
        var pos = vec3i(round(extremum.xyz));
        let mip = i32(round(extremum.w));
        var val = get_pix(mip, pos);
        let grad = get_gradient(mip, pos);
        let hessian = get_hessian(mip, pos);
        let update = compute_least_squares(hessian, grad);
        extremum = vec4f(extremum.xyz + update.xyz, extremum.w);
        val += update.w;
        if all(abs(update.xyz) < vec3f(0.6)) {
            break;
        }
    }
    if any(extremum < vec4f(0.0)) || any(extremum.xyz >= texture_size) {
        discard_extremum = true;
    }
    if discard_extremum {
        extrema_storage[global_id.x] = vec4f(0.0);
    } else {
        extrema_storage[global_id.x] = extremum;
    }
}
