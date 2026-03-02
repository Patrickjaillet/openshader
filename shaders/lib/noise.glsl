// ============================================================
// lib/noise.glsl  —  OpenShader noise library
// Usage : #include "lib/noise.glsl"
// ============================================================
#ifndef LIB_NOISE_GLSL
#define LIB_NOISE_GLSL

// ── Hash helpers ─────────────────────────────────────────────
float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

// ── Value noise ───────────────────────────────────────────────
// Returns [0, 1]
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);   // smoothstep
    return mix(
        mix(hash12(i),           hash12(i + vec2(1,0)), u.x),
        mix(hash12(i + vec2(0,1)), hash12(i + vec2(1,1)), u.x),
        u.y
    );
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash11(dot(i,           vec3(1,57,113))),
                hash11(dot(i+vec3(1,0,0),vec3(1,57,113))), u.x),
            mix(hash11(dot(i+vec3(0,1,0),vec3(1,57,113))),
                hash11(dot(i+vec3(1,1,0),vec3(1,57,113))), u.x), u.y),
        mix(mix(hash11(dot(i+vec3(0,0,1),vec3(1,57,113))),
                hash11(dot(i+vec3(1,0,1),vec3(1,57,113))), u.x),
            mix(hash11(dot(i+vec3(0,1,1),vec3(1,57,113))),
                hash11(dot(i+vec3(1,1,1),vec3(1,57,113))), u.x), u.y)
    );
}

// ── Gradient noise (Perlin-style) ─────────────────────────────
// Returns [-1, 1]
float gnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);  // quintic

    vec2 ga = hash22(i)           * 2.0 - 1.0;
    vec2 gb = hash22(i+vec2(1,0)) * 2.0 - 1.0;
    vec2 gc = hash22(i+vec2(0,1)) * 2.0 - 1.0;
    vec2 gd = hash22(i+vec2(1,1)) * 2.0 - 1.0;

    float va = dot(ga, f);
    float vb = dot(gb, f - vec2(1,0));
    float vc = dot(gc, f - vec2(0,1));
    float vd = dot(gd, f - vec2(1,1));

    return mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
}

// ── Fractal Brownian Motion ───────────────────────────────────
// octaves : number of layers (4–8 recommended)
// Returns approx. [0, 1]
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float sum = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        value     += amplitude * (vnoise(p * frequency) * 2.0 - 1.0);
        sum       += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
        p         += vec2(1.7, 9.2);   // domain shift to break alignment
    }
    return value / sum * 0.5 + 0.5;
}

// Convenience overload — 6 octaves
float fbm(vec2 p) { return fbm(p, 6); }

float fbm(vec3 p, int octaves) {
    float value = 0.0, amplitude = 0.5, frequency = 1.0, sum = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        value     += amplitude * (vnoise(p * frequency) * 2.0 - 1.0);
        sum       += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
        p         += vec3(1.7, 9.2, 3.4);
    }
    return value / sum * 0.5 + 0.5;
}

float fbm(vec3 p) { return fbm(p, 6); }

// ── Voronoi / Worley noise ────────────────────────────────────
// Returns vec2(dist_to_nearest_cell_center, cell_id_hash)
vec2 voronoi(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float min_dist = 8.0;
    float min_id   = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2  neighbor = vec2(float(x), float(y));
            vec2  point    = hash22(i + neighbor);
            vec2  diff     = neighbor + point - f;
            float d        = length(diff);
            if (d < min_dist) {
                min_dist = d;
                min_id   = hash12(i + neighbor);
            }
        }
    }
    return vec2(min_dist, min_id);
}

// ── Simplex noise 2D ──────────────────────────────────────────
// Returns [-1, 1]   (Gustavson-style, no mod289 permutation)
float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i  = mod(i, 289.0);
    vec3 p = vec3(
        mod((34.0 * vec3(i.x, i.x+i1.x, i.x+1.0) + 1.0)
            * vec3(i.x, i.x+i1.x, i.x+1.0), 289.0)
        + vec3(i.y, i.y+i1.y, i.y+1.0));
    p = mod((34.0*p + 1.0)*p, 289.0);
    p = p * (1.0/41.0) - 0.5;
    // could normalize properly but this is compact & good enough
    vec3 m = max(0.5 - vec3(dot(x0,x0),
                             dot(x12.xy, x12.xy),
                             dot(x12.zw, x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x  = 2.0*fract(p*C.www) - 1.0;
    vec3 h  = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314*(a0*a0 + h*h);
    vec3 g;
    g.x  = a0.x  * x0.x    + h.x  * x0.y;
    g.yz = a0.yz * x12.xz  + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

#endif // LIB_NOISE_GLSL
