// ============================================================
// lib/sdf.glsl  —  OpenShader signed-distance functions
// Usage : #include "lib/sdf.glsl"
// Reference : Inigo Quilez / iquilezles.org (reimplemented)
// ============================================================
#ifndef LIB_SDF_GLSL
#define LIB_SDF_GLSL

// ── 2D primitives (return signed distance, <0 inside) ────────

float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdEquilateralTriangle(vec2 p) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k*p.y, -k*p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0, 0.0);
    return -length(p) * sign(p.y);
}

float sdRing(vec2 p, float r, float thickness) {
    return abs(length(p) - r) - thickness;
}

float sdPie(vec2 p, vec2 sc, float r) {
    p.x = abs(p.x);
    float l = length(p) - r;
    float m = length(p - sc * clamp(dot(p, sc), 0.0, r));
    return max(l, m * sign(sc.y * p.x - sc.x * p.y));
}

// ── 3D primitives ─────────────────────────────────────────────

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0) - r;
}

float sdTorus(vec3 p, vec2 t) {
    return length(vec2(length(p.xz) - t.x, p.y)) - t.y;
}

float sdCylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

float sdCone(vec3 p, vec2 c, float h) {
    vec2 q = h * vec2(c.x / c.y, -1.0);
    vec2 w = vec2(length(p.xz), p.y);
    vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    vec2 b2 = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b2, b2));
    float s = max(k * (w.x*q.y - w.y*q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    float m = p.x + p.y + p.z - s;
    vec3 r;
    if (3.0*p.x < m) r = p.xyz;
    else if (3.0*p.y < m) r = p.yzx;
    else if (3.0*p.z < m) r = p.zxy;
    else return m * 0.57735027;
    float k = clamp(0.5*(r.z - r.y + s), 0.0, s);
    return length(vec3(r.x, r.y - s + k, r.z - k));
}

// ── Boolean operators ─────────────────────────────────────────

float sdUnion(float a, float b)        { return min(a, b); }
float sdIntersect(float a, float b)    { return max(a, b); }
float sdSubtract(float base, float sub){ return max(base, -sub); }

// Smooth union (k controls blend radius, k~0.1)
float sdSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

float sdSmoothIntersect(float a, float b, float k) {
    float h = clamp(0.5 - 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) + k*h*(1.0-h);
}

float sdSmoothSubtract(float base, float sub, float k) {
    float h = clamp(0.5 - 0.5*(base+sub)/k, 0.0, 1.0);
    return mix(base, -sub, h) + k*h*(1.0-h);
}

// ── Domain operations ─────────────────────────────────────────

// Infinite repetition (period c)
vec3 opRepeat(vec3 p, vec3 c) {
    return mod(p + 0.5*c, c) - 0.5*c;
}

// Limited repetition  (n = number of copies in each axis, 0-indexed)
vec3 opRepeatLimited(vec3 p, float c, vec3 n) {
    return p - c * clamp(round(p/c), -n, n);
}

// Twist around Y axis
vec3 opTwist(vec3 p, float k) {
    float c = cos(k * p.y);
    float s = sin(k * p.y);
    return vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z);
}

// Bend along XZ plane
vec3 opBend(vec3 p, float k) {
    float c = cos(k * p.x);
    float s = sin(k * p.x);
    return vec3(c*p.x - s*p.y, s*p.x + c*p.y, p.z);
}

// ── Normals (central differences) ────────────────────────────
// Usage:  vec3 n = sdfNormal(p, 1e-3, myScene);
// Requires GLSL 4.0+ for function pointers workaround — use
// the macro version below for GLSL 3.3 compatibility.
//
// Macro: NORMAL(p, eps, expr)  where expr uses variable q
// Example:
//   #define SCENE(q) sdSphere(q, 1.0)
//   vec3 n = SDF_NORMAL(p, 1e-3, SCENE);
//
#define SDF_NORMAL(pos, eps, fn) normalize(vec3( \
    fn(pos + vec3(eps,0,0)) - fn(pos - vec3(eps,0,0)), \
    fn(pos + vec3(0,eps,0)) - fn(pos - vec3(0,eps,0)), \
    fn(pos + vec3(0,0,eps)) - fn(pos - vec3(0,0,eps))  \
))

// ── Raymarching helper ────────────────────────────────────────
// Returns hit distance (>MAX_DIST = miss).  scene() must be defined.
// Usage:
//   #define scene(p)  sdSphere(p, 1.0)
//   float t = rmMarch(ro, rd, 0.001, 100.0, 100);
//
#define RM_MAX_STEPS 128
float rmMarch(vec3 ro, vec3 rd, float tmin, float tmax, float precis) {
    float t = tmin;
    for (int i = 0; i < RM_MAX_STEPS; i++) {
        float d = scene(ro + rd*t);
        if (d < precis || t > tmax) break;
        t += d;
    }
    return t;
}

#endif // LIB_SDF_GLSL
