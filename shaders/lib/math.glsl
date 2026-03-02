// ============================================================
// lib/math.glsl  —  OpenShader math & color utilities
// Usage : #include "lib/math.glsl"
// ============================================================
#ifndef LIB_MATH_GLSL
#define LIB_MATH_GLSL

// ── Constants ─────────────────────────────────────────────────
#define PI      3.14159265358979323846
#define TAU     6.28318530717958647692
#define PHI     1.61803398874989484820   // golden ratio
#define SQRT2   1.41421356237309504880
#define SQRT3   1.73205080756887729352
#define E       2.71828182845904523536
#define DEG2RAD 0.01745329251994329577
#define RAD2DEG 57.2957795130823208768

// ── Scalar helpers ────────────────────────────────────────────

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2  saturate(vec2  x) { return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3  x) { return clamp(x, 0.0, 1.0); }

float remap(float v, float lo, float hi, float outLo, float outHi) {
    return outLo + (v - lo) / (hi - lo) * (outHi - outLo);
}

float remap01(float v, float lo, float hi) {
    return saturate((v - lo) / (hi - lo));
}

float map(float v, float lo, float hi) { return remap01(v, lo, hi); }

// ── Easing functions ──────────────────────────────────────────
// All take t in [0,1] and return [0,1]

float easeInQuad(float t)  { return t*t; }
float easeOutQuad(float t) { return t*(2.0 - t); }
float easeInOutQuad(float t) {
    return t < 0.5 ? 2.0*t*t : -1.0 + (4.0 - 2.0*t)*t;
}

float easeInCubic(float t)  { return t*t*t; }
float easeOutCubic(float t) { float u = 1.0-t; return 1.0 - u*u*u; }
float easeInOutCubic(float t) {
    return t < 0.5 ? 4.0*t*t*t : 1.0 - pow(-2.0*t + 2.0, 3.0)*0.5;
}

float easeInExpo(float t)  { return t == 0.0 ? 0.0 : pow(2.0, 10.0*t - 10.0); }
float easeOutExpo(float t) { return t == 1.0 ? 1.0 : 1.0 - pow(2.0, -10.0*t); }

float easeInElastic(float t) {
    return t == 0.0 ? 0.0 : t == 1.0 ? 1.0
         : -pow(2.0, 10.0*t - 10.0) * sin((t*10.0 - 10.75) * (2.0*PI/3.0));
}

float easeOutElastic(float t) {
    return t == 0.0 ? 0.0 : t == 1.0 ? 1.0
         : pow(2.0, -10.0*t) * sin((t*10.0 - 0.75) * (2.0*PI/3.0)) + 1.0;
}

float easeOutBounce(float t) {
    if (t < 1.0/2.75)      return 7.5625*t*t;
    else if (t < 2.0/2.75) { t -= 1.5/2.75;   return 7.5625*t*t + 0.75; }
    else if (t < 2.5/2.75) { t -= 2.25/2.75;  return 7.5625*t*t + 0.9375; }
    else                   { t -= 2.625/2.75;  return 7.5625*t*t + 0.984375; }
}

// ── Color space conversions ───────────────────────────────────

// RGB → HSV  (all in [0,1])
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0*d + e)), d / (q.x + e), q.x);
}

// HSV → RGB
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// sRGB ↔ Linear
vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linearToSrgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0/2.2)); }

// Linear → Reinhard tone mapping
vec3 reinhardToneMap(vec3 c) { return c / (1.0 + c); }

// Oklab (perceptual) — Björn Ottosson 2020
vec3 linearSrgb2Oklab(vec3 c) {
    float l = 0.4122214708*c.r + 0.5363325363*c.g + 0.0514459929*c.b;
    float m = 0.2119034982*c.r + 0.6806995451*c.g + 0.1073969566*c.b;
    float s = 0.0883024619*c.r + 0.2817188376*c.g + 0.6299787005*c.b;
    l = pow(l, 1.0/3.0); m = pow(m, 1.0/3.0); s = pow(s, 1.0/3.0);
    return vec3(0.2104542553*l+0.7936177850*m-0.0040720468*s,
                1.9779984951*l-2.4285922050*m+0.4505937099*s,
                0.0259040371*l+0.7827717662*m-0.8086757660*s);
}

vec3 oklab2LinearSrgb(vec3 c) {
    float l = c.x+0.3963377774*c.y+0.2158037573*c.z;
    float m = c.x-0.1055613458*c.y-0.0638541728*c.z;
    float s = c.x-0.0894841775*c.y-1.2914855480*c.z;
    l=l*l*l; m=m*m*m; s=s*s*s;
    return vec3(+4.0767416621*l-3.3077115913*m+0.2309699292*s,
                -1.2684380046*l+2.6097574011*m-0.3413193965*s,
                -0.0041960863*l-0.7034186147*m+1.7076147010*s);
}

// ── Palette (Inigo Quilez cosine palette) ─────────────────────
// palette(t, a, b, c, d) — t in [0,1]
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c*t + d));
}

// Preset palettes
vec3 paletteRainbow(float t)   { return palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.00,0.33,0.67)); }
vec3 paletteNeon(float t)      { return palette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.30,0.20,0.20)); }
vec3 paletteCold(float t)      { return palette(t, vec3(0.5,0.5,0.7), vec3(0.5,0.5,0.3), vec3(1.0), vec3(0.0,0.1,0.2)); }
vec3 paletteLava(float t)      { return palette(t, vec3(0.5,0.1,0.0), vec3(0.5,0.4,0.0), vec3(1.0,0.8,0.5), vec3(0.0,0.1,0.8)); }
vec3 paletteGrayscale(float t) { return vec3(t); }

// ── 2D transforms ─────────────────────────────────────────────

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec2 rotate2(vec2 p, float a) { return rot2(a) * p; }

vec2 scale2(vec2 p, float s)      { return p / s; }
vec2 translate2(vec2 p, vec2 off) { return p - off; }

// Polar coords
vec2 toPolar(vec2 p)    { return vec2(length(p), atan(p.y, p.x)); }
vec2 fromPolar(vec2 r)  { return r.x * vec2(cos(r.y), sin(r.y)); }

// Mirror / tile
vec2 mirrorRepeat(vec2 p, vec2 period) {
    vec2 t = mod(p / period, 2.0);
    return period * abs(t - floor(t + 0.5));
}

// ── Utilities ─────────────────────────────────────────────────

// Signed distance to 1D step (smooth version of step, same API)
float linearstep(float lo, float hi, float x) {
    return clamp((x - lo) / (hi - lo), 0.0, 1.0);
}

// Anti-aliased 1px line at distance d
float aafill(float d, float aa) { return clamp(-d / aa + 0.5, 0.0, 1.0); }
float aafill(float d)           { return aafill(d, fwidth(d)); }

// Checker pattern
float checker(vec2 p, float scale) {
    vec2 q = floor(p * scale);
    return mod(q.x + q.y, 2.0);
}

// Pseudo-random float from 2D seed
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

#endif // LIB_MATH_GLSL
