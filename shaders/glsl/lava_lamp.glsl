#version 330 core
// Lava Lamp - Pure GLSL Format
// Métaballs flottantes avec effet chaleur et distorsion

uniform vec2  uResolution;
uniform float uTime;
uniform float uIntensity;

out vec4 fragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Champ de métaballs
float metafield(vec2 uv) {
    float field = 0.0;
    // 7 billes animées
    for (int i = 0; i < 7; i++) {
        float fi = float(i);
        float speed = 0.4 + hash(vec2(fi, 1.0)) * 0.5;
        float phase = hash(vec2(fi, 2.0)) * 6.28;
        float radius = 0.08 + hash(vec2(fi, 3.0)) * 0.12;

        // Trajectoire sinusoïdale
        vec2 center = vec2(
            0.5 + 0.35 * sin(uTime * speed + phase),
            0.3 + 0.55 * (0.5 + 0.5 * sin(uTime * speed * 0.7 + phase * 1.3))
        );

        float dist = length(uv - center);
        field += (radius * radius) / (dist * dist + 0.0001);
    }
    return field;
}

vec3 lavaColor(float field, float t) {
    // Isosurface threshold = 1.0
    float edge = smoothstep(0.9, 1.1, field);

    // Gradient interne (chaud au centre, plus sombre aux bords)
    float heat = clamp(field - 1.0, 0.0, 2.0) * 0.5;

    // Palette lave : noir → rouge → orange → jaune
    vec3 cold   = vec3(0.05, 0.0, 0.0);
    vec3 warm   = vec3(0.8, 0.1, 0.0);
    vec3 hot    = vec3(1.0, 0.5, 0.05);
    vec3 bright = vec3(1.0, 0.95, 0.5);

    vec3 lava = mix(cold, warm, smoothstep(0.0, 0.3, heat));
    lava = mix(lava, hot,    smoothstep(0.2, 0.6, heat));
    lava = mix(lava, bright, smoothstep(0.5, 1.0, heat));

    return lava * edge;
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    float aspect = uResolution.x / uResolution.y;
    uv.x *= aspect;
    // Recentrer
    uv.x /= aspect;

    float t = uTime;
    float intensity = 0.5 + uIntensity * 0.5;

    // Distorsion thermique (shimmer)
    float shimmer = sin(uv.x * 40.0 + t * 3.0) * sin(uv.y * 50.0 - t * 2.0);
    vec2 distort = uv + vec2(shimmer * 0.003, shimmer * 0.002) * intensity;

    float field = metafield(distort) * intensity;

    // Fond : verre sombre avec reflets
    vec3 glass = vec3(0.02, 0.01, 0.0);
    float vignette = 1.0 - smoothstep(0.3, 0.9, length(uv - 0.5));
    glass *= vignette;

    // Lueur ambiante rouge
    float glow = metafield(uv) * 0.3;
    glass += vec3(0.3, 0.05, 0.0) * smoothstep(0.5, 1.2, glow) * 0.3;

    // Lave
    vec3 col = glass + lavaColor(field, t);

    // Effet brillance (spéculaire)
    float spec = smoothstep(1.05, 1.3, field);
    col += vec3(1.0, 0.9, 0.7) * spec * 0.6;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
