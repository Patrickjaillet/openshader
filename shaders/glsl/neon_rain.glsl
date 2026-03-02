#version 330 core
// Neon City Rain - Pure GLSL Format
// Reflets de néons dans une rue pluvieuse

uniform vec2  uResolution;
uniform float uTime;
uniform float uIntensity;

out vec4 fragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash1(float x) {
    return fract(sin(x * 127.1) * 43758.5453);
}

// Pluie : gouttes qui tombent
float rain(vec2 uv, float t) {
    float r = 0.0;
    for (int layer = 0; layer < 4; layer++) {
        float fl = float(layer);
        float scale = 15.0 + fl * 10.0;
        float speed = 1.5 + fl * 0.8;
        vec2 p = uv * vec2(scale, scale * 0.15);
        p.y -= t * speed;
        vec2 id = floor(p);
        float offset = hash1(id.x + fl * 100.0);
        p.y += offset;
        vec2 gv = fract(p) - vec2(0.5, 0.0);
        float drop = smoothstep(0.04, 0.0, abs(gv.x));
        drop *= smoothstep(0.5, 0.3, abs(gv.y - 0.3));
        drop *= (0.3 + 0.7 * hash(id + fl));
        r += drop * (0.5 / (fl + 1.0));
    }
    return clamp(r, 0.0, 1.0);
}

// Néon : signe lumineux
vec3 neonSign(vec2 uv, vec2 pos, vec2 size, vec3 color, float t, float flicker_seed) {
    vec2 d = abs(uv - pos) - size;
    float dist = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    float flicker = 0.85 + 0.15 * sin(t * 8.0 * flicker_seed + flicker_seed);
    float glow = exp(-abs(dist) * 15.0) * flicker;
    float core = exp(-abs(dist) * 40.0) * flicker;
    return color * (glow * 0.6 + core) * 1.5;
}

// Flaque d'eau avec reflets
vec3 puddle(vec2 uv, float t) {
    // Ondulations concentriques
    float ripple = sin(length(uv - vec2(0.5, 0.15)) * 40.0 - t * 5.0) * 0.5 + 0.5;
    ripple += sin(length(uv - vec2(0.3, 0.1)) * 35.0 - t * 4.0) * 0.5 + 0.5;
    ripple *= 0.5;
    return vec3(ripple * 0.05);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    float t = uTime * 0.5;
    float aspect = uResolution.x / uResolution.y;

    // Perspective : trottoir en bas, ciel en haut
    float horizon = 0.55;
    bool isGround = uv.y < horizon;

    vec2 uvAspect = vec2(uv.x * aspect, uv.y);

    // Fond nuit
    vec3 col = vec3(0.01, 0.01, 0.03);

    // Immeubles silhouette
    float building1 = step(uv.x, 0.2) * step(0.4, uv.y) * step(uv.y, 1.0);
    float building2 = step(0.15, uv.x) * step(uv.x, 0.38) * step(0.55, uv.y);
    float building3 = step(0.65, uv.x) * step(0.45, uv.y);
    float building4 = step(0.55, uv.x) * step(uv.x, 0.72) * step(0.6, uv.y);
    float buildings = clamp(building1 + building2 + building3 + building4, 0.0, 1.0);
    // Fenêtres lumineuses
    vec2 win = fract(uv * vec2(20.0, 15.0));
    float window = step(0.3, win.x) * step(win.x, 0.7) * step(0.2, win.y) * step(win.y, 0.8);
    float win_on = step(0.6, hash(floor(uv * vec2(20.0, 15.0))));
    col = mix(col, vec3(0.02, 0.02, 0.04), buildings);
    col += vec3(0.3, 0.25, 0.15) * window * win_on * buildings * 0.3;

    // Néons colorés
    float i = uIntensity * 0.5 + 0.5;
    col += neonSign(uvAspect, vec2(0.5 * aspect, 0.68), vec2(0.12 * aspect, 0.02), vec3(1.0, 0.1, 0.4), uTime, 1.3) * i;
    col += neonSign(uvAspect, vec2(0.25 * aspect, 0.72), vec2(0.06 * aspect, 0.015), vec3(0.1, 0.8, 1.0), uTime, 2.1) * i;
    col += neonSign(uvAspect, vec2(0.78 * aspect, 0.65), vec2(0.08 * aspect, 0.018), vec3(0.4, 1.0, 0.2), uTime, 0.7) * i;
    col += neonSign(uvAspect, vec2(0.12 * aspect, 0.62), vec2(0.05 * aspect, 0.012), vec3(1.0, 0.5, 0.0), uTime, 1.7) * i;

    // Sol mouillé : reflets des néons
    if (isGround) {
        float mirror_y = (horizon - uv.y) / horizon;
        vec2 mirror_uv = vec2(uvAspect.x, (horizon + mirror_y * 0.3) * aspect / aspect);
        vec3 reflection = vec3(0.0);
        reflection += neonSign(vec2(uvAspect.x, horizon + mirror_y * 0.2), vec2(0.5 * aspect, 0.68), vec2(0.12 * aspect, 0.02), vec3(1.0, 0.1, 0.4), uTime, 1.3);
        reflection += neonSign(vec2(uvAspect.x, horizon + mirror_y * 0.2), vec2(0.25 * aspect, 0.72), vec2(0.06 * aspect, 0.015), vec3(0.1, 0.8, 1.0), uTime, 2.1);
        reflection += neonSign(vec2(uvAspect.x, horizon + mirror_y * 0.2), vec2(0.78 * aspect, 0.65), vec2(0.08 * aspect, 0.018), vec3(0.4, 1.0, 0.2), uTime, 0.7);

        float wet = smoothstep(horizon, horizon - 0.1, uv.y);
        col = mix(col, vec3(0.03, 0.03, 0.06), 0.6);
        col += reflection * wet * 0.5 * i;
        col += puddle(uv, uTime) * wet;
    }

    // Pluie
    float r = rain(uv * vec2(1.0, aspect / uResolution.x * uResolution.y), uTime);
    col += vec3(0.5, 0.6, 0.8) * r * 0.4 * i;

    // Brume légère
    col += vec3(0.02, 0.02, 0.04) * smoothstep(0.5, 0.0, uv.y) * 0.5;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));
    fragColor = vec4(col, 1.0);
}
