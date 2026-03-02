#version 330 core
// Aurora Borealis - Pure GLSL Format
// Aurores boréales ondulantes avec couches de lumière

uniform vec2  uResolution;
uniform float uTime;
uniform float uIntensity;

out vec4 fragColor;

float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i.x + i.y * 57.0);
    float b = hash(i.x + 1.0 + i.y * 57.0);
    float c = hash(i.x + (i.y + 1.0) * 57.0);
    float d = hash(i.x + 1.0 + (i.y + 1.0) * 57.0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 6; i++) {
        v += amp * noise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return v;
}

vec3 aurora(vec2 uv, float t) {
    // Distorsion horizontale animée
    float wave = fbm(vec2(uv.x * 2.0, t * 0.3)) * 0.4;
    float y = uv.y + wave - 0.2;

    // Gradient vertical (aurore dans le tiers supérieur)
    float band = exp(-abs(y - 0.6) * 8.0);
    band += exp(-abs(y - 0.75) * 12.0) * 0.5;

    // Couleurs variant dans le temps
    float hue = fbm(vec2(uv.x * 3.0 + t * 0.1, t * 0.15)) * 2.0;
    vec3 col1 = vec3(0.0, 0.8 + 0.2 * sin(hue), 0.4 + 0.4 * cos(hue * 1.3));    // cyan-vert
    vec3 col2 = vec3(0.3 + 0.3 * sin(hue * 0.7), 0.0, 0.9 + 0.1 * cos(hue));   // violet
    vec3 col3 = vec3(0.0, 0.6 + 0.4 * cos(hue * 2.0), 0.8);                      // bleu

    float m = fbm(vec2(uv.x * 4.0 - t * 0.2, uv.y + t * 0.05));
    vec3 aurora_col = mix(mix(col1, col2, m), col3, sin(t * 0.2 + uv.x) * 0.5 + 0.5);

    return aurora_col * band * (0.6 + uIntensity * 0.4);
}

vec3 stars(vec2 uv, float t) {
    vec3 col = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        vec2 p = uv * (30.0 + float(i) * 20.0);
        vec2 id = floor(p);
        vec2 gv = fract(p) - 0.5;
        float h = hash(id.x + id.y * 13.7 + float(i) * 100.0);
        float size = 0.03 + h * 0.04;
        float twinkle = 0.7 + 0.3 * sin(t * (2.0 + h * 5.0) + h * 6.28);
        float star = smoothstep(size, 0.0, length(gv)) * twinkle;
        col += vec3(0.8, 0.9, 1.0) * star * h;
    }
    return col;
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    float t = uTime * 0.5;

    // Fond nocturne dégradé
    vec3 sky = mix(vec3(0.0, 0.02, 0.08), vec3(0.0, 0.0, 0.02), uv.y);

    // Étoiles
    vec3 col = sky + stars(uv, uTime);

    // Aurore
    col += aurora(uv, t);

    // Reflet sur sol enneigé (bas de l'écran)
    if (uv.y < 0.15) {
        float snow = fbm(vec2(uv.x * 5.0, uTime * 0.05)) * 0.05;
        vec3 ground = vec3(0.05, 0.08, 0.15) + snow;
        float blend = smoothstep(0.0, 0.15, uv.y);
        // Reflet aurora dans la neige
        vec3 reflect_uv_col = aurora(vec2(uv.x, 0.15 - uv.y + 0.15), t) * 0.4;
        col = mix(ground + reflect_uv_col, col, blend);
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));
    fragColor = vec4(col, 1.0);
}
