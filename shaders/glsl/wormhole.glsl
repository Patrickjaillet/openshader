#version 330 core
// Wormhole - Pure GLSL Format
// Voyage dans un tunnel de distorsion spatio-temporelle

uniform vec2  uResolution;
uniform float uTime;
uniform float uIntensity;

out vec4 fragColor;

float hash(float n) { return fract(sin(n) * 43758.5453); }
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash2(i), hash2(i + vec2(1,0)), f.x),
        mix(hash2(i + vec2(0,1)), hash2(i + vec2(1,1)), f.x),
        f.y
    );
}

// Étoiles filantes dans le tunnel
float starfield(vec2 uv, float z) {
    float stars = 0.0;
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        vec2 p = uv * (5.0 + fi * 3.0);
        vec2 id = floor(p);
        vec2 gv = fract(p) - 0.5;
        float h = hash2(id + fi * 37.0);
        float size = 0.03 + h * 0.02;
        // Stretch vers le bord (effet vitesse)
        float stretch = 1.0 + z * 0.5;
        vec2 sgv = gv * vec2(1.0, stretch);
        stars += smoothstep(size, 0.0, length(sgv)) * h;
    }
    return clamp(stars, 0.0, 1.0);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - uResolution.xy * 0.5) / uResolution.y;
    float t = uTime * (0.5 + uIntensity * 0.5);

    // Coordonnées polaires
    float r = length(uv);
    float angle = atan(uv.y, uv.x);

    // Tunnel : distorsion en perspective
    float tunnel_r = 1.0 / (r + 0.01);
    float tunnel_t = angle / (2.0 * 3.14159) + 0.5;
    float tunnel_z = tunnel_r - t * 2.0;

    // Texture du tunnel : grille hexagonale déformée
    vec2 tuv = vec2(tunnel_t * 8.0, tunnel_z * 2.0);
    // Ondulation des parois
    tuv.x += sin(tunnel_z * 3.0 + t) * 0.3;
    tuv.y += cos(tunnel_t * 6.0 + t * 0.5) * 0.2;

    // Grille du tunnel
    vec2 grid = fract(tuv) - 0.5;
    float grid_line = min(abs(grid.x), abs(grid.y));
    float tube = smoothstep(0.05, 0.0, grid_line) * exp(-r * 1.5);

    // Distorsion gravitationnelle autour du trou
    float warp = smoothstep(0.4, 0.0, r);
    vec2 warped = uv * (1.0 + warp * 2.0);
    float swirl = atan(warped.y, warped.x) + t * 2.0;
    warped = vec2(cos(swirl), sin(swirl)) * length(warped);

    // Nébuleuse colorée
    float nebula = noise(warped * 4.0 + t * 0.2) * noise(warped * 8.0 - t * 0.1);
    nebula = pow(nebula, 2.0);

    // Couleur du tunnel
    float hue = tunnel_t + t * 0.05;
    vec3 tube_col = vec3(
        0.5 + 0.5 * sin(hue * 6.28 + 0.0),
        0.5 + 0.5 * sin(hue * 6.28 + 2.09),
        0.5 + 0.5 * sin(hue * 6.28 + 4.19)
    );

    // Nébuleuse violette/bleue
    vec3 neb_col = mix(
        vec3(0.2, 0.0, 0.8),
        vec3(0.0, 0.5, 1.0),
        nebula
    );

    // Fond : trou noir au centre
    vec3 col = vec3(0.0);

    // Anneau lumineux autour du trou (relativité)
    float ring = exp(-abs(r - 0.12) * 25.0);
    col += vec3(1.0, 0.8, 0.3) * ring * 2.0;

    // Distorsion des étoiles d'arrière-plan
    col += vec3(0.7, 0.8, 1.0) * starfield(uv * 3.0, tunnel_r * 0.1) * smoothstep(0.15, 0.5, r);

    // Tunnel lumineux
    col += tube_col * tube * 1.5;

    // Nébuleuse
    col += neb_col * nebula * smoothstep(0.3, 0.0, r) * 0.8;

    // Vignette externe
    col *= smoothstep(1.0, 0.3, r);

    // Bord du trou de ver (brillance)
    float edge = smoothstep(0.18, 0.12, r) * smoothstep(0.0, 0.12, r);
    col += vec3(0.3, 0.6, 1.0) * edge * 0.5;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
