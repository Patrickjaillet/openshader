#version 330 core
// Crystal Geode - Pure GLSL Format
// Géode cristalline avec lumière interne et facettes

uniform vec2  uResolution;
uniform float uTime;
uniform float uIntensity;

out vec4 fragColor;

vec2 hash2(vec2 p) {
    return fract(sin(vec2(
        dot(p, vec2(127.1, 311.7)),
        dot(p, vec2(269.5, 183.3))
    )) * 43758.5453);
}

// Voronoi avec distance et ID de cellule
vec3 voronoi(vec2 p) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float min_d = 1e9;
    float min_d2 = 1e9;
    vec2 min_id = vec2(0.0);

    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            vec2 offset = vec2(float(i), float(j));
            vec2 h = hash2(ip + offset);
            // Animer légèrement les cellules
            h = 0.5 + 0.5 * sin(uTime * 0.3 + 6.28 * h);
            vec2 r = offset + h - fp;
            float d = dot(r, r);
            if (d < min_d) {
                min_d2 = min_d;
                min_d = d;
                min_id = ip + offset;
            } else if (d < min_d2) {
                min_d2 = d;
            }
        }
    }
    return vec3(sqrt(min_d), sqrt(min_d2), min_id.x + min_id.y * 31.1);
}

// Bruit pour la texture de la géode
float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * (fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453) * 2.0 - 1.0);
        p = p * 2.0 + vec2(5.2, 1.3);
        a *= 0.5;
    }
    return v;
}

vec3 crystalColor(float cell_id, float depth, float t) {
    // Palette gemmes : améthyste, tourmaline, quartz rose, aigue-marine
    float hue = fract(cell_id * 0.137 + t * 0.02);
    vec3 palettes[5];
    palettes[0] = vec3(0.6, 0.1, 1.0);   // améthyste
    palettes[1] = vec3(0.0, 0.8, 0.7);   // aigue-marine
    palettes[2] = vec3(1.0, 0.3, 0.6);   // tourmaline
    palettes[3] = vec3(0.2, 0.6, 1.0);   // saphir
    palettes[4] = vec3(0.9, 0.7, 0.2);   // citrine

    int idx = int(cell_id * 5.0) % 5;
    vec3 base = palettes[idx];

    // Variation de luminosité selon profondeur
    float brightness = 0.4 + 0.6 * depth;
    return base * brightness;
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    float aspect = uResolution.x / uResolution.y;
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
    float t = uTime * 0.3;

    // Zoom oscillant pour effet respiration
    float zoom = 3.0 + sin(t * 0.4) * 0.5;

    // Déformation polaire : géode circulaire
    float r = length(p);
    float angle = atan(p.y, p.x);

    // Distorsion radiale (facettes irrégulières)
    float distort = 0.05 * sin(angle * 7.0 + t) * sin(angle * 11.0 - t * 0.7);
    float r_distorted = r + distort;

    // Coordonnées voronoï
    vec2 vp = vec2(
        cos(angle) * r_distorted,
        sin(angle) * r_distorted
    ) * zoom;

    vec3 vor = voronoi(vp);
    float d1 = vor.x;
    float d2 = vor.y;
    float cell_id = fract(vor.z * 0.1731);

    // Bord des cristaux (joints)
    float edge = d2 - d1;
    float crystal_edge = smoothstep(0.04, 0.12, edge);

    // Profondeur illusoire : cellules plus petites = plus enfoncées
    float depth = 1.0 - d1 * 2.0;
    depth = clamp(depth, 0.0, 1.0);

    // Couleur de la facette
    vec3 base_col = crystalColor(cell_id, depth, t);

    // Lumière interne (source centrale chaude)
    float light_dist = r;
    float inner_light = exp(-light_dist * 2.5) * (0.6 + uIntensity * 0.4);
    vec3 light_col = vec3(1.0, 0.85, 0.6) * inner_light;

    // Reflets spéculaires sur les facettes
    float spec_angle = fract(cell_id + t * 0.1) * 6.28;
    vec2 spec_dir = vec2(cos(spec_angle), sin(spec_angle));
    float spec = max(0.0, dot(normalize(p + vec2(0.1)), spec_dir));
    spec = pow(spec, 6.0) * crystal_edge;

    // Transparence/brillance des bords
    float rim = (1.0 - crystal_edge) * 0.8;
    vec3 rim_col = vec3(1.0) * rim;

    // Composition
    vec3 col = base_col * crystal_edge;
    col += light_col * crystal_edge;
    col += vec3(1.0, 0.95, 1.0) * spec * 1.5;
    col += rim_col * 0.3;

    // Vignette (bords de la géode)
    float geode_mask = smoothstep(0.55, 0.35, r);
    col *= geode_mask;

    // Fond rocheux (extérieur de la géode)
    float rock_noise = 0.5 + 0.5 * fbm(p * 3.0 + t * 0.05);
    vec3 rock = mix(vec3(0.05, 0.03, 0.02), vec3(0.15, 0.1, 0.08), rock_noise);
    col = mix(rock, col, geode_mask);

    // Lueur globale
    col += base_col * 0.05 * (1.0 - geode_mask);

    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));
    fragColor = vec4(col, 1.0);
}
