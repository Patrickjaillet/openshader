// ==========================================================
// NAME : ANCIENT DESERT RELIC
// ==========================================================
// DESCRIPTION : A high-fidelity raymarched scene featuring a 
// floating fractal relic over procedural sand dunes. It utilizes 
// KIFS folding for the geometry, soft shadows, ambient occlusion, 
// and a complete PBR-inspired lighting stack.
// ==========================================================
// Credits : Patrick JAILLET
// https://shaderstudio.xo.je
// https://renderforge.ct.ws

#if HW_PERFORMANCE==0
#define AA 1
#else
#define AA 2
#endif

#define ZERO (min(iFrame,0))

// --- Mathematical Utilities ---

// Standard 2D rotation matrix for coordinate manipulation.
mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// Union operator for SDFs: returns the closest distance and the associated Material ID.
vec2 opU(vec2 d1, vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
}

// Signed Distance Function for a 3D Box.
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// --- Scene Mapping ---

vec2 map(vec3 p) {
    // 1. TERRAIN: SAND DUNES
    // Constructed using layered sine waves at different frequencies to simulate wind-blown sand.
    float dSand = p.y + 1.2;
    dSand += 0.4 * sin(p.x * 0.3) * cos(p.z * 0.3); // Large dunes
    dSand += 0.1 * sin(p.x * 1.2 - p.z * 0.8);      // Medium ripples
    dSand += 0.015 * sin(p.x * 12.0) * cos(p.z * 12.0); // Fine sand texture
    vec2 res = vec2(dSand, 1.0); // Material ID 1.0: Sand
    
    // 2. THE CENTRAL RELIC (KIFS Fractal)
    // We isolate a local coordinate space 'q' for the relic, applying floating animation and rotation.
    vec3 q = p;
    q.y -= -0.2 + 0.3 * sin(iTime * 0.6);
    q.xy *= rot(iTime * 0.1);
    q.xz *= rot(iTime * 0.15);
    
    float s = 1.0; // Scaling factor to track space compression
    vec3 fq = q;
    // Kaleidoscopic Iterated Function System loop:
    // This folds space using abs() and rotates it repeatedly to create self-similar complexity.
    for (int i = 0; i < 5; i++) {
        fq = abs(fq) - vec3(0.6, 0.8, 0.6); // Fold
        fq.xy *= rot(0.5);                  // Rotate
        fq.xz *= rot(0.3);
        fq *= 1.4;                          // Scale up
        s *= 1.4;
    }
    // The final shape is a box evaluated in the highly distorted space 'fq'.
    float dRelic = (sdBox(fq, vec3(0.4))) / s - 0.005;
    res = opU(res, vec2(dRelic, 2.0)); // Material ID 2.0: Dark Relic
    
    // 3. ORBITAL RINGS
    // Constructed using length() operations to create thin circular tori.
    vec3 r = p;
    r.y -= -0.2 + 0.3 * sin(iTime * 0.6);
    r.xz *= rot(-iTime * 0.2);
    r.xy *= rot(0.2 * sin(iTime * 0.5));
    float dRing1 = length(vec2(length(r.xz) - 2.8, r.y)) - 0.08;
    
    vec3 r2 = p;
    r2.y -= -0.2 + 0.3 * sin(iTime * 0.6);
    r2.xz *= rot(iTime * 0.25);
    r2.yz *= rot(0.3 * cos(iTime * 0.4));
    float dRing2 = length(vec2(length(r2.xz) - 3.5, r2.y)) - 0.05;
    
    res = opU(res, vec2(dRing1, 3.0)); // Material ID 3.0: Gold
    res = opU(res, vec2(dRing2, 3.0));
    
    return res;
}

// --- Rendering Engine ---

// Raycasting function to find intersections with the scene geometry.
vec2 raycast(vec3 ro, vec3 rd) {
    vec2 res = vec2(-1.0, -1.0);
    float t = 0.1;
    for (int i = 0; i < 200; i++) {
        vec2 h = map(ro + rd * t);
        // Precision increases as we get farther away to optimize performance.
        if (abs(h.x) < 0.001 * t) {
            res = vec2(t, h.y);
            break;
        }
        t += h.x * 0.7; // Use a 0.7 multiplier for safer marching through thin fractal walls.
        if (t > 40.0) break;
    }
    return res;
}

// Calculates the surface normal using finite difference.
vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
    ));
}

// Soft Shadow calculation using the penumbra method.
float calcShadow(vec3 ro, vec3 rd) {
    float res = 1.0;
    float t = 0.05;
    for (int i = 0; i < 40; i++) {
        float h = map(ro + rd * t).x;
        res = min(res, 12.0 * h / t);
        t += clamp(h, 0.02, 0.3);
        if (h < 0.001 || t > 15.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Ambient Occlusion: Simulates light blocking in crevices and corners.
float calcOcclusion(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = ZERO; i < 5; i++) {
        float h = 0.02 + 0.1 * float(i) / 4.0;
        float d = map(p + h * n).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

// Main lighting and shading function.
vec3 render(vec3 ro, vec3 rd) {
    vec3 sunDir = normalize(vec3(0.8, 0.15, 0.6));
    
    // Background Sky: Gradient with a procedural sun glow.
    vec3 sky = mix(vec3(0.9, 0.5, 0.3), vec3(0.2, 0.3, 0.5), clamp(rd.y, 0.0, 1.0));
    float sun = pow(max(dot(rd, sunDir), 0.0), 150.0);
    sky += vec3(1.0, 0.9, 0.6) * sun * 2.0;
    
    vec3 col = sky;
    vec2 res = raycast(ro, rd);
    
    if (res.y > -0.5) {
        float t = res.x;
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 ref = reflect(rd, n);
        
        // Define Material Properties based on ID
        vec3 albedo = vec3(0.0);
        float roughness = 1.0;
        
        if (res.y < 1.5) {      // SAND
            albedo = vec3(0.85, 0.55, 0.3);
            roughness = 0.9;
        } else if (res.y < 2.5) { // RELIC
            albedo = vec3(0.05, 0.05, 0.06);
            roughness = 0.3;
        } else {                // GOLD RINGS
            albedo = vec3(0.9, 0.7, 0.2);
            roughness = 0.2;
        }
        
        float occ = calcOcclusion(p, n);
        float sha = calcShadow(p, sunDir);
        
        // Lighting Components:
        // Diffuse: Standard Lambertian light.
        float dif = clamp(dot(n, sunDir), 0.0, 1.0);
        // Ambient: Sky light reflection.
        float amb = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
        // Back-light: Subsurface-like scattering from the opposite side.
        float bac = clamp(dot(n, normalize(vec3(-sunDir.x, 0.0, -sunDir.z))), 0.0, 1.0);
        // Fresnel: Reflectivity at grazing angles.
        float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 5.0);
        
        vec3 lin = vec3(0.0);
        lin += dif * vec3(1.2, 0.9, 0.6) * sha;
        lin += amb * vec3(0.3, 0.4, 0.5) * occ;
        lin += bac * vec3(0.2, 0.15, 0.1) * occ;
        
        col = albedo * lin;
        
        // Specular layer for shiny materials (Relic and Gold).
        if (roughness < 0.5) {
            float spe = pow(clamp(dot(ref, sunDir), 0.0, 1.0), mix(10.0, 100.0, 1.0 - roughness));
            col += vec3(1.0, 0.9, 0.7) * spe * sha * (1.0 - roughness);
            col += sky * fre * (1.0 - roughness) * occ * 0.5;
        }
        
        // Atmosphere: Fog blending based on distance.
        col = mix(col, sky, 1.0 - exp(-0.002 * t * t));
    }
    
    return col;
}

// Camera matrix setup for a LookAt system.
mat3 setCamera(vec3 ro, vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 tot = vec3(0.0);

    // Multi-Sampling Loop for Anti-Aliasing (AA).
#if AA > 1
    for (int m = ZERO; m < AA; m++)
    for (int n = ZERO; n < AA; n++) {
        vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
        vec2 p = (-iResolution.xy + 2.0 * (fragCoord + o)) / iResolution.y;
#else
        vec2 p = (-iResolution.xy + 2.0 * fragCoord) / iResolution.y;
#endif

        // Cinematic Camera Orbit.
        vec3 ro = vec3(5.0 * cos(iTime * 0.15), 0.5 + 0.5 * sin(iTime * 0.1), 5.0 * sin(iTime * 0.15));
        vec3 ta = vec3(0.0, -0.2, 0.0); // Target look-at point.
        
        mat3 ca = setCamera(ro, ta, 0.0);
        vec3 rd = ca * normalize(vec3(p, 2.0));

        vec3 col = render(ro, rd);
        
        // Gamma Correction for linear to sRGB space.
        col = pow(col, vec3(0.4545));
        tot += col;

#if AA > 1
    }
    tot /= float(AA * AA);
#endif

    // Post-Processing: Subtle Vignette to draw focus to the relic.
    vec2 q = fragCoord / iResolution.xy;
    tot *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);
    
    fragColor = vec4(tot, 1.0);
}