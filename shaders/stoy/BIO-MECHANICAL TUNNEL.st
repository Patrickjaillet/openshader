// ==========================================================
// BIO-MECHANICAL TUNNEL TUTORIAL
// ==========================================================
// A deep-dive into cinematic raymarching featuring a 
// procedural path, ambient occlusion, and volumetric glow.
// Credits: Patrick JAILLET
// ==========================================================

// --- Performance Settings ---
#if HW_PERFORMANCE==0
#define AA 1 // Low quality for mobile/older GPUs
#else
#define AA 2 // High quality anti-aliasing
#endif

#define ZERO (min(iFrame,0))

// 1. UTILITY & PATH FUNCTIONS
// ----------------------------------------------------------

// Standard 2D rotation
mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// Defines the camera and tunnel curves. 
// Changing these values changes the "winding" of the tunnel.
vec2 path(float z) {
    return vec2(sin(z * 0.2) * 1.5, cos(z * 0.15) * 1.2);
}

// Smooth blending function for organic geometry
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// 2. THE GEOMETRY (THE MAP)
// ----------------------------------------------------------
float map(vec3 p) {
    vec3 q = p;
    // Align tunnel geometry with the procedural path
    q.xy -= path(q.z);
    
    float a = atan(q.y, q.x);
    float r = length(q.xy);
    
    // Base Tunnel: An inverted cylinder with sine-wave deformation
    float base = 2.5 - r + 0.3 * sin(q.z * 1.5 - iTime * 1.5) * cos(a * 6.0 + q.z);
    
    // Ribs: Repetitive ring structures along the Z axis
    vec3 op = q;
    op.z = mod(op.z + iTime, 3.0) - 1.5;
    float ribs = length(vec2(r - 2.6, op.z)) - 0.3;
    ribs -= 0.15 * sin(a * 8.0 + p.z * 2.0);
    
    // Blend the ribs into the tunnel wall
    float d = smin(base, ribs, 0.6);
    
    // Micro-texture: high-frequency noise for a rough, biological look
    d -= 0.03 * sin(p.x * 12.0) * sin(p.y * 12.0) * sin(p.z * 12.0);
    
    return d * 0.5;
}

// 3. RAYMARCHING & VOLUMETRICS
// ----------------------------------------------------------
float raycast(vec3 ro, vec3 rd, out float glow) {
    float t = 0.01;
    glow = 0.0;
    for (int i = 0; i < 150; i++) {
        float h = map(ro + rd * t);
        
        // Glow accumulation: adds light based on proximity to surfaces
        glow += 0.005 / (0.01 + abs(h));
        
        if (abs(h) < 0.002 * t) return t;
        t += h;
        if (t > 50.0) break;
    }
    return -1.0;
}

// Surface Normal calculation for lighting
vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.002, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

// 4. LIGHTING & FOG
// ----------------------------------------------------------
// Ambient Occlusion: Darkens cracks and crevices
float calcOcclusion(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = ZERO; i < 5; i++) {
        float h = 0.05 + 0.15 * float(i);
        float d = map(p + h * n);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

vec3 render(vec3 ro, vec3 rd) {
    vec3 col = vec3(0.01, 0.015, 0.01); // Dark base color
    float glow = 0.0;
    
    float t = raycast(ro, rd, glow);
    
    if (t > 0.0) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        
        // Albedo: The base material color (mossy/bone gray)
        vec3 albedo = vec3(0.15, 0.16, 0.14);
        albedo += 0.05 * sin(p.z * 5.0) * sin(p.y * 5.0);
        
        float occ = calcOcclusion(p, n);
        
        // Lighting components: Diffuse + Fresnel (edge highlights)
        vec3 lig = normalize(vec3(0.0, 0.0, 1.0));
        float dif = max(dot(n, lig), 0.0);
        float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 3.0);
        
        vec3 lin = vec3(0.0);
        lin += dif * vec3(0.5, 0.6, 0.5) * occ;
        lin += vec3(0.1, 0.12, 0.1) * occ;
        lin += fre * vec3(0.3, 0.4, 0.3) * occ;
        
        col = albedo * lin;
        
        // Exponential Fog
        float fog = 1.0 - exp(-0.02 * t * t);
        col = mix(col, vec3(0.02, 0.03, 0.025), fog);
        
        // Core light: a faint red glow from within the tunnel
        vec3 coreLig = vec3(0.4, 0.1, 0.05) * exp(-0.1 * t);
        col += coreLig * (1.0 - fog) * 0.5;
    }
    
    // Add the accumulated volumetric glow
    col += vec3(0.2, 0.25, 0.15) * glow * 0.015;
    
    return col;
}

// 5. CAMERA & MAIN
// ----------------------------------------------------------
mat3 setCamera(vec3 ro, vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 tot = vec3(0.0);

#if AA > 1
    for (int m = ZERO; m < AA; m++)
    for (int n = ZERO; n < AA; n++) {
        vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
        vec2 p = (-iResolution.xy + 2.0 * (fragCoord + o)) / iResolution.y;
#else
        vec2 p = (-iResolution.xy + 2.0 * fragCoord) / iResolution.y;
#endif

        // Move camera forward along the path
        float camTime = iTime * 2.0;
        vec3 ro = vec3(path(camTime), camTime);
        vec3 ta = vec3(path(camTime + 2.0), camTime + 2.0); // Focus point
        
        mat3 ca = setCamera(ro, ta, sin(iTime * 0.5) * 0.2);
        vec3 rd = ca * normalize(vec3(p, 1.5));

        vec3 col = render(ro, rd);
        
        // Gamma Correction
        col = pow(col, vec3(0.4545));
        tot += col;

#if AA > 1
    }
    tot /= float(AA * AA);
#endif

    // Post-Process: Vignetting (darkened corners)
    vec2 q = fragCoord / iResolution.xy;
    tot *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(tot, 1.0);
}