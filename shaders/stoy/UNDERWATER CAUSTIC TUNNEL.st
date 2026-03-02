// ==========================================================
// UNDERWATER CAUSTIC TUNNEL
// ==========================================================
// A deep-sea raymarching effect featuring procedural caustics,
// swirling water displacement, and volumetric lighting.
// Credits: Patrick JAILLET
// ==========================================================

// 1. MATHEMATICAL UTILITIES
// ----------------------------------------------------------

// Standard 2D rotation matrix.
mat2 rot(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

// 3D Hash function used for procedural noise generation.
float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// 3D Value Noise: Smoothly interpolates between hash values.
float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = hash31(i);
    float b = hash31(i + vec3(1.0, 0.0, 0.0));
    float c = hash31(i + vec3(0.0, 1.0, 0.0));
    float d = hash31(i + vec3(1.0, 1.0, 0.0));
    float e = hash31(i + vec3(0.0, 0.0, 1.0));
    float _f = hash31(i + vec3(1.0, 0.0, 1.0));
    float g = hash31(i + vec3(0.0, 1.0, 1.0));
    float h = hash31(i + vec3(1.0, 1.0, 1.0));
    
    return mix(
        mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
        mix(mix(e, _f, f.x), mix(g, h, f.x), f.y),
        f.z
    );
}

// 2. PROCEDURAL PATH & DISPLACEMENT
// ----------------------------------------------------------

// Defines the "S-curve" path of the tunnel based on depth (z).
vec2 getPath(float z) {
    float x = sin(z * 0.1) * 4.0 + sin(z * 0.3) * 1.5;
    float y = cos(z * 0.15) * 2.0;
    return vec2(x, y);
}

// FBM Bump: Layers absolute noise values to create sharp,
// "caustic-like" ridges in the geometry and lighting.
float fbmBump(vec3 p) {
    float val = 0.0;
    float amp = 0.5;
    vec3 shift = vec3(0.0, 0.0, iTime * 0.5);
    
    for (int i = 0; i < 5; i++) {
        val += amp * abs(noise(p - shift) * 2.0 - 1.0); 
        p = p * 2.0;
        amp *= 0.5;
        p.xy *= rot(0.5);
        p.yz *= rot(0.3);
    }
    return val;
}

// 3. THE TUNNEL SDF
// ----------------------------------------------------------
float map(vec3 p) {
    // Offset position by the procedural path
    vec2 pathOffset = getPath(p.z);
    p.xy -= pathOffset;
    
    float tunnelRadius = 3.8;
    float dTunnel = tunnelRadius - length(p.xy);
    
    // Rotate coordinates around Z to create a swirling effect
    float swirlAngle = p.z * 0.1 + iTime * 0.5;
    vec3 pSwirl = p;
    pSwirl.xy *= rot(swirlAngle);
    
    // Apply organic displacement using FBM
    float displacement = fbmBump(pSwirl * 0.6) * 1.2;
    
    // Return distance (softened by 0.5 to prevent artifacts)
    return (dTunnel - displacement) * 0.5;
}

// Normal calculation for light reflections
vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.002, 0.0); 
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

// 4. MAIN RENDERING ENGINE
// ----------------------------------------------------------
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    
    // Camera movement: Moving forward along the procedural path
    float speed = 3.0;
    float camZ = iTime * speed;
    vec3 ro = vec3(0.0, 0.0, camZ);
    ro.xy += getPath(camZ);
    
    // Look-at Target: A point further ahead in the tunnel
    vec3 target = vec3(0.0, 0.0, camZ + 2.0);
    target.xy += getPath(camZ + 2.0);
    
    // View Matrix Setup
    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    vec3 rd = normalize(fwd * 1.5 + right * uv.x + up * uv.y);
    
    float t = 0.0;
    float d = 0.0;
    float glow = 0.0;
    
    // RAYMARCHING LOOP
    for(int i = 0; i < 150; i++) {
        vec3 p = ro + rd * t;
        d = map(p); 
        
        // Volumetric Glow accumulation: Based on proximity to surface
        float distMetric = max(d, 0.02);
        glow += 0.002 / (0.0005 + distMetric * distMetric);
        
        if(d < 0.001 || t > 60.0) break;
        t += d;
    }
    
    vec3 col = vec3(0.0);
    vec3 lightPos = ro + vec3(0.0, 1.0, 4.0); 

    if(t < 60.0) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 l = normalize(lightPos - p);
        vec3 viewDir = normalize(ro - p);
        
        // Local Swirl coordinates for surface texture
        vec3 pCorrected = p;
        pCorrected.xy -= getPath(p.z);
        float swirlAngle = pCorrected.z * 0.1 + iTime * 0.5;
        vec3 pSwirl = pCorrected;
        pSwirl.xy *= rot(swirlAngle);
        
        // Water Colors
        vec3 waterBase = vec3(0.0, 0.15, 0.2); 
        vec3 waterHigh = vec3(0.0, 0.5, 0.45); 
        
        // Surface albedo mixed by noise
        float bumpFactor = fbmBump(pSwirl * 0.6);
        vec3 baseCol = mix(waterBase, waterHigh, bumpFactor);

        // Standard Shading: Diffuse, Specular, and Fresnel
        float diff = max(dot(n, l), 0.0);
        float spec = pow(max(dot(reflect(-l, n), -viewDir), 0.0), 30.0);
        float fresnel = pow(1.0 - max(dot(viewDir, n), 0.0), 4.0);
        
        // CAUSTICS Simulation: High-power FBM focused into "light-threads"
        float caustics = fbmBump(pSwirl * 0.8 + vec3(0.0, 0.0, iTime));
        caustics = pow(caustics, 3.0); 
        float causticLight = diff * caustics;
        
        // Composite Final Color
        col = baseCol * (diff * 0.5 + 0.3); 
        col += vec3(0.6, 0.9, 0.8) * spec * 1.2; 
        col += vec3(0.2, 0.7, 0.6) * fresnel * 0.8;
        col += vec3(0.5, 0.9, 0.8) * causticLight * 1.5; 
        
        // Distance Fog: Fades everything into deep blue darkness
        float fog = 1.0 - exp(-t * 0.035);
        vec3 fogCol = vec3(0.0, 0.08, 0.12);
        col = mix(col, fogCol, fog);
    } else {
        col = vec3(0.0, 0.08, 0.12);
    }
    
    // Apply accumulated Cyan/Emerald glow
    col += vec3(0.1, 0.4, 0.5) * glow * 0.15;
    
    // Post-Process: Gamma Correction & Vignetting
    col = pow(col, vec3(0.4545));
    col *= pow(1.0 - length(uv * 0.4), 1.2);
    
    fragColor = vec4(col, 1.0);
}