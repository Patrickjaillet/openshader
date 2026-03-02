// ==========================================================
// NAME : BIOLOGICAL GYROID TUNNEL
// ==========================================================
// DESCRIPTION : A high-fidelity organic tunnel journey. It 
// combines a twisting camera path, gyroid-displaced walls, 
// volumetric density accumulation, and triplanar bump-mapped 
// textures for a visceral, cinematic experience.
// ==========================================================
// Credits : Patrick JAILLET
// https://shaderstudio.xo.je
// https://renderforge.ct.ws

#define T (t * .5)
#define A(a) abs(dot(sin(4.*t + .1*p.z + .06*p/a), vec3(a)))
#define R(a) mat2(cos(a + vec4(0, 33, 11, 0)))
#define N normalize

// --- Color & Path Utilities ---

// Generates a smooth cosine-based color palette
vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ) {
    return a + b*cos( 6.28318*(c*t+d) );
}

// Defines the camera and tunnel spline (horizontal and vertical swaying)
vec2 path(float z) {
    return vec2(sin(z * 0.15) * 2.5, cos(z * 0.1) * 1.5);
}

// Polynomial smooth minimum for blending geometries
float smin(float a, float b, float k) {
   float f = max(0., 1. - abs(b - a)/k);
   return min(a, b) - k*.25*f*f;
}

// Gyroid function: creates porous, organic-looking geometric volumes
float gyroid(vec3 p, float t) {
    vec3 q = cos(p/1.5);
    q += cos(T + q.yzx*3.)*.5;
    return 1.1 + dot(sin(p), cos(p.yzx));
}

// --- Texturing & Surface Logic ---

// Triplanar Mapping: projects a 2D texture from 3 axes and blends them.
// This prevents stretching on complex organic surfaces.
vec3 tex3D(sampler2D tex, in vec3 p, in vec3 n) {    
    n = max(n*n - .2, .001);
    n /= dot(n, vec3(1.)); 
    vec3 tx = texture(tex, p.yz).xyz;
    vec3 ty = texture(tex, p.zx).xyz;
    vec3 tz = texture(tex, p.xy).xyz;
    return mat3(tx*tx, ty*ty, tz*tz)*n; 
}

// Triplanar Bump Mapping: Perturbs the normal based on texture grayscale gradients
vec3 texBump(sampler2D tx, in vec3 p, in vec3 n, float bf, float t) {
    const vec2 e = vec2(.001, 0);
    mat3 m = mat3(tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    vec3 g = vec3(.299, .587, .114)*m;
    g = (g - dot(tex3D(tx, p, n), vec3(.299, .587, .114)))/e.x; 
    g -= n*dot(n, g);
    return normalize(n + g*bf);
} 

// --- Scene Mapping (SDF) ---

float M(vec3 p, float t) {
    // Micro-texture displacement using multiple sine frequencies
    float tex = A(.001) - A(.002) + A(.003);
    
    // Position coordinates relative to the tunnel path
    p.xy -= path(p.z);
    
    // Main tunnel cylinder
    float tun = 5.0 - length(p.xy);
    
    // Add organic gyroid noise to the walls
    float gyr = gyroid(p * 0.6, t) * 1.2;
    float walls = tun - gyr;
    
    // Apply micro-texturing to the surface
    walls -= tex * 0.4;
    
    return walls * 0.6; // Safety multiplier for raymarching stability
}

// --- Rendering & Post-Process ---

void mainImage(out vec4 o, vec2 u) {
    vec3 r = iResolution, finalCol = vec3(0);
    float samples = 3.0; // Temporal samples for motion blur/smoothing
    
    for(float s_idx = 0.; s_idx < samples; s_idx++) {
        float t = iTime + (s_idx / samples) * 0.02;
        vec2 uv = (u + u - r.xy) / r.y;
        
        // 1. Camera Logic
        // ro = Ray Origin (tracking the path spline)
        vec3 ro = vec3(path(t * 5.0), t * 5.0);
        vec3 look = vec3(path(t * 5.0 + 1.0), t * 5.0 + 1.0);
        
        // Construct view matrix
        vec3 f = N(look - ro), ri = N(cross(vec3(0,1,0), f)),
             D = N(uv.x * ri + uv.y * cross(f, ri) + 1.2 * f);

        float i, d = 0., s, glow = 0., volume = 0.;
        vec3 p, col = vec3(0);

        // 2. Raymarching Loop
        for (i = 0.; i < 70.; i++) {
            p = ro + D * d;
            s = M(p, t);
            
            // Volumetric Density: Accumulates fog/haze near the gyroid structures
            float dens = gyroid(p * 0.2, t) * 0.5 + 0.5;
            dens *= smoothstep(5.0, 0.0, length(p.xy - path(p.z)));
            volume += dens * dens * 0.15;
            
            // Proximity Glow: Accumulates light based on distance to surface
            glow += 0.015 / (0.01 + s*s);
            
            if (abs(s) < .01 || d > 40.) break;
            d += s * 0.8;
        }

        // 3. Shading & Coloring
        vec3 tintCol = palette(p.z * 0.03 + t * 0.1, 
                               vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.33, 0.67));
        tintCol = pow(tintCol, vec3(1.5)) * 5.0;

        if (d < 40.) {
            // Calculate surface normal
            vec2 e = vec2(.005, 0);
            vec3 n = normalize(vec3(M(p + e.xyy, t) - M(p - e.xyy, t),
                                    M(p + e.yxy, t) - M(p - e.yxy, t),
                                    M(p + e.yyx, t) - M(p - e.yyx, t)));

            // Apply triplanar bump mapping and texturing from iChannel0
            vec3 q = texBump(iChannel0, T+p, n, .03, t);
            col = tex3D(iChannel0, T+p, n);
            col = col * col * tintCol;
            
            // Specular Reflection (Phong-lite)
            float spec = pow(max(dot(reflect(D, q), -D), 0.0), 20.0);
            col = col * max(dot(q, -D), .1) + spec * 0.5;
        }
        
        // Composite Glow and Volumetric layers
        col += glow * tintCol * 0.02;
        col += volume * tintCol * 0.1;
        
        // Apply distance-based exponential fog
        finalCol += col * exp(-0.05 * d);
    }

    // 4. Final Output Processing
    finalCol /= samples;
    // Square root and tanh tonemapping for HDR-to-LDR cinematic look
    o = vec4(tanh(sqrt(finalCol * 0.8)), 1.0);
}