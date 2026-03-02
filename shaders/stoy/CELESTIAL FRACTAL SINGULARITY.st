// ==========================================================
// NAME : CELESTIAL FRACTAL SINGULARITY
// ==========================================================
// DESCRIPTION : A deep-space visual combining a multi-layered 
// starfield, a domain-warped FBM nebula, and a centralized 
// trigonometric fractal. Features glitch-triggered horizontal 
// shifts and pulsating bloom.
// ==========================================================
// Credits : Patrick JAILLET
// Original by XOR -- https://x.com/XorDev/status/1970235131240817039

#define NUM_LAYERS 4.0
#define TIME_SPEED 0.1

// --- Mathematical Utilities ---

// Standard 2D Hash function for star distribution
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 2D Rotation matrix
mat2 rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

// --- Starfield System ---

vec3 StarLayer(vec2 uv, float t) {
    vec3 col = vec3(0);
    vec2 gv = fract(uv) - 0.5; // Grid UV (-0.5 to 0.5)
    vec2 id = floor(uv);       // Cell ID
    
    // Grid search to handle overlapping star glows
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 offset = vec2(x, y);
            float n = hash21(id + offset);
            float size = fract(n * 345.32);
            
            // Randomize star position within the cell
            vec2 p = offset - gv + (vec2(n, fract(n * 34.)) - 0.5) * 0.8;
            float d = length(p);
            
            // Sparkle animation
            float sparkle = sin(t * 2. + n * 6.2831) * 0.5 + 0.5;
            float star = 0.003 / (d * d); // Inverse square law for glow
            
            // Masking and coloring
            star *= smoothstep(1.0, 0.2, d);
            star *= smoothstep(0.2, 1.0, size);
            star *= 0.5 + sparkle * sparkle * 0.8;
            
            vec3 color = sin(vec3(0.2, 0.3, 0.9) * fract(n * 2345.2) * 6.28) * 0.5 + 0.5;
            col += star * size * (color + 0.5);
        }
    }
    return col;
}

// --- Nebula Background (FBM & Domain Warping) ---

float noise(in vec2 uv) {
    return sin(1.5 * uv.x) * sin(1.5 * uv.y);
}

// Fractal Brownian Motion: Summing multiple octaves of noise
float fbm(vec2 p) {
    float f = 0.0;
    mat2 m = mat2(0.8, 0.6, -0.6, 0.8);
    f += 0.5000 * noise(p); p = m * p * 2.02;
    f += 0.2500 * noise(p); p = m * p * 2.03;
    f += 0.1250 * noise(p); p = m * p * 2.01;
    f += 0.0625 * noise(p);
    return f / 0.9375;
}

// Domain Warping: FBM used as an input to another FBM
// This creates organic, fluid-like gaseous structures.
vec3 Nebula(vec2 uv, float t) {
    vec2 p = uv;
    p.x -= t * 0.03;
    p.y += t * 0.02;
    
    // Warping layers
    vec2 q = vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
    vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2)), fbm(p + 4.0 * q + vec2(8.3, 2.8)));
    
    float f = fbm(p + 4.0 * r);
    vec3 col = mix(vec3(0.05, 0.02, 0.1), vec3(0.1, 0.2, 0.3), clamp(f * f * 4.0, 0.0, 1.0));
    return f * col;
}

// --- Main Pipeline ---

void mainImage(out vec4 o, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float t = iTime * TIME_SPEED;
    
    // 1. NEBULA AND STARS (Background)
    vec3 backCol = Nebula(uv * 1.2, t);
    vec2 starUV = uv * rot(t * 0.05);
    
    // Parallax layers for depth
    for(float i = 0.0; i < 1.0; i += 1.0 / NUM_LAYERS) {
        float depth = fract(i + t);
        float scale = mix(20.0, 0.5, depth);
        float fade = smoothstep(0.0, 0.2, depth) * smoothstep(1.0, 0.8, depth);
        backCol += StarLayer(starUV * scale + i * 453.2, t) * fade;
    }

    // 2. CENTRAL BLOOM & GLITCH EFFECTS
    float pulse = 0.008 + 0.015 * sin(iTime * 1.2);
    float bloomMask = exp(-length(uv) * 5.0);
    vec3 bloomCol = vec3(1.0, 0.8, 0.4) * pulse * bloomMask;

    // Horizontal glitch trigger
    float glitchTime = iTime * 15.0;
    float glitchTrigger = step(0.99, fract(sin(dot(vec2(floor(glitchTime), uv.y*2.0), vec2(12.9898,78.233))) * 43758.5453));
    float glitchShift = (fract(sin(glitchTime) * 43758.5453) - 0.5) * 0.05 * glitchTrigger * bloomMask;

    // 3. THE FRACTAL CORE (Raymarched Singularity)
    vec4 fractalCol = vec4(0.0);
    float fi = 0.0, fz = 0.0, fd = 0.0, fs = 0.0;
    
    // Fractal loop: Uses Rodrigues' rotation formula and trigonometric iteration
    for(; fi++ < 90.0; fractalCol += vec4(fs, 1.0, fz / 5.0, 1.0) / fs / fs * fi) {
        vec2 gFrag = fragCoord + vec2(glitchShift * iResolution.x, 0.0);
        vec3 dir = vec3(gFrag * 2.0 - iResolution.xy, -iResolution.y);
        vec3 p = fz * normalize(dir);
        vec3 a = vec3(0.0);
        
        p.z += 9.0;
        a -= 0.6;
        
        // Complex space rotation and warping
        a = mix(dot(a, p) * a, p, cos(fs - iTime)) - sin(fs - iTime) * cross(a, p);
        fs = sqrt(length(a - a.zxy));
        
        // Trigonometric detail layering
        for(fd = 0.0; fd++ < 9.0; a += cos(a * fd + iTime).yzx / fd);
        
        fd = dot(a, a / a);
        fz += sqrt(max(fd, -fd * 0.1)) * fs / 14.0;
    }
    
    // Combine Bloom with Fractal
    fractalCol.rgb += bloomCol * 5e5;
    
    // 4. FINAL COMPOSITION
    // Use hyperbolic tangent (tanh) for smooth high-range compression
    vec3 finalFractal = tanh(fractalCol.rgb * fractalCol.rgb / 4e6);
    vec3 finalCol = backCol + finalFractal;
    
    o = vec4(finalCol, 1.0);
}