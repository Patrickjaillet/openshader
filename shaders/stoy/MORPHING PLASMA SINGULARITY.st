// ==========================================================
// NAME : MORPHING PLASMA SINGULARITY
// ==========================================================
// DESCRIPTION : A hybrid raymarching scene featuring a 
// box-to-sphere morphing central object. Includes soft 
// shadows, Fresnel reflections, and a procedural plasma 
// background projected via triplanar-style logic.
// ==========================================================
// Credits : Patrick JAILLET
// https://shaderstudio.xo.je
// https://renderforge.ct.ws

// --- Utilities & Math ---

// Standard 2D rotation matrix
mat2 rot(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

// Cosine-based color palette for smooth color transitions
vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.263, 0.416, 0.557);
    return a + b * cos(6.28318 * (c * t + d));
}

// Procedural Plasma: Combines distance from center and sine-wave interference
vec3 getPlasma(vec2 uv, float time) {
    vec2 p = uv;
    float d = length(p);
    vec3 col = palette(d + time * 0.4);
    // Layering sine/cosine for organic liquid movement
    col += palette(abs(sin(p.x * 2.0 + time)) + abs(cos(p.y * 2.0 + time * 0.5)));
    col = pow(col, vec3(2.0)); // Increase contrast
    return col;
}

// --- Signed Distance Functions (SDF) ---

// 

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdSphere(vec3 p, float s) {
    return length(p) - s;
}

// Scene Map: Handles object rotation, morphing, and the back wall
float map(vec3 p) {
    vec3 pObj = p;
    float t = iTime;
    
    // Rotate the object in 3D space
    pObj.xy *= rot(t * 0.6);
    pObj.xz *= rot(t * 0.8);
    
    // Calculate primitive distances
    float box = sdBox(pObj, vec3(0.7));
    float sphere = sdSphere(pObj, 0.9);
    
    // Morph factor: Oscillates between 0 (Box) and 1 (Sphere)
    float morph = smoothstep(-0.8, 0.8, sin(iTime));
    
    float dObj = mix(box, sphere, morph);
    float dWall = 5.0 - p.z; // Vertical plane at Z=5
    
    return min(dObj, dWall);
}

// --- Lighting & Rendering ---

// Finite difference method to find the surface normal
vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

// Soft Shadow algorithm: Checks for occluders along the light path
// 
float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 32; i++) {
        float h = map(ro + rd * t);
        if(h < 0.001) return 0.0; // Hard shadow hit
        res = min(res, k * h / t); // Penumbra calculation
        t += h;
        if(t > maxt) break;
    }
    return res;
}

// --- Main Pipeline ---

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    // Normalizing UVs (-1 to 1)
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    
    // ro = ray origin (camera), rd = ray direction
    vec3 ro = vec3(0.0, 0.0, -4.0);
    vec3 rd = normalize(vec3(uv, 1.5));
    
    float t = 0.0;
    float d = 0.0;
    int hitObj = 0; 
    
    // Raymarching Loop
    for(int i = 0; i < 100; i++) {
        vec3 p = ro + rd * t;
        d = map(p);
        if(abs(d) < 0.001) {
            // Determine if we hit the object or the wall
            if (p.z > 2.0) hitObj = 2; // Wall
            else hitObj = 1;          // Morphing Object
            break;
        }
        t += d;
        if(t > 20.0) break;
    }
    
    vec3 col = vec3(0.0);
    vec3 lightPos = vec3(2.0 * sin(iTime), 2.0, -3.0); // Moving light source
    
    if(t < 20.0) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 l = normalize(lightPos - p);
        float diff = max(dot(n, l), 0.0);
        
        if (hitObj == 2) { 
            // SHADING: BACK WALL
            vec3 plasma = getPlasma(p.xy * 0.3, iTime);
            float shadow = softShadow(p + n * 0.02, l, 0.05, 10.0, 16.0);
            col = plasma * (diff * 0.5 + 0.5) * shadow;
        } 
        else if (hitObj == 1) { 
            // SHADING: MORPHING OBJECT
            vec3 r = reflect(rd, n);
            // The object reflects the plasma texture from its environment
            vec3 refPlasma = getPlasma(r.xy + r.z, iTime);
            
            float spec = pow(max(dot(reflect(-l, n), -rd), 0.0), 30.0);
            // Fresnel: Higher reflection at shallow angles
            float fresnel = pow(1.0 + dot(rd, n), 2.0);
            
            col = refPlasma * 0.8 + spec + fresnel * 0.5;
        }
    }
    
    // Tonemapping and Gamma Correction
    col = col / (col + vec3(1.0));
    col = pow(col, vec3(0.4545));
    
    fragColor = vec4(col, 1.0);
}