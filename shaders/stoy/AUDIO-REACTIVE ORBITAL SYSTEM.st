// ==========================================================
// NAME : AUDIO-REACTIVE ORBITAL SYSTEM
// ==========================================================
// DESCRIPTION : A stylized 3D solar system featuring all eight 
// planets and a central sun. Orbits are procedurally calculated. 
// The scene reacts to bass frequencies (kick), causing planetary 
// pulsing, camera shake, and sun intensity fluctuations.
// ==========================================================
// Credits : Patrick JAILLET
// https://shaderstudio.xo.je
// https://renderforge.ct.ws

#define MAX_STEPS 128
#define MAX_DIST 400.0
#define SURF_DIST 0.01

// --- AUDIO LOGIC ---

// Fetches low-frequency data (kick) from iChannel0.
// If no audio is present, it provides a rhythmic fallback pulse.
float getKick() {
    float fft = texture(iChannel0, vec2(0.05, 0.25)).x;
    
    if (fft < 0.01) {
        return smoothstep(0.2, 0.0, abs(sin(iTime * 3.0))) * 0.5;
    }
    
    return pow(fft, 3.0) * 3.0;
}

// --- COSMIC DATA ---
// vec3(Orbital Distance, Orbital Speed, Radius)
vec3 orbMer = vec3(12.0, 4.0, 0.8);
vec3 orbVen = vec3(18.0, 3.0, 1.1);
vec3 orbEar = vec3(26.0, 2.0, 1.2);
vec3 orbMar = vec3(35.0, 1.5, 0.9);
vec3 orbJup = vec3(60.0, 0.8, 4.0);
vec3 orbSat = vec3(90.0, 0.6, 3.5);
vec3 orbUra = vec3(120.0, 0.4, 2.0);
vec3 orbNep = vec3(150.0, 0.2, 1.9);

// Calculates the 3D position of a planet on the XZ plane based on time.
vec3 getPos(float dist, float speed, float timeOffset) {
    float t = iTime * 0.5 * speed + timeOffset;
    return vec3(cos(t) * dist, 0.0, sin(t) * dist);
}

// --- SDF GEOMETRY ---

float sdSphere(vec3 p, float s) { return length(p) - s; }
float sdTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz) - t.x, p.y); return length(q) - t.y; }

// Simple hash and noise for surface textures
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.98, 78.23))) * 43758.54); }
float noise(vec3 p) {
    vec3 i = floor(p); vec3 f = fract(p); f = f*f*(3.0-2.0*f);
    return mix(mix(mix(hash(i.xy+vec2(0,0)), hash(i.xy+vec2(1,0)), f.x),
                   mix(hash(i.xy+vec2(0,1)), hash(i.xy+vec2(1,1)), f.x), f.y),
               mix(mix(hash(i.xy+vec2(0,0)), hash(i.xy+vec2(1,0)), f.x),
                   mix(hash(i.xy+vec2(0,1)), hash(i.xy+vec2(1,1)), f.x), f.y), f.z);
}

// --- SCENE MAPPING ---

vec2 map(vec3 p) {
    float d = MAX_DIST;
    float id = -1.0;
    
    float kick = getKick();
    float pulse = 1.0 + kick * 0.5; // Planets grow on beat
    
    // 1. SUN
    float dSun = sdSphere(p, 6.0 * (1.0 + kick * 0.1));
    if(dSun < d) { d = dSun; id = 0.0; }
    
    // 2. PLANETARY SYSTEM
    // Mercury
    vec3 pMer = p - getPos(orbMer.x, orbMer.y, 0.0);
    float dMer = sdSphere(pMer, orbMer.z * pulse);
    if(dMer < d) { d = dMer; id = 1.0; }

    // Venus
    vec3 pVen = p - getPos(orbVen.x, orbVen.y, 1.0);
    float dVen = sdSphere(pVen, orbVen.z * pulse);
    if(dVen < d) { d = dVen; id = 2.0; }

    // Earth
    vec3 pEar = p - getPos(orbEar.x, orbEar.y, 2.0);
    float dEar = sdSphere(pEar, orbEar.z * pulse);
    if(dEar < d) { d = dEar; id = 3.0; }
    
    // Mars
    vec3 pMar = p - getPos(orbMar.x, orbMar.y, 3.5);
    float dMar = sdSphere(pMar, orbMar.z * pulse);
    if(dMar < d) { d = dMar; id = 4.0; }
    
    // Jupiter
    vec3 pJup = p - getPos(orbJup.x, orbJup.y, 1.5);
    float dJup = sdSphere(pJup, orbJup.z * pulse);
    if(dJup < d) { d = dJup; id = 5.0; }
    
    // Saturn (Sphere + Torus Ring)
    vec3 posSat = getPos(orbSat.x, orbSat.y, 4.0);
    vec3 pSat = p - posSat;
    pSat.yz *= mat2(0.9, -0.4, 0.4, 0.9); // Tilt the ring
    float dSat = sdSphere(p - posSat, orbSat.z * pulse);
    float dRing = sdTorus(pSat, vec2(6.0 * (1.0+kick*0.2), 0.5)); 
    dSat = min(dSat, dRing);
    if(dSat < d) { d = dSat; id = 6.0; }
    
    // Uranus
    vec3 pUra = p - getPos(orbUra.x, orbUra.y, 5.0);
    float dUra = sdSphere(pUra, orbUra.z * pulse);
    if(dUra < d) { d = dUra; id = 7.0; }

    // Neptune
    vec3 pNep = p - getPos(orbNep.x, orbNep.y, 6.0);
    float dNep = sdSphere(pNep, orbNep.z * pulse);
    if(dNep < d) { d = dNep; id = 8.0; }
    
    return vec2(d, id);
}

// --- SHADING ---

// Assigns colors and texture noise based on the planet ID
vec3 getMaterial(float id, vec3 p, vec3 n) {
    vec3 col = vec3(0.0);
    float bright = 1.0 + getKick();
    
    if(id == 0.0) col = vec3(1.0, 0.5, 0.1) * (2.0 + noise(p*2.0 - iTime)); // Sun
    if(id == 1.0) col = vec3(0.5, 0.5, 0.5) * (0.8 + noise(p*10.0));       // Mercury
    if(id == 2.0) col = vec3(0.9, 0.7, 0.2);                               // Venus
    if(id == 3.0) col = mix(vec3(0.0, 0.2, 0.6), vec3(0.1, 0.5, 0.1), smoothstep(0.4, 0.6, noise(p*3.0))); // Earth
    if(id == 4.0) col = vec3(0.8, 0.3, 0.1);                               // Mars
    if(id == 5.0) col = vec3(0.8, 0.6, 0.4) * (0.8 + sin(p.y*10.0)*0.2);   // Jupiter
    if(id == 6.0) col = vec3(0.8, 0.7, 0.5);                               // Saturn
    if(id == 7.0) col = vec3(0.4, 0.8, 0.9);                               // Uranus
    if(id == 8.0) col = vec3(0.2, 0.3, 0.8);                               // Neptune
    
    return col * bright;
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(0.01, 0.0);
    float d = map(p).x;
    return normalize(vec3(d-map(p-e.xyy).x, d-map(p-e.yxy).x, d-map(p-e.yyx).x));
}

// --- MAIN RENDERING ---

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float kick = getKick();

    // Camera Orbit logic
    float camDist = 80.0 + sin(iTime * 0.1) * 40.0;
    float camSpeed = 0.2;
    vec3 ro = vec3(cos(iTime*camSpeed)*camDist, 40.0 + sin(iTime*0.2)*20.0, sin(iTime*camSpeed)*camDist);
    vec3 ta = vec3(0.0, 0.0, 0.0);
    
    // Apply camera shake on kick
    ro += (vec3(hash(uv+iTime), hash(uv+1.0), hash(uv))-0.5) * kick * 2.0;

    // View Matrix
    vec3 f = normalize(ta - ro);
    vec3 r = normalize(cross(vec3(0,1,0), f));
    vec3 u = cross(f, r);
    vec3 rd = normalize(f + uv.x*r + uv.y*u);
    
    // Raymarching
    float dO = 0.0;
    vec3 p;
    float id = -1.0;
    for(int i=0; i<MAX_STEPS; i++) {
        p = ro + rd * dO;
        vec2 res = map(p);
        dO += res.x;
        id = res.y;
        if(abs(res.x) < SURF_DIST || dO > MAX_DIST) break;
    }
    
    vec3 col = vec3(0.0);
    
    // Distant procedural stars
    col += pow(hash(rd.xy * 100.0), 50.0) * (0.5 + kick);
    
    // Sun Glow (Halo)
    float sunGlow = max(0.0, dot(rd, normalize(vec3(0)-ro)));
    col += vec3(1.0, 0.6, 0.2) * pow(sunGlow, 10.0) * 0.5;

    if(dO < MAX_DIST) {
        vec3 n = getNormal(p);
        vec3 lightDir = normalize(vec3(0.0) - p); // Sunlight always comes from origin
        
        vec3 mate = getMaterial(id, p, n);
        
        if(id == 0.0) {
            col = mate; // Sun is self-illuminated
        } else {
            // Standard diffuse lighting for planets
            float diff = max(0.0, dot(n, lightDir));
            float amb = 0.05;
            col = mate * (diff + amb);
            
            // Atmospheric Rim Light (Fresnel)
            float rim = pow(1.0 - max(0.0, dot(n, -rd)), 4.0);
            col += vec3(0.5, 0.7, 1.0) * rim * (0.5 + kick);
        }
    }
    
    // Gamma Correction
    col = pow(col, vec3(0.4545));
    
    // Audio Visualizer Bar at the bottom
    if (fragCoord.y < 5.0 && fragCoord.x < kick * iResolution.x) col.r += 1.0;

    fragColor = vec4(col, 1.0);
}