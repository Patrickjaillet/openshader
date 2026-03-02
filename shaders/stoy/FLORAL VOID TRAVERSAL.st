// ==========================================================
// NAME : FLORAL VOID TRAVERSAL
// ==========================================================
// DESCRIPTION : A 3D raymarching scene where the camera 
// follows a curved path through an infinite field of 
// procedural flowers. Features domain repetition, 
// path-based camera orientation, and atmospheric fog.
// ==========================================================
// Credits : Patrick JAILLET
// https://shaderstudio.xo.je
// https://renderforge.ct.ws

// --- Camera Path Logic ---

// Determines the time/speed of traversal with a slight oscillating "drift".
#define T (iTime*1.5 + 5. + 1.5*sin(iTime*.5))

// Defines the 3D Spline path (horizontal swaying along Z).
#define P(z) (vec3(cos((z) * .07) * 16., 0, (z)))

// Standard 2D rotation matrix.
#define R(a) mat2(cos(a), -sin(a), sin(a), cos(a))
#define N normalize

// --- Procedural Geometry ---

// Creates a single flower using Polar Coordinates and Box/Sphere SDFs.
float flowerField(vec3 p) {
    // Domain Repetition: Splitting space into 1x1x1 cubes.
    vec3 id = floor(p);
    vec3 q = fract(p) - 0.5;

    // Randomize rotation for each flower based on its grid ID.
    q.xy *= R(id.z * 1.5);
    q.xz *= R(id.x * 2.1);

    // 
    float r = length(q.xy);
    float a = atan(q.y, q.x);

    // Petal deformation: A circle radius modulated by a sine wave.
    float petals = 0.25 + 0.1 * sin(a * 6.0);
    // Combine the petal shape with a thin Z-slice to make them flat.
    float dFlower = max(r - petals, abs(q.z) - 0.02);

    // The yellow center of the flower.
    float dCenter = length(q) - 0.12;

    // Blend the parts together.
    return min(dFlower, dCenter) - 0.01;
}

// Global scene map: Combines the flowers and carves out the camera tunnel.
float map(vec3 p) {
    // Carve a hole around the path P(p.z) so the flowers don't hit the camera.
    return max(2. - length((p - P(p.z)).xy), flowerField(p));
}

// --- Main Pipeline ---

void mainImage(out vec4 o, in vec2 u) {
    vec3 r = iResolution;
    // Normalize coordinates.
    u = (u - r.xy / 2.) / r.y;

    // 1. Camera Frame Setup (Frenet-Serret inspired)
    // ro = ray origin (current path position)
    vec3 p = P(T * 2.), ro = p;
    // Z = Forward vector (looking ahead on the path)
    vec3 Z = N(P(T * 2. + 7.) - p);
    // X = Right vector (orthogonal to forward and up)
    vec3 X = N(vec3(Z.z, 0., -Z.x));
    // rd = Ray Direction, including a slight "head tilt" over time.
    vec3 rd = normalize(vec3(R(sin(iTime * .15) * .3) * u, 1.) * mat3(-X, cross(X, Z), Z));

    // 2. Raymarching
    float d = 0., s;
    for (int i = 0; i < 90; i++) {
        p = ro + rd * d;
        s = map(p);
        if (s < 0.001 || d > 60.) break;
        d += s * 0.7; // Safe step multiplier for thin geometry.
    }

    // 3. Shading & Coloring
    if (d < 60.) {
        vec3 id = floor(p);
        // Generate a unique color for each flower based on its grid ID.
        vec3 col = 0.6 + 0.4 * cos(id.x * 1.2 + id.y * 2.3 + id.z * 3.4 + vec3(0, 2, 4));

        // Calculate surface normal for lighting.
        vec2 e = vec2(.01, 0);
        vec3 n = N(vec3(
            map(p + e.xyy) - map(p - e.xyy),
            map(p + e.yxy) - map(p - e.yxy),
            map(p + e.yyx) - map(p - e.yyx)
        ));

        // Basic Diffuse lighting.
        vec3 l = N(vec3(1., 2., -1.));
        float diff = max(dot(n, l), 0.3);
        
        vec3 finalColor = col * diff;
        // Apply distance fog to fade into the light sky.
        o = vec4(mix(vec3(0.8, 0.9, 1.0), finalColor, exp(-0.06 * d)), 1.);
    } else {
        // Sky background color.
        o = vec4(0.8, 0.9, 1., 1.);
    }
}