// ==========================================================
// NAME : DYNAMIC JULIA ORB LIGHTS
// ==========================================================
// This shader combines a Julia Set fractal with a distance 
// estimation algorithm to create a "light trap" effect. 
// It features 40 dynamic, color-cycling orbs that illuminate 
// the fractal boundaries through inverse-square light falloff.
// ==========================================================
// Credits : Patrick JAILLET
// https://shaderstudio.xo.je
// https://renderforge.ct.ws

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    // 1. COORDINATE NORMALIZATION
    // We transform pixel coordinates so that (0,0) is the center.
    // 'uv0' is preserved for the light orb positions, while 'uv' 
    // is scaled for the fractal view.
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec2 uv0 = uv;
    uv *= 1.5;
    
    // 2. THE JULIA CONSTANT (c)
    // Unlike the Mandelbrot set where 'c' is the pixel position, 
    // in a Julia set, 'c' is a constant for the whole image.
    // We animate 'c' in a complex path using sine and cosine, 
    // or allow the user to control it via iMouse.
    vec2 c = vec2(0.0);
    if (iMouse.z > 0.0) {
        c = (iMouse.xy - 0.5 * iResolution.xy) / iResolution.y * 1.5;
    } else {
        float t = iTime * 0.3;
        // Base orbit for the fractal shape
        c = vec2(0.7885 * cos(t), 0.7885 * sin(t));
        // Add secondary harmonics for organic movement
        c += vec2(sin(t*0.7)*0.1, cos(t*0.5)*0.1);
    }
    
    // 3. FRACTAL ITERATION & DISTANCE ESTIMATION
    // We iterate the function $z_{n+1} = z_n^2 + c$.
    // To get smooth edges, we calculate the derivative 'dz'.
    vec2 z = uv;
    vec2 dz = vec2(1.0, 0.0);
    float r2 = 0.0;
    float iter = 0.0;
    float maxIter = 200.0;
    
    

    for(float i=0.0; i<250.0; i++) {
        if(i > maxIter) break;
        if(r2 > 4.0) break; // Escape radius squared
        
        // Derivative update: dz = 2 * z * dz
        // This is essential for the distance estimation formula.
        dz = 2.0 * vec2(z.x*dz.x - z.y*dz.y, z.x*dz.y + z.y*dz.x);
        
        // Complex squaring: $z^2 = (x^2 - y^2) + i(2xy)$
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        
        r2 = dot(z,z);
        iter++;
    }
    
    // 4. DISTANCE ESTIMATION CALCULATION
    // The formula $d = 0.5 \cdot \frac{\sqrt{r^2} \cdot \log(r^2)}{|dz|}$ 
    // gives us the approximate distance to the fractal boundary.
    
    float d = 0.5 * sqrt(r2) * log(r2) / length(dz);
    d = clamp(d, 0.0, 1.0);

    // 5. LIGHT ORB SYSTEM
    // We create a "swarm" of 40 lights moving in pseudo-random orbits.
    vec3 totalLight = vec3(0.0);
    
    for(float i = 0.0; i < 40.0; i++){
        float t = iTime * (0.2 + i * 0.01);
        float seed = i * 12.345;
        
        // Orb motion: Multi-layered trigonometric functions create 
        // a chaotic, non-repeating path.
        vec2 orbPos = vec2(
            sin(t * 0.8 + seed) * 1.6 + cos(t * 1.2 + seed * 0.5) * 0.4,
            cos(t * 0.9 - seed) * 1.0 + sin(t * 1.5 + seed) * 0.4
        );
        
        // Color Palette: Using a cosine gradient to cycle orb colors.
        vec3 a = vec3(0.5, 0.5, 0.5);
        vec3 b = vec3(0.5, 0.5, 0.5);
        vec3 c_pal = vec3(1.0, 1.0, 1.0);
        vec3 d_pal = vec3(0.263, 0.416, 0.557);
        vec3 orbCol = a + b * cos(6.28318 * (c_pal * (seed * 0.1 + iTime * 0.2) + d_pal));
        
        // Lighting Falloff: Inverse square law simulation.
        // We add a tiny epsilon (0.001) to avoid division by zero.
        
        float distToOrb = length(uv0 - orbPos);
        float lightIntensity = 0.008 / (0.001 + distToOrb * distToOrb);
        
        totalLight += orbCol * lightIntensity;
    }
    
    // 6. FINAL COMPOSITION
    vec3 col = vec3(0.0);
    if (iter < maxIter) {
        // We use the distance estimate 'd' to mask the accumulated light.
        // This traps the light orbs "inside" the fractal structure.
        float fractalIntensity = 1.0 / (1.0 + d * 80.0);
        col = totalLight * fractalIntensity;
    }
    
    // 7. POST-PROCESSING
    // Gamma correction ($1/2.2$) to correct color linear space.
    col = pow(col, vec3(0.4545));
    
    // Subtle vignette based on the original uv0 coordinates.
    col *= 1.0 - length(uv0 * 0.5);
    
    fragColor = vec4(col, 1.0);
}