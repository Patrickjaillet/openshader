// ==========================================================
// NAME : DYNAMIC MAGNETIC DIPOLE FIELD
// ==========================================================
// DESCRIPTION : A physical simulation of a magnetic field with 
// two moving poles. It uses vector math to calculate field 
// strength and generates animated field lines using trigonometric 
// patterns for a classic "iron filing" aesthetic.
// ==========================================================
// Credits : Patrick JAILLET
// https://shaderstudio.xo.je
// https://renderforge.ct.ws

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // 1. Coordinate Setup
    // Normalize pixel coordinates (centered at 0,0, adjusted for aspect ratio)
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    // 2. Pole Animation
    // Define two poles (source and sink) moving in an elliptical orbit
    vec2 pole1 = vec2(sin(iTime * 0.5) * 0.4, cos(iTime * 0.3) * 0.2);
    vec2 pole2 = vec2(-sin(iTime * 0.5) * 0.4, -cos(iTime * 0.3) * 0.2);
    
    // 3. Vector Field Math
    // Calculate the distance vector from the current pixel to each pole
    vec2 r1 = uv - pole1;
    vec2 r2 = uv - pole2;
    
    // Calculate the Magnetic Field Vector (B)
    // The strength follows an inverse power law. Standard physics uses $1/r^2$, 
    // but here we use $r / |r|^{2.4}$ (equivalent to $1/r^{1.4}$) for better visual contrast.
    // The subtraction creates the dipole effect (arrows flowing from one to the other).
    vec2 B = r1 / pow(dot(r1, r1), 1.2) - r2 / pow(dot(r2, r2), 1.2);
    
    // 4. Field Line Generation
    // Calculate the magnitude (strength) of the magnetic field at this point
    float fieldStrength = length(B);
    
    // Create animated concentric rings based on the field strength
    // This simulates the "Field Lines" moving over time
    float texturePattern = sin(fieldStrength * 20.0 - iTime * 2.0);
    float lines = smoothstep(0.6, 0.9, texturePattern);
    
    // 5. Shading and Aesthetics
    // Subtle vignette to darken the corners
    float vignette = 1.0 - dot(uv, uv) * 1.5;
    
    // Vintage "Drafting Paper" color palette
    vec3 backgroundColor = vec3(0.9, 0.86, 0.78);
    vec3 fieldColor = vec3(0.18, 0.12, 0.10);
    
    // Blend colors based on the generated lines
    vec3 finalColor = mix(backgroundColor, fieldColor, lines);
    
    // Mask out the infinite strength at the center of the poles for a cleaner look
    finalColor *= smoothstep(0.0, 0.2, fieldStrength);
    
    // Apply vignette
    finalColor *= clamp(vignette, 0.0, 1.0);
    
    // Final output
    fragColor = vec4(finalColor, 1.0);
}