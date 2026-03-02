#if HW_PERFORMANCE==0
#define AA 1
#else
#define AA 2
#endif

#define ZERO (min(iFrame,0))

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdSphere(vec3 p, float s) {
    return length(p) - s;
}

float map(vec3 p) {
    vec3 q = p;
    
    q.xz *= rot(iTime * 0.1);
    q.yz *= rot(iTime * 0.05);
    
    vec3 id = floor(q / 4.0);
    q = mod(q, 4.0) - 2.0;
    
    float phase = sin(id.x*12. + id.y*3. + id.z*7. + iTime);
    q.xy *= rot(phase * 0.5);
    
    float d1 = sdTorus(q, vec2(1.0, 0.3));
    
    vec3 q2 = q;
    q2.xz *= rot(iTime * 1.5 + phase);
    float d2 = sdSphere(q2 - vec3(1.0, 0.0, 0.0), 0.4);
    
    vec3 q3 = q;
    q3.yz *= rot(iTime * 1.2 + phase);
    float d3 = sdSphere(q3 - vec3(0.0, 1.0, 0.0), 0.4);

    float d = smin(d1, d2, 0.4);
    d = smin(d, d3, 0.4);
    
    d += 0.03 * sin(p.x * 3.0) * sin(p.y * 2.5 + iTime) * sin(p.z * 3.0);
    
    return d * 0.7;
}

float raycast(vec3 ro, vec3 rd) {
    float t = 0.1;
    for (int i = 0; i < 128; i++) {
        float h = map(ro + rd * t);
        if (h < 0.001 * t) return t;
        t += h;
        if (t > 30.0) break;
    }
    return -1.0;
}

vec3 calcNormal(vec3 p) {
    const float h = 0.001; 
    const vec2 k = vec2(1,-1);
    return normalize(k.xyy*map(p + k.xyy*h) + 
                     k.yyx*map(p + k.yyx*h) + 
                     k.yxy*map(p + k.yxy*h) + 
                     k.xxx*map(p + k.xxx*h));
}

vec3 getPastelColor(vec3 p) {
    vec3 colA = vec3(1.0, 0.8, 0.9); 
    vec3 colB = vec3(0.7, 0.9, 1.0);
    vec3 colC = vec3(0.8, 1.0, 0.8); 
    vec3 colD = vec3(1.0, 0.9, 0.7); 
    
    float t = p.y * 0.2 + p.x * 0.1 + iTime * 0.1;
    
    vec3 finalCol = mix(colA, colB, smoothstep(-2.0, 0.0, sin(t)));
    finalCol = mix(finalCol, colC, smoothstep(0.0, 2.0, sin(t + 1.5)));
    finalCol = mix(finalCol, colD, smoothstep(-2.0, 0.0, cos(t * 0.7)));
    
    return finalCol;
}

vec3 render(vec3 ro, vec3 rd) {

    vec3 bgCol = mix(vec3(1.0, 0.95, 0.9), vec3(0.9, 0.95, 1.0), rd.y * 0.5 + 0.5);
    vec3 col = bgCol;
    
    float t = raycast(ro, rd);
    
    if (t > 0.0) {
        vec3 pos = ro + t * rd;
        vec3 nor = calcNormal(pos);
        

        vec3 materialCol = getPastelColor(pos);

        vec3 lig1Dir = normalize(vec3(0.6, 1.0, 0.5));
        float dif1 = clamp(0.5 + 0.5 * dot(nor, lig1Dir), 0.0, 1.0);
        dif1 = pow(dif1, 1.5); 
        

        vec3 lig2Dir = normalize(vec3(-0.8, 0.2, -0.3));
        float dif2 = clamp(0.5 + 0.5 * dot(nor, lig2Dir), 0.0, 1.0);
        
        float ao = smoothstep(-0.5, 1.0, nor.y);

        vec3 lighting = vec3(0.0);
        lighting += dif1 * vec3(1.0, 0.98, 0.95); 
        lighting += dif2 * vec3(0.9, 0.95, 1.0) * 0.7;
        lighting += vec3(0.8, 0.7, 0.9) * 0.3; 
        col = materialCol * lighting * ao;

        vec3 ref = reflect(rd, nor);
        float spe = pow(clamp(dot(ref, lig1Dir), 0.0, 1.0), 8.0);
        col += spe * vec3(1.0, 1.0, 0.9) * 0.4;

        float fogAmount = 1.0 - exp(-0.08 * t);

        vec3 fogCol = mix(bgCol, vec3(1.0, 0.9, 0.95), smoothstep(0.0, 10.0, t) * 0.5);
        col = mix(col, fogCol, fogAmount);
    }
    
    col = pow(col, vec3(0.9)); 
    col = mix(col, vec3(dot(col, vec3(0.33))), 0.1);
    
    return clamp(col, 0.0, 1.0);
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
        
        float camTime = iTime * 0.2;
        vec3 ro = vec3(4.0 * cos(camTime), 2.0 + sin(camTime*0.7), 4.0 * sin(camTime));
        vec3 ta = vec3(0.0, 0.0, 0.0);
        
        vec3 cw = normalize(ta - ro);
        vec3 cp = vec3(0.0, 1.0, 0.0);
        vec3 cu = normalize(cross(cw, cp));
        vec3 cv = cross(cu, cw);
        vec3 rd = normalize(p.x * cu + p.y * cv + 2.5 * cw);

        vec3 col = render(ro, rd);
        col = pow(col, vec3(0.4545));
        
        tot += col;
#if AA > 1
    }
    tot /= float(AA * AA);
#endif

    vec2 q = fragCoord / iResolution.xy;
    tot *= 0.8 + 0.2 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.1);

    fragColor = vec4(tot, 1.0);
}