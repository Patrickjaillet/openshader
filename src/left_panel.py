"""
left_panel.py — Panneau gauche : FX, Files, Info.

13 effets post-processing GLSL intégrés, composables simultanément.
Chaque effet est une fonction GLSL indépendante. FXComposer génère
un shader unique en chaînant tous les effets dont le toggle est ON.
Le signal effect_changed(str|None) déclenche la recompilation de la passe Post.
"""

import os
import sys
import shutil
import subprocess
import webbrowser
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QScrollArea, QDoubleSpinBox, QCheckBox, QSlider,
    QGroupBox, QFormLayout, QFileDialog, QMenu, QSizePolicy, QFrame, QComboBox,
    QInputDialog, QMessageBox, QDialog, QDialogButtonBox, QButtonGroup, QRadioButton,
    QSpinBox, QLineEdit, QToolButton, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData, QUrl, QSize
from PyQt6.QtGui  import QFont, QDrag, QPixmap, QPainter, QColor

from .logger import get_logger
from .ai_param_detector import AIParamDetector, ShaderParam

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Fonctions GLSL composables — chaque effet est une fonction
# apply_FX(vec4 col, vec2 uv, vec2 fragCoord) → vec4
# Le compositeur les chaîne en un seul mainImage.
# ─────────────────────────────────────────────────────────────────────────────

_FX_CHROMATIC = """\
uniform float uChromatic;
vec4 fx_chromatic(vec4 col, vec2 uv, vec2 C) {
    vec2 d = (uv-.5)*uChromatic;
    return vec4(texture(iChannel0,uv+d).r,
                texture(iChannel0,uv  ).g,
                texture(iChannel0,uv-d).b, 1.0);
}"""

_FX_BLOOM = """\
uniform float uBloom;
vec4 fx_bloom(vec4 col, vec2 uv, vec2 C) {
    vec2 px = uBloom*2./iResolution.xy;
    vec4 blur = vec4(0.0);
    for(int x=-3;x<=3;x++) for(int y=-3;y<=3;y++)
        blur += texture(iChannel0, uv+vec2(x,y)*px);
    blur /= 49.0;
    float lum = dot(blur.rgb, vec3(.299,.587,.114));
    return vec4(col.rgb + blur.rgb*lum*uBloom*2., 1.);
}"""

_FX_VIGNETTE = """\
uniform float uVignette;
vec4 fx_vignette(vec4 col, vec2 uv, vec2 C) {
    vec2 p = uv-.5;
    float v = 1.0 - dot(p,p)*uVignette*3.;
    return vec4(col.rgb*clamp(v,0.,1.), 1.);
}"""

_FX_BLUR = """\
uniform float uBlurRadius;
vec4 fx_blur(vec4 col, vec2 uv, vec2 C) {
    if(uBlurRadius<.01) return col;
    vec4 acc=vec4(0.); float w=0.;
    vec2 px = uBlurRadius/iResolution.xy;
    for(int x=-3;x<=3;x++) for(int y=-3;y<=3;y++){
        float wt=exp(-.5*float(x*x+y*y));
        acc+=texture(iChannel0,uv+vec2(x,y)*px)*wt; w+=wt;
    }
    return acc/w;
}"""

_FX_GLITCH = """\
uniform float uGlitch;
float _glitch_h(float v){return fract(sin(v)*43758.5453);}
vec4 fx_glitch(vec4 col, vec2 uv, vec2 C) {
    float s = step(.97-uGlitch*.5, _glitch_h(floor(uv.y*150.)+iTime*7.));
    float sh = s*uGlitch*.05*sin(iTime*30.+uv.y*50.);
    return vec4(texture(iChannel0,vec2(uv.x+sh,uv.y)).r,
                col.g,
                texture(iChannel0,vec2(uv.x-sh,uv.y)).b, 1.);
}"""

_FX_CRT = """\
uniform float uScanlines;
uniform float uCurvature;
vec4 fx_crt(vec4 col, vec2 uv, vec2 C) {
    vec2 cc = uv-.5;
    vec2 cuv = uv + cc*dot(cc,cc)*uCurvature*4.;
    if(cuv.x<0.||cuv.x>1.||cuv.y<0.||cuv.y>1.) return vec4(0.,0.,0.,1.);
    vec4 c2 = texture(iChannel0, cuv);
    float sc = 1.-uScanlines*.45*(1.-abs(sin(cuv.y*iResolution.y*3.14159)));
    return vec4(c2.rgb*sc, 1.);
}"""

_FX_GRAIN = """\
uniform float uGrain;
float _grain_rand(vec2 p){return fract(sin(dot(p,vec2(12.989,78.233)))*43758.545);}
vec4 fx_grain(vec4 col, vec2 uv, vec2 C) {
    float g = _grain_rand(uv+fract(iTime))-.5;
    return vec4(clamp(col.rgb+g*uGrain*.2, 0., 1.), 1.);
}"""

_FX_COLOR = """\
uniform float uSaturation;
uniform float uContrast;
uniform float uBrightness;
vec4 fx_color(vec4 col, vec2 uv, vec2 C) {
    vec3 c = col.rgb + uBrightness;
    c = (c-.5)*uContrast+.5;
    float lum = dot(c, vec3(.299,.587,.114));
    c = mix(vec3(lum), c, uSaturation);
    return vec4(clamp(c,0.,1.), 1.);
}"""

_FX_PIXEL = """\
uniform float uPixelSize;
vec4 fx_pixel(vec4 col, vec2 uv, vec2 C) {
    float s = max(1., uPixelSize);
    vec2 puv = (floor(C/s)*s+.5*s)/iResolution.xy;
    return texture(iChannel0, clamp(puv,0.,1.));
}"""

_FX_RETRO = """\
uniform float uColors;
vec4 fx_retro(vec4 col, vec2 uv, vec2 C) {
    float n = max(2., uColors);
    return vec4(floor(col.rgb*n)/(n-1.), 1.);
}"""

_FX_KALEIDO = """\
uniform float uKaleido;
vec4 fx_kaleido(vec4 col, vec2 uv, vec2 C) {
    vec2 kuv = uv-.5;
    kuv.x *= iResolution.x/iResolution.y;
    float r=length(kuv), a=atan(kuv.y,kuv.x);
    float n=max(2.,uKaleido), seg=3.14159/n;
    a=mod(a,2.*seg); a=abs(a-seg);
    vec2 nuv=vec2(cos(a),sin(a))*r;
    nuv.x /= iResolution.x/iResolution.y;
    return texture(iChannel0, nuv+.5);
}"""

_FX_MIRROR = """\
uniform float uMirrorX;
uniform float uMirrorY;
vec4 fx_mirror(vec4 col, vec2 uv, vec2 C) {
    vec2 muv = uv;
    if(uMirrorX>.5) muv.x = muv.x<.5 ? muv.x*2. : (1.-muv.x)*2.;
    if(uMirrorY>.5) muv.y = muv.y<.5 ? muv.y*2. : (1.-muv.y)*2.;
    return texture(iChannel0, clamp(muv,0.,1.));
}"""

_FX_HUESHIFT = """\
uniform float uHueShift;
vec3 _rgb2hsv(vec3 c){
    vec4 K=vec4(0.,-1./3.,2./3.,-1.);
    vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));
    vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));
    float d=q.x-min(q.w,q.y); float e=1.e-10;
    return vec3(abs(q.z+(q.w-q.y)/(6.*d+e)),d/(q.x+e),q.x);
}
vec3 _hsv2rgb(vec3 c){
    vec4 K=vec4(1.,2./3.,1./3.,3.);
    vec3 p=abs(fract(c.xxx+K.xyz)*6.-K.www);
    return c.z*mix(K.xxx,clamp(p-K.xxx,0.,1.),c.y);
}
vec4 fx_hueshift(vec4 col, vec2 uv, vec2 C) {
    vec3 hsv=_rgb2hsv(col.rgb);
    hsv.x=fract(hsv.x+uHueShift);
    return vec4(_hsv2rgb(hsv),1.);
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Nouveaux FX v2.1 (+20 effets)
# ─────────────────────────────────────────────────────────────────────────────

_FX_SHARPEN = """\
uniform float uSharpen;
vec4 fx_sharpen(vec4 col, vec2 uv, vec2 C) {
    vec2 px = 1.0/iResolution.xy;
    vec4 nb = ( texture(iChannel0,uv+vec2( px.x,0.))+texture(iChannel0,uv+vec2(-px.x,0.))
               +texture(iChannel0,uv+vec2(0., px.y))+texture(iChannel0,uv+vec2(0.,-px.y)) );
    return vec4(clamp(col.rgb*(1.+uSharpen*4.) - nb.rgb*uSharpen, 0.,1.), 1.);
}"""

_FX_SOBEL = """\
uniform float uEdge;
vec4 fx_sobel(vec4 col, vec2 uv, vec2 C) {
    vec2 px = uEdge/iResolution.xy;
    float tl=dot(texture(iChannel0,uv+vec2(-px.x, px.y)).rgb,vec3(.33));
    float tm=dot(texture(iChannel0,uv+vec2(    0., px.y)).rgb,vec3(.33));
    float tr=dot(texture(iChannel0,uv+vec2( px.x, px.y)).rgb,vec3(.33));
    float ml=dot(texture(iChannel0,uv+vec2(-px.x,    0.)).rgb,vec3(.33));
    float mr=dot(texture(iChannel0,uv+vec2( px.x,    0.)).rgb,vec3(.33));
    float bl=dot(texture(iChannel0,uv+vec2(-px.x,-px.y)).rgb,vec3(.33));
    float bm=dot(texture(iChannel0,uv+vec2(    0.,-px.y)).rgb,vec3(.33));
    float br=dot(texture(iChannel0,uv+vec2( px.x,-px.y)).rgb,vec3(.33));
    float gx = -tl-2.*ml-bl+tr+2.*mr+br;
    float gy = -tl-2.*tm-tr+bl+2.*bm+br;
    float e = clamp(length(vec2(gx,gy))*2.,0.,1.);
    return vec4(mix(col.rgb, vec3(e), uEdge), 1.);
}"""

_FX_POSTERIZE = """\
uniform float uPosterize;
vec4 fx_posterize(vec4 col, vec2 uv, vec2 C) {
    float n = max(2., uPosterize);
    vec3 c = floor(col.rgb * n + .5) / n;
    return vec4(c, 1.);
}"""

_FX_DUOTONE = """\
uniform float uDuoR1; uniform float uDuoG1; uniform float uDuoB1;
uniform float uDuoR2; uniform float uDuoG2; uniform float uDuoB2;
vec4 fx_duotone(vec4 col, vec2 uv, vec2 C) {
    float lum = dot(col.rgb, vec3(.299,.587,.114));
    vec3 dark  = vec3(uDuoR1,uDuoG1,uDuoB1);
    vec3 light = vec3(uDuoR2,uDuoG2,uDuoB2);
    return vec4(mix(dark, light, lum), 1.);
}"""

_FX_NEON = """\
uniform float uNeon;
vec4 fx_neon(vec4 col, vec2 uv, vec2 C) {
    vec3 c = col.rgb;
    float lum = dot(c, vec3(.299,.587,.114));
    vec3 edge = c - lum;
    vec3 sat = c + edge * uNeon * 2.;
    float bloom = pow(max(lum-.5,0.)*2., 2.) * uNeon;
    return vec4(clamp(sat + sat*bloom, 0.,1.), 1.);
}"""

_FX_THERMAL = """\
uniform float uThermal;
vec4 fx_thermal(vec4 col, vec2 uv, vec2 C) {
    float t = dot(col.rgb, vec3(.299,.587,.114));
    t = mix(t, t*t, uThermal);
    vec3 cold  = vec3(0.,.1,.8);
    vec3 warm  = vec3(.1,.9,.2);
    vec3 hot   = vec3(1.,.3,.0);
    vec3 white = vec3(1.,1.,1.);
    vec3 out_c = t < .33 ? mix(cold,warm,t*3.)
               : t < .66 ? mix(warm,hot,(t-.33)*3.)
               :            mix(hot,white,(t-.66)*3.);
    return vec4(mix(col.rgb, out_c, uThermal), 1.);
}"""

_FX_OLDFILM = """\
uniform float uOldFilm;
float _of_rand(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
vec4 fx_oldfilm(vec4 col, vec2 uv, vec2 C) {
    // Sépia
    vec3 s = vec3(dot(col.rgb,vec3(.393,.769,.189)),
                  dot(col.rgb,vec3(.349,.686,.168)),
                  dot(col.rgb,vec3(.272,.534,.131)));
    // Scratches verticales
    float t = floor(iTime*12.)*1000.;
    float scratch = step(.98, _of_rand(vec2(floor(uv.x*200.+t),1.)));
    float scratchV = scratch * step(fract(uv.y*80.+t*0.7),.6)*(1.-uv.y*.3);
    // Bruit temporel
    float noise = (_of_rand(uv+fract(iTime*7.))-0.5)*0.12;
    // Flicker
    float flicker = 1.-.05*_of_rand(vec2(floor(iTime*15.)));
    vec3 out_c = clamp(mix(col.rgb, s, uOldFilm)*flicker + noise*uOldFilm + scratchV*uOldFilm, 0.,1.);
    return vec4(out_c, 1.);
}"""

_FX_HALFTONE = """\
uniform float uHalftone;
vec4 fx_halftone(vec4 col, vec2 uv, vec2 C) {
    float s = max(2., uHalftone);
    vec2 cell = floor(C/s);
    vec2 center = (cell*s + s*.5)/iResolution.xy;
    float avg = dot(texture(iChannel0,center).rgb, vec3(.299,.587,.114));
    vec2 local = (C - cell*s - s*.5)/(s*.5);
    float dot_ = step(length(local), sqrt(avg)*1.05);
    return vec4(vec3(dot_), 1.);
}"""

_FX_OILPAINT = """\
uniform float uOilRadius;
vec4 fx_oilpaint(vec4 col, vec2 uv, vec2 C) {
    int r = max(1, int(uOilRadius));
    vec4 best = vec4(0.); float bestV = -1.;
    vec2 px = 1./iResolution.xy;
    for(int x=-r;x<=r;x++) for(int y=-r;y<=r;y++){
        vec2 off = vec2(float(x),float(y))*px;
        vec4 s = texture(iChannel0, uv+off);
        float v = dot(s.rgb,vec3(.299,.587,.114));
        float w = float(x*x+y*y) <= float(r*r) ? 1.0 : 0.0;
        if(w>.5 && v>bestV){bestV=v; best=s;}
    }
    return mix(col, best, .75);
}"""

_FX_FISHEYE = """\
uniform float uFisheye;
vec4 fx_fisheye(vec4 col, vec2 uv, vec2 C) {
    vec2 p = uv-.5;
    p.x *= iResolution.x/iResolution.y;
    float r = length(p);
    float f = 1. + uFisheye*r*r;
    p /= f;
    p.x /= iResolution.x/iResolution.y;
    p += .5;
    if(p.x<0.||p.x>1.||p.y<0.||p.y>1.) return vec4(0.,0.,0.,1.);
    return texture(iChannel0, p);
}"""

_FX_RGBSPLIT = """\
uniform float uRGBSplit;
uniform float uRGBAngle;
vec4 fx_rgbsplit(vec4 col, vec2 uv, vec2 C) {
    float a = uRGBAngle * 6.2832;
    vec2 d = vec2(cos(a),sin(a)) * uRGBSplit;
    float r = texture(iChannel0, uv + d).r;
    float g = texture(iChannel0, uv).g;
    float b = texture(iChannel0, uv - d).b;
    return vec4(r,g,b,1.);
}"""

_FX_WARP = """\
uniform float uWarp;
uniform float uWarpFreq;
vec4 fx_warp(vec4 col, vec2 uv, vec2 C) {
    float f = max(.1, uWarpFreq);
    vec2 off = vec2(sin(uv.y*f*6.28+iTime*2.),
                    cos(uv.x*f*6.28+iTime*1.5)) * uWarp * .02;
    return texture(iChannel0, clamp(uv+off, 0.,1.));
}"""

_FX_ZOOM = """\
uniform float uZoom;
uniform float uZoomX;
uniform float uZoomY;
vec4 fx_zoom(vec4 col, vec2 uv, vec2 C) {
    vec2 center = vec2(uZoomX, uZoomY);
    vec2 zuv = center + (uv-center) / max(.05, uZoom);
    if(zuv.x<0.||zuv.x>1.||zuv.y<0.||zuv.y>1.) return vec4(0.,0.,0.,1.);
    return texture(iChannel0, zuv);
}"""

_FX_TILT_SHIFT = """\
uniform float uTiltFocus;
uniform float uTiltBlur;
vec4 fx_tiltshift(vec4 col, vec2 uv, vec2 C) {
    float dist = abs(uv.y - uTiltFocus);
    float blur = smoothstep(0., uTiltBlur, dist - .1);
    if(blur < .01) return col;
    vec4 acc = vec4(0.); float w = 0.;
    vec2 px = blur * 4. / iResolution.xy;
    for(int i=-4;i<=4;i++) for(int j=-4;j<=4;j++){
        float wt = exp(-.5*float(i*i+j*j));
        acc += texture(iChannel0, uv+vec2(i,j)*px)*wt; w+=wt;
    }
    return acc/w;
}"""

_FX_DITHERING = """\
uniform float uDither;
float _bayer(vec2 pos){
    int x=int(mod(pos.x,4.)); int y=int(mod(pos.y,4.));
    int idx = x + y*4;
    float bayer[16];
    bayer[0]=0.;bayer[1]=8.;bayer[2]=2.;bayer[3]=10.;
    bayer[4]=12.;bayer[5]=4.;bayer[6]=14.;bayer[7]=6.;
    bayer[8]=3.;bayer[9]=11.;bayer[10]=1.;bayer[11]=9.;
    bayer[12]=15.;bayer[13]=7.;bayer[14]=13.;bayer[15]=5.;
    return bayer[idx]/16.;
}
vec4 fx_dithering(vec4 col, vec2 uv, vec2 C) {
    float threshold = _bayer(C);
    float n = max(2., uDither);
    vec3 q = floor(col.rgb * n + threshold) / (n-1.);
    return vec4(clamp(q,0.,1.), 1.);
}"""

_FX_RECOLOR = """\
uniform float uRecolorHue;
uniform float uRecolorSat;
uniform float uRecolorVal;
vec3 _rc_hsv2rgb(vec3 c){
    vec4 K=vec4(1.,2./3.,1./3.,3.);
    return c.z*mix(K.xxx, clamp(abs(fract(c.x+K.xyz)*6.-K.www)-K.xxx,0.,1.),c.y);
}
vec3 _rc_rgb2hsv(vec3 c){
    float mx=max(c.r,max(c.g,c.b)),mn=min(c.r,min(c.g,c.b));
    float d=mx-mn; float h=0.;
    if(d>.0001){
        if(mx==c.r) h=mod((c.g-c.b)/d,6.);
        else if(mx==c.g) h=(c.b-c.r)/d+2.;
        else h=(c.r-c.g)/d+4.;
        h/=6.;
    }
    return vec3(h, mx>.0001?d/mx:0., mx);
}
vec4 fx_recolor(vec4 col, vec2 uv, vec2 C) {
    vec3 hsv = _rc_rgb2hsv(col.rgb);
    hsv.x = fract(hsv.x + uRecolorHue);
    hsv.y = clamp(hsv.y * uRecolorSat, 0.,1.);
    hsv.z = clamp(hsv.z * uRecolorVal, 0.,1.);
    return vec4(_rc_hsv2rgb(hsv), 1.);
}"""

_FX_WAVE = """\
uniform float uWaveAmp;
uniform float uWaveFreq;
uniform float uWaveAxis;
vec4 fx_wave(vec4 col, vec2 uv, vec2 C) {
    float f = max(.1, uWaveFreq) * 6.2832;
    vec2 off = vec2(0.);
    if(uWaveAxis < .5)
        off.x = sin(uv.y*f + iTime*3.) * uWaveAmp * .03;
    else
        off.y = sin(uv.x*f + iTime*3.) * uWaveAmp * .03;
    return texture(iChannel0, clamp(uv+off,0.,1.));
}"""

_FX_TUNNEL = """\
uniform float uTunnelSpeed;
uniform float uTunnelZoom;
vec4 fx_tunnel(vec4 col, vec2 uv, vec2 C) {
    vec2 p = uv - .5;
    p.x *= iResolution.x / iResolution.y;
    float r = length(p);
    float a = atan(p.y, p.x);
    float z = 1. / (r + .01);
    vec2 tuv = vec2(a / 6.2832 + .5,
                    fract(z * uTunnelZoom - iTime * uTunnelSpeed * .5));
    return texture(iChannel0, tuv);
}"""

_FX_RELIEF = """\
uniform float uRelief;
vec4 fx_relief(vec4 col, vec2 uv, vec2 C) {
    vec2 px = uRelief / iResolution.xy;
    float tl = dot(texture(iChannel0,uv+vec2(-px.x, px.y)).rgb,vec3(.33));
    float br = dot(texture(iChannel0,uv+vec2( px.x,-px.y)).rgb,vec3(.33));
    float emb = clamp((tl-br+1.)*.5, 0.,1.);
    return vec4(vec3(emb), 1.);
}"""



# ─────────────────────────────────────────────────────────────────────────────
# Nouveaux FX v3.0 (+20 effets)
# ─────────────────────────────────────────────────────────────────────────────

_FX_ASCII = """uniform float uAsciiSize;
float _asc_lum(vec2 u){return dot(texture(iChannel0,u).rgb,vec3(.299,.587,.114));}
vec4 fx_ascii(vec4 col, vec2 uv, vec2 C) {
    float s = max(4., uAsciiSize);
    vec2 cell = floor(C/s)*s;
    float avg = _asc_lum(cell/iResolution.xy);
    // 10 niveaux de densité ASCII via grille de points
    float lvl = floor(avg*9.9);
    vec2 loc = fract(C/s)*s;
    float p = 0.;
    if(lvl>=1.) p=max(p, step(length(loc-vec2(s*.5))-s*.15,0.));
    if(lvl>=3.) p=max(p, step(abs(loc.x-s*.5)-s*.08,0.)*step(abs(loc.y-s*.5)-s*.35,0.));
    if(lvl>=5.) p=max(p, step(abs(loc.x-s*.5)-s*.35,0.)*step(abs(loc.y-s*.5)-s*.08,0.));
    if(lvl>=7.) p=max(p, step(abs(loc.x-loc.y)-s*.1,0.)*step(length(loc-vec2(s*.5))-s*.4,0.));
    if(lvl>=9.) p=1.;
    return vec4(vec3(p)*col.rgb*2., 1.);
}"""

_FX_WATERCOLOR = """uniform float uWatercolor;
uniform float uWaterEdge;
float _wc_hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
float _wc_noise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=_wc_hash(i),b=_wc_hash(i+vec2(1,0)),c=_wc_hash(i+vec2(0,1)),d=_wc_hash(i+vec2(1,1));
    vec2 u=f*f*(3.-2.*f);
    return mix(mix(a,b,u.x),mix(c,d,u.x),u.y);
}
vec4 fx_watercolor(vec4 col, vec2 uv, vec2 C) {
    vec2 n1 = vec2(_wc_noise(uv*8.+iTime*.1), _wc_noise(uv*8.+vec2(5.2,1.3)+iTime*.1));
    vec2 wuv = uv + n1 * uWatercolor * .025;
    vec4 s = texture(iChannel0, clamp(wuv,0.,1.));
    // Bords sombres (aquarelle sèche)
    vec2 px = uWaterEdge/iResolution.xy;
    float edge = 0.;
    for(int i=-1;i<=1;i++) for(int j=-1;j<=1;j++){
        vec2 d=vec2(float(i),float(j))*px;
        edge += length(texture(iChannel0,uv+d).rgb - s.rgb);
    }
    edge = clamp(edge * uWaterEdge * 0.5, 0., 1.);
    return vec4(mix(s.rgb, s.rgb*0.3, edge*0.5), 1.);
}"""

_FX_CROSSHATCH = """uniform float uHatchDensity;
uniform float uHatchAngle;
vec4 fx_crosshatch(vec4 col, vec2 uv, vec2 C) {
    float lum = dot(col.rgb, vec3(.299,.587,.114));
    float a = uHatchAngle * 3.14159;
    float d = max(2., uHatchDensity);
    // 1ère famille de hachures
    float h1 = fract((C.x*cos(a) - C.y*sin(a)) / d);
    float h2 = fract((C.x*cos(a+1.5708) - C.y*sin(a+1.5708)) / d);
    float line1 = step(1.-lum, h1);
    float line2 = step(1.-lum*0.6, h2);
    float ink = min(line1*line2, 1.);
    return vec4(vec3(ink), 1.);
}"""

_FX_HOLOGRAM = """uniform float uHoloGlitch;
uniform float uHoloColor;
float _holo_rand(float x){return fract(sin(x*127.1)*43758.5);}
vec4 fx_hologram(vec4 col, vec2 uv, vec2 C) {
    // Scanlines holographiques
    float scan = abs(sin(uv.y * 200. + iTime*4.)) * .3 + .7;
    // Glitch horizontal aléatoire par bandes
    float band = floor(uv.y * 20. + iTime*3.);
    float glitch = (_holo_rand(band) - .5) * uHoloGlitch * .05;
    vec4 s = texture(iChannel0, vec2(uv.x + glitch, uv.y));
    // Teinte holographique (cyan/vert)
    float hue = fract(uv.y * 0.3 + iTime * .1 + uHoloColor);
    vec3 hcol = .5 + .5*cos(6.2832*(hue+vec3(0.,.33,.67)));
    float lum = dot(s.rgb, vec3(.299,.587,.114));
    vec3 out_c = mix(s.rgb, hcol * lum * 2., .45) * scan;
    return vec4(clamp(out_c,0.,1.), 1.);
}"""

_FX_FROSTED = """uniform float uFrost;
uniform float uFrostScale;
float _fr_hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
vec2 _fr_noise2(vec2 p){
    return vec2(_fr_hash(p),_fr_hash(p+vec2(17.4,3.1)));
}
vec4 fx_frosted(vec4 col, vec2 uv, vec2 C) {
    float sc = max(.5, uFrostScale);
    vec2 cell = floor(uv * iResolution.xy / sc);
    vec2 jitter = (_fr_noise2(cell) - .5) * uFrost * .04;
    return texture(iChannel0, clamp(uv + jitter, 0., 1.));
}"""

_FX_LIQUID = """uniform float uLiquid;
uniform float uLiquidSpeed;
vec4 fx_liquid(vec4 col, vec2 uv, vec2 C) {
    float t = iTime * uLiquidSpeed;
    vec2 p = uv * 6.;
    float n = sin(p.x + sin(p.y + t)) + sin(p.y * 1.3 + sin(p.x * 1.1 + t*1.2));
    float n2 = sin(p.x * .7 + p.y * .9 + t * .8);
    vec2 off = vec2(n, n2) * uLiquid * .015;
    return texture(iChannel0, clamp(uv + off, 0., 1.));
}"""

_FX_STAINED = """uniform float uStainedLight;
uniform float uStainedSat;
float _st_hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
vec4 fx_stained(vec4 col, vec2 uv, vec2 C) {
    // Vitrail : segments de Voronoï colorés
    float minD = 1e9; vec2 minP = vec2(0.);
    for(int i=0;i<9;i++){
        vec2 grid = vec2(float(i/3)-1., float(mod(float(i),3.))-1.);
        vec2 cell = floor(uv*4.) + grid;
        vec2 pt = cell + vec2(_st_hash(cell), _st_hash(cell+vec2(1.,0.)));
        float d = length(uv*4. - pt);
        if(d < minD){minD=d; minP=cell;}
    }
    // Couleur de la cellule
    vec3 hue = .5+.5*cos(6.2832*(vec3(_st_hash(minP),_st_hash(minP+1.),_st_hash(minP+2.))+vec3(0.,.33,.67)));
    float lum = dot(col.rgb, vec3(.299,.587,.114));
    vec3 glass = hue * lum * uStainedLight;
    glass = mix(glass, glass*vec3(uStainedSat,1.,1./uStainedSat), .3);
    // Joints de plomb
    float lead = smoothstep(.06, .12, minD);
    return vec4(glass * lead, 1.);
}"""

_FX_PIXELSORT = """uniform float uSortThresh;
uniform float uSortDir;
vec4 fx_pixelsort(vec4 col, vec2 uv, vec2 C) {
    float lum = dot(col.rgb, vec3(.299,.587,.114));
    // Trie pixels dans la direction choisie si sous le seuil
    float triggered = step(uSortThresh, lum);
    vec2 dir = uSortDir < .5 ? vec2(0.,1.) : vec2(1.,0.);
    vec2 step_ = dir / iResolution.xy;
    // Accumulation vers une position "sorted"
    vec4 acc = col;
    for(int i=1;i<=8;i++){
        vec2 off = dir * float(i);
        vec4 s = texture(iChannel0, uv + off*step_*4.);
        float sl = dot(s.rgb, vec3(.299,.587,.114));
        if(sl > lum && triggered > .5) acc = s;
    }
    return mix(col, acc, triggered);
}"""

_FX_AURA = """uniform float uAuraRadius;
uniform float uAuraIntensity;
vec4 fx_aura(vec4 col, vec2 uv, vec2 C) {
    // Auréole colorée autour des zones lumineuses
    float lum = dot(col.rgb, vec3(.299,.587,.114));
    vec4 blur = vec4(0.); float w = 0.;
    vec2 px = uAuraRadius / iResolution.xy;
    for(int i=-5;i<=5;i++) for(int j=-5;j<=5;j++){
        float wt = exp(-.5*float(i*i+j*j)/9.);
        blur += texture(iChannel0, uv+vec2(float(i),float(j))*px)*wt; w+=wt;
    }
    blur /= w;
    float blum = dot(blur.rgb, vec3(.299,.587,.114));
    // Halo coloré basé sur la teinte locale
    vec3 hue = normalize(blur.rgb + .001) * blum;
    vec3 aura = hue * uAuraIntensity * smoothstep(.3, .9, blum);
    return vec4(clamp(col.rgb + aura, 0., 1.), 1.);
}"""

_FX_DATAMOSH = """uniform float uDatamosh;
float _dm_hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
vec4 fx_datamosh(vec4 col, vec2 uv, vec2 C) {
    // Blocs de macroblocs déplacés (compression artefact)
    float bsize = 16.;
    vec2 block = floor(C/bsize);
    float seed = floor(iTime*8.) + _dm_hash(block)*100.;
    float active = step(1.-uDatamosh, _dm_hash(vec2(seed, seed*.7)));
    vec2 motion = (_dm_hash(vec2(seed,1.))-0.5, _dm_hash(vec2(seed,2.))-0.5) * uDatamosh * 40.;
    vec2 displaced = uv + motion / iResolution.xy * active;
    return mix(col, texture(iChannel0, clamp(displaced,0.,1.)), active*uDatamosh);
}"""

_FX_VORONOI_COL = """uniform float uVoronoiScale;
uniform float uVoronoiMix;
float _vc_hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
vec4 fx_voronoi_col(vec4 col, vec2 uv, vec2 C) {
    float sc = max(.5, uVoronoiScale);
    vec2 p = uv * sc;
    vec2 cell = floor(p);
    float minD = 1e9; vec2 minCell = cell;
    for(int i=-1;i<=1;i++) for(int j=-1;j<=1;j++){
        vec2 nb = cell + vec2(float(i),float(j));
        vec2 pt = nb + vec2(_vc_hash(nb), _vc_hash(nb+13.7));
        float d = length(p - pt);
        if(d < minD){minD=d; minCell=nb;}
    }
    // Couleur de la cellule basée sur la texture originale en son centre
    vec2 cUV = (minCell + vec2(_vc_hash(minCell), _vc_hash(minCell+13.7))) / sc;
    vec4 cCol = texture(iChannel0, clamp(cUV, 0., 1.));
    // Bordures
    float edge = smoothstep(.03, .06, minD);
    return mix(cCol * edge, col, 1. - uVoronoiMix);
}"""

_FX_SCANLINE_COLOR = """uniform float uScanColor;
uniform float uScanFreq;
vec4 fx_scanline_color(vec4 col, vec2 uv, vec2 C) {
    float freq = max(1., uScanFreq);
    // Lignes R/G/B alternées (écran LCD simulé)
    int px = int(mod(C.x, 3.));
    vec3 mask = px==0 ? vec3(1.,0.,0.) : px==1 ? vec3(0.,1.,0.) : vec3(0.,0.,1.);
    float scan = abs(sin(C.y * 3.14159 / freq)) * .4 + .6;
    vec3 out_c = mix(col.rgb, col.rgb * mask * 1.5, uScanColor) * scan;
    return vec4(clamp(out_c, 0., 1.), 1.);
}"""

_FX_DISSOLVE = """uniform float uDissolve;
uniform float uDissolveScale;
float _dis_hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
float _dis_noise(vec2 p){
    vec2 i=floor(p),f=fract(p),u=f*f*(3.-2.*f);
    return mix(mix(_dis_hash(i),_dis_hash(i+vec2(1,0)),u.x),
               mix(_dis_hash(i+vec2(0,1)),_dis_hash(i+vec2(1,1)),u.x),u.y);
}
vec4 fx_dissolve(vec4 col, vec2 uv, vec2 C) {
    float sc = max(.5, uDissolveScale);
    float n = _dis_noise(uv * sc * 8.);
    float alpha = step(n, 1. - uDissolve);
    return mix(vec4(0.,0.,0.,0.), col, alpha);
}"""

_FX_RETROWAVE = """uniform float uRetroGrid;
uniform float uRetroGlow;
vec4 fx_retrowave(vec4 col, vec2 uv, vec2 C) {
    // Grille perspective néon + teinte synthwave
    vec2 p = uv - vec2(.5,.4);
    p.x *= iResolution.x/iResolution.y;
    float z = p.y + .001;
    vec2 grid = vec2(fract(p.x/(z+.001)*uRetroGrid + iTime*.5),
                     fract(1./(z+.001)*uRetroGrid*.5 - iTime*.3));
    float line = step(.93, max(grid.x, grid.y));
    // Dégradé de couleur synthwave
    vec3 synth = mix(vec3(.7,0.,1.), vec3(0.,.8,1.), uv.y);
    float glow = uRetroGlow * line;
    vec3 out_c = col.rgb + synth * glow * (1.-uv.y*uv.y);
    return vec4(clamp(out_c, 0., 1.), 1.);
}"""

_FX_SPARKLE = """uniform float uSparkle;
uniform float uSparkleDensity;
float _sp_hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
vec4 fx_sparkle(vec4 col, vec2 uv, vec2 C) {
    float lum = dot(col.rgb, vec3(.299,.587,.114));
    // Scintillements sur les hautes lumières
    float density = max(4., uSparkleDensity);
    vec2 cell = floor(C / density);
    float t = floor(iTime * 12.);
    float r = _sp_hash(cell + t);
    float bright = _sp_hash(cell + t + 7.3);
    // Croix de scintillement
    vec2 local = fract(C/density) - .5;
    float cross_ = max(step(abs(local.x),.06)*step(abs(local.y),.45),
                       step(abs(local.y),.06)*step(abs(local.x),.45));
    float sparkle = step(1.-uSparkle, r) * cross_ * bright * smoothstep(.5,1.,lum);
    return vec4(clamp(col.rgb + sparkle*vec3(1.,.95,.8)*2., 0., 1.), 1.);
}"""

_FX_MIRROR_KALEID = """uniform float uMKSegs;
uniform float uMKSpin;
vec4 fx_mirror_kaleid(vec4 col, vec2 uv, vec2 C) {
    vec2 p = uv - .5;
    p.x *= iResolution.x/iResolution.y;
    float angle = atan(p.y, p.x) + iTime * uMKSpin * .2;
    float r = length(p);
    float segs = max(2., uMKSegs);
    float seg = 6.2832 / segs;
    float a = mod(angle, seg);
    if(a > seg*.5) a = seg - a;
    // Répétition octogonale
    a = mod(a, seg*.5);
    if(a > seg*.25) a = seg*.5 - a;
    vec2 nuv = vec2(cos(a), sin(a)) * r / (iResolution.x/iResolution.y);
    nuv += .5;
    return texture(iChannel0, clamp(nuv, 0., 1.));
}"""

_FX_AURORA = """uniform float uAuroraSpeed;
uniform float uAuroraWave;
vec4 fx_aurora(vec4 col, vec2 uv, vec2 C) {
    float t = iTime * uAuroraSpeed;
    // Couches de aurora borealis
    float a = 0.;
    for(float i=1.;i<=4.;i++){
        float wave = sin(uv.x*uAuroraWave*i*1.3 + t*i*.7 + i*2.1) * .5 + .5;
        float band = exp(-pow((uv.y - .3 - wave*.4)*4., 2.));
        a += band / i;
    }
    vec3 aurCol = mix(vec3(0.,.8,.4), vec3(0.,.4,.9), sin(t*.3)*.5+.5);
    aurCol = mix(aurCol, vec3(.8,.2,1.), sin(t*.5+1.)*.3+.3);
    return vec4(clamp(col.rgb + aurCol*a*.8, 0.,1.), 1.);
}"""

_FX_CONVOLVE_EMBOSS = """uniform float uEmbossAngle;
uniform float uEmbossStrength;
vec4 fx_emboss_dir(vec4 col, vec2 uv, vec2 C) {
    float a = uEmbossAngle * 6.2832;
    vec2 d = vec2(cos(a),sin(a)) * uEmbossStrength / iResolution.xy;
    vec4 s1 = texture(iChannel0, uv + d);
    vec4 s2 = texture(iChannel0, uv - d);
    float emb = dot(s1.rgb - s2.rgb, vec3(.33)) * .5 + .5;
    // Conserve la couleur originale teintée par le relief
    float lum = dot(col.rgb, vec3(.299,.587,.114));
    return vec4(col.rgb * emb * 1.5, 1.);
}"""

_FX_INFRARED = """uniform float uInfrared;
vec4 fx_infrared(vec4 col, vec2 uv, vec2 C) {
    // Simulation infrarouge : végétation blanche, ciel sombre
    float r = col.r; float g = col.g; float b = col.b;
    // Végétation (vert dominant) → blanc; ciel (bleu) → noir
    float foliage = smoothstep(.1, .5, g - max(r,b));
    float sky     = smoothstep(.1, .5, b - max(r,g));
    float ir = mix(dot(col.rgb, vec3(.299,.587,.114)), 1., foliage);
    ir = mix(ir, 0., sky*.5);
    // Grain argentique infrarouge
    float grain = fract(sin(dot(uv+fract(iTime),vec2(127.1,311.7)))*43758.5) * .06;
    float out_val = clamp(ir + grain*uInfrared, 0.,1.);
    // Teinte légèrement chaude
    vec3 warm = vec3(out_val*1.05, out_val*.97, out_val*.9);
    return vec4(mix(col.rgb, warm, uInfrared), 1.);
}"""

# ─────────────────────────────────────────────────────────────────────────────
# EffectDef + catalogue
# ─────────────────────────────────────────────────────────────────────────────

class EffectDef:
    def __init__(self, name: str, icon: str, fn_name: str, glsl_fn: str, params: list):
        self.name    = name
        self.icon    = icon
        self.fn_name = fn_name   # nom de la fonction GLSL (ex: "fx_chromatic")
        self.glsl    = glsl_fn   # corps de la fonction composable
        self.params  = params    # list of dict

EFFECTS: list[EffectDef] = [
    EffectDef("Aberration chr.", "🌈", "fx_chromatic", _FX_CHROMATIC, [
        dict(name="uChromatic",  label="Force",     mn=0.0,  mx=0.02, step=0.001, default=0.006),
    ]),
    EffectDef("Bloom",           "✨", "fx_bloom",    _FX_BLOOM, [
        dict(name="uBloom",      label="Intensité", mn=0.0,  mx=1.0,  step=0.01,  default=0.4),
    ]),
    EffectDef("Vignette",        "🔲", "fx_vignette", _FX_VIGNETTE, [
        dict(name="uVignette",   label="Force",     mn=0.0,  mx=2.0,  step=0.05,  default=0.6),
    ]),
    EffectDef("Flou",            "💧", "fx_blur",     _FX_BLUR, [
        dict(name="uBlurRadius", label="Rayon",     mn=0.0,  mx=8.0,  step=0.1,   default=1.5),
    ]),
    EffectDef("Glitch",          "⚡", "fx_glitch",   _FX_GLITCH, [
        dict(name="uGlitch",     label="Intensité", mn=0.0,  mx=1.0,  step=0.01,  default=0.35),
    ]),
    EffectDef("CRT / Scanlines", "📺", "fx_crt",      _FX_CRT, [
        dict(name="uScanlines",  label="Lignes",    mn=0.0,  mx=1.0,  step=0.05,  default=0.7),
        dict(name="uCurvature",  label="Courbure",  mn=0.0,  mx=0.3,  step=0.01,  default=0.08),
    ]),
    EffectDef("Film Grain",      "🎞", "fx_grain",    _FX_GRAIN, [
        dict(name="uGrain",      label="Grain",     mn=0.0,  mx=1.0,  step=0.01,  default=0.35),
    ]),
    EffectDef("Couleur",         "🎨", "fx_color",    _FX_COLOR, [
        dict(name="uSaturation", label="Saturation",mn=0.0,  mx=3.0,  step=0.05,  default=1.0),
        dict(name="uContrast",   label="Contraste", mn=0.0,  mx=3.0,  step=0.05,  default=1.0),
        dict(name="uBrightness", label="Luminosité",mn=-1.0, mx=1.0,  step=0.02,  default=0.0),
    ]),
    EffectDef("Hue Shift",       "🌊", "fx_hueshift", _FX_HUESHIFT, [
        dict(name="uHueShift",   label="Teinte",    mn=0.0,  mx=1.0,  step=0.01,  default=0.0),
    ]),
    EffectDef("Pixelisation",    "🟦", "fx_pixel",    _FX_PIXEL, [
        dict(name="uPixelSize",  label="Taille px", mn=1.0,  mx=32.0, step=0.5,   default=4.0),
    ]),
    EffectDef("Rétro palette",   "🕹", "fx_retro",    _FX_RETRO, [
        dict(name="uColors",     label="Couleurs",  mn=2.0,  mx=32.0, step=1.0,   default=8.0),
    ]),
    EffectDef("Kaleidoscope",    "🔮", "fx_kaleido",  _FX_KALEIDO, [
        dict(name="uKaleido",    label="Segments",  mn=2.0,  mx=16.0, step=1.0,   default=6.0),
    ]),
    EffectDef("Miroir",          "🪞", "fx_mirror",   _FX_MIRROR, [
        dict(name="uMirrorX",    label="Miroir X",  mn=0.0,  mx=1.0,  step=1.0,   default=0.0),
        dict(name="uMirrorY",    label="Miroir Y",  mn=0.0,  mx=1.0,  step=1.0,   default=0.0),
    ]),

    # ── 20 nouveaux effets v2.1 ───────────────────────────────────────────
    EffectDef("Netteté",         "🔍", "fx_sharpen",   _FX_SHARPEN, [
        dict(name="uSharpen",    label="Force",     mn=0.0,  mx=2.0,  step=0.05,  default=0.5),
    ]),
    EffectDef("Contours",        "✏️", "fx_sobel",     _FX_SOBEL, [
        dict(name="uEdge",       label="Intensité", mn=0.5,  mx=4.0,  step=0.1,   default=1.0),
    ]),
    EffectDef("Postérisation",   "🎭", "fx_posterize", _FX_POSTERIZE, [
        dict(name="uPosterize",  label="Niveaux",   mn=2.0,  mx=16.0, step=1.0,   default=4.0),
    ]),
    EffectDef("Duotone",         "🎨", "fx_duotone",   _FX_DUOTONE, [
        dict(name="uDuoR1",      label="Sombre R",  mn=0.0,  mx=1.0,  step=0.01,  default=0.05),
        dict(name="uDuoG1",      label="Sombre G",  mn=0.0,  mx=1.0,  step=0.01,  default=0.0),
        dict(name="uDuoB1",      label="Sombre B",  mn=0.0,  mx=1.0,  step=0.01,  default=0.3),
        dict(name="uDuoR2",      label="Clair R",   mn=0.0,  mx=1.0,  step=0.01,  default=1.0),
        dict(name="uDuoG2",      label="Clair G",   mn=0.0,  mx=1.0,  step=0.01,  default=0.85),
        dict(name="uDuoB2",      label="Clair B",   mn=0.0,  mx=1.0,  step=0.01,  default=0.3),
    ]),
    EffectDef("Néon",            "💜", "fx_neon",      _FX_NEON, [
        dict(name="uNeon",       label="Intensité", mn=0.0,  mx=3.0,  step=0.05,  default=1.0),
    ]),
    EffectDef("Thermique",       "🌡", "fx_thermal",   _FX_THERMAL, [
        dict(name="uThermal",    label="Mix",       mn=0.0,  mx=1.0,  step=0.02,  default=0.8),
    ]),
    EffectDef("Vieux film",      "📽", "fx_oldfilm",   _FX_OLDFILM, [
        dict(name="uOldFilm",    label="Intensité", mn=0.0,  mx=1.0,  step=0.02,  default=0.7),
    ]),
    EffectDef("Demi-teintes",    "⚫", "fx_halftone",  _FX_HALFTONE, [
        dict(name="uHalftone",   label="Taille",    mn=2.0,  mx=20.0, step=0.5,   default=6.0),
    ]),
    EffectDef("Peinture",        "🖌", "fx_oilpaint",  _FX_OILPAINT, [
        dict(name="uOilRadius",  label="Rayon",     mn=1.0,  mx=6.0,  step=0.5,   default=2.0),
    ]),
    EffectDef("Fisheye",         "🐟", "fx_fisheye",   _FX_FISHEYE, [
        dict(name="uFisheye",    label="Force",     mn=0.0,  mx=3.0,  step=0.05,  default=0.8),
    ]),
    EffectDef("RGB Split",       "🌀", "fx_rgbsplit",  _FX_RGBSPLIT, [
        dict(name="uRGBSplit",   label="Décalage",  mn=0.0,  mx=0.05, step=0.001, default=0.008),
        dict(name="uRGBAngle",   label="Angle",     mn=0.0,  mx=1.0,  step=0.01,  default=0.0),
    ]),
    EffectDef("Déformation",     "〰", "fx_warp",      _FX_WARP, [
        dict(name="uWarp",       label="Amplitude", mn=0.0,  mx=1.0,  step=0.02,  default=0.4),
        dict(name="uWarpFreq",   label="Fréquence", mn=0.5,  mx=8.0,  step=0.1,   default=2.0),
    ]),
    EffectDef("Zoom",            "🔎", "fx_zoom",      _FX_ZOOM, [
        dict(name="uZoom",       label="Zoom",      mn=0.1,  mx=4.0,  step=0.05,  default=1.5),
        dict(name="uZoomX",      label="Centre X",  mn=0.0,  mx=1.0,  step=0.01,  default=0.5),
        dict(name="uZoomY",      label="Centre Y",  mn=0.0,  mx=1.0,  step=0.01,  default=0.5),
    ]),
    EffectDef("Tilt-Shift",      "📷", "fx_tiltshift", _FX_TILT_SHIFT, [
        dict(name="uTiltFocus",  label="Focus Y",   mn=0.0,  mx=1.0,  step=0.01,  default=0.5),
        dict(name="uTiltBlur",   label="Transition",mn=0.05, mx=0.8,  step=0.02,  default=0.2),
    ]),
    EffectDef("Tramage",         "🔲", "fx_dithering", _FX_DITHERING, [
        dict(name="uDither",     label="Niveaux",   mn=2.0,  mx=16.0, step=1.0,   default=4.0),
    ]),
    EffectDef("Recolorisation",  "🌈", "fx_recolor",   _FX_RECOLOR, [
        dict(name="uRecolorHue", label="Teinte",    mn=0.0,  mx=1.0,  step=0.01,  default=0.0),
        dict(name="uRecolorSat", label="Saturation",mn=0.0,  mx=3.0,  step=0.05,  default=1.0),
        dict(name="uRecolorVal", label="Luminosité",mn=0.0,  mx=3.0,  step=0.05,  default=1.0),
    ]),
    EffectDef("Vague",           "🌊", "fx_wave",      _FX_WAVE, [
        dict(name="uWaveAmp",    label="Amplitude", mn=0.0,  mx=1.0,  step=0.02,  default=0.4),
        dict(name="uWaveFreq",   label="Fréquence", mn=0.5,  mx=8.0,  step=0.1,   default=2.0),
        dict(name="uWaveAxis",   label="Axe (0=X 1=Y)", mn=0.0, mx=1.0, step=1.0, default=0.0),
    ]),
    EffectDef("Tunnel",          "🕳", "fx_tunnel",    _FX_TUNNEL, [
        dict(name="uTunnelSpeed",label="Vitesse",   mn=0.0,  mx=3.0,  step=0.05,  default=0.8),
        dict(name="uTunnelZoom", label="Zoom",      mn=0.1,  mx=5.0,  step=0.1,   default=1.5),
    ]),
    EffectDef("Relief",          "⛰", "fx_relief",    _FX_RELIEF, [
        dict(name="uRelief",     label="Profondeur",mn=0.5,  mx=6.0,  step=0.1,   default=2.0),
    ]),

    # ── 20 nouveaux effets v3.0 ───────────────────────────────────────────────
    EffectDef("ASCII Art",        "🔤", "fx_ascii",         _FX_ASCII, [
        dict(name="uAsciiSize",    label="Taille cell.", mn=4.0,  mx=20.0, step=1.0,   default=8.0),
    ]),
    EffectDef("Aquarelle",        "🖌", "fx_watercolor",    _FX_WATERCOLOR, [
        dict(name="uWatercolor",   label="Diffusion",    mn=0.0,  mx=1.0,  step=0.02,  default=0.5),
        dict(name="uWaterEdge",    label="Bords",        mn=0.0,  mx=3.0,  step=0.05,  default=1.0),
    ]),
    EffectDef("Hachures",         "✏️", "fx_crosshatch",    _FX_CROSSHATCH, [
        dict(name="uHatchDensity", label="Densité",      mn=2.0,  mx=12.0, step=0.5,   default=5.0),
        dict(name="uHatchAngle",   label="Angle",        mn=0.0,  mx=1.0,  step=0.01,  default=0.125),
    ]),
    EffectDef("Hologramme",       "🔷", "fx_hologram",      _FX_HOLOGRAM, [
        dict(name="uHoloGlitch",   label="Glitch",       mn=0.0,  mx=1.0,  step=0.02,  default=0.3),
        dict(name="uHoloColor",    label="Couleur",      mn=0.0,  mx=1.0,  step=0.01,  default=0.0),
    ]),
    EffectDef("Verre dépoli",     "🧊", "fx_frosted",       _FX_FROSTED, [
        dict(name="uFrost",        label="Diffusion",    mn=0.0,  mx=1.0,  step=0.02,  default=0.5),
        dict(name="uFrostScale",   label="Grain",        mn=1.0,  mx=20.0, step=0.5,   default=6.0),
    ]),
    EffectDef("Liquide",          "💧", "fx_liquid",         _FX_LIQUID, [
        dict(name="uLiquid",       label="Amplitude",    mn=0.0,  mx=1.0,  step=0.02,  default=0.4),
        dict(name="uLiquidSpeed",  label="Vitesse",      mn=0.0,  mx=3.0,  step=0.05,  default=0.8),
    ]),
    EffectDef("Vitrail",          "🪟", "fx_stained",        _FX_STAINED, [
        dict(name="uStainedLight", label="Luminosité",   mn=0.5,  mx=3.0,  step=0.05,  default=1.5),
        dict(name="uStainedSat",   label="Saturation",   mn=0.5,  mx=3.0,  step=0.05,  default=1.5),
    ]),
    EffectDef("Pixel Sort",       "🔀", "fx_pixelsort",      _FX_PIXELSORT, [
        dict(name="uSortThresh",   label="Seuil",        mn=0.0,  mx=1.0,  step=0.02,  default=0.5),
        dict(name="uSortDir",      label="Dir (0=V 1=H)",mn=0.0,  mx=1.0,  step=1.0,   default=0.0),
    ]),
    EffectDef("Aura",             "🌟", "fx_aura",           _FX_AURA, [
        dict(name="uAuraRadius",   label="Rayon",        mn=1.0,  mx=8.0,  step=0.2,   default=3.0),
        dict(name="uAuraIntensity",label="Intensité",    mn=0.0,  mx=2.0,  step=0.05,  default=0.7),
    ]),
    EffectDef("Datamosh",         "📼", "fx_datamosh",       _FX_DATAMOSH, [
        dict(name="uDatamosh",     label="Intensité",    mn=0.0,  mx=1.0,  step=0.02,  default=0.4),
    ]),
    EffectDef("Voronoï",          "🔶", "fx_voronoi_col",    _FX_VORONOI_COL, [
        dict(name="uVoronoiScale", label="Échelle",      mn=0.5,  mx=8.0,  step=0.1,   default=3.0),
        dict(name="uVoronoiMix",   label="Intensité",    mn=0.0,  mx=1.0,  step=0.02,  default=0.7),
    ]),
    EffectDef("LCD / Subpixel",   "🖥", "fx_scanline_color", _FX_SCANLINE_COLOR, [
        dict(name="uScanColor",    label="Intensité",    mn=0.0,  mx=1.0,  step=0.02,  default=0.5),
        dict(name="uScanFreq",     label="Fréquence",    mn=1.0,  mx=4.0,  step=0.5,   default=1.0),
    ]),
    EffectDef("Dissolution",      "💨", "fx_dissolve",       _FX_DISSOLVE, [
        dict(name="uDissolve",     label="Quantité",     mn=0.0,  mx=1.0,  step=0.01,  default=0.0),
        dict(name="uDissolveScale",label="Échelle bruit",mn=0.5,  mx=4.0,  step=0.1,   default=1.0),
    ]),
    EffectDef("Retrowave",        "🌆", "fx_retrowave",      _FX_RETROWAVE, [
        dict(name="uRetroGrid",    label="Grille",       mn=1.0,  mx=8.0,  step=0.1,   default=3.0),
        dict(name="uRetroGlow",    label="Néon",         mn=0.0,  mx=2.0,  step=0.05,  default=0.8),
    ]),
    EffectDef("Scintillement",    "✨", "fx_sparkle",        _FX_SPARKLE, [
        dict(name="uSparkle",      label="Densité",      mn=0.0,  mx=1.0,  step=0.02,  default=0.3),
        dict(name="uSparkleDensity",label="Taille",      mn=4.0,  mx=20.0, step=1.0,   default=8.0),
    ]),
    EffectDef("Kaléidoscope II",  "🔯", "fx_mirror_kaleid",  _FX_MIRROR_KALEID, [
        dict(name="uMKSegs",       label="Segments",     mn=3.0,  mx=20.0, step=1.0,   default=8.0),
        dict(name="uMKSpin",       label="Rotation",     mn=-1.0, mx=1.0,  step=0.05,  default=0.2),
    ]),
    EffectDef("Aurore boréale",   "🌌", "fx_aurora",         _FX_AURORA, [
        dict(name="uAuroraSpeed",  label="Vitesse",      mn=0.0,  mx=2.0,  step=0.05,  default=0.5),
        dict(name="uAuroraWave",   label="Fréquence",    mn=1.0,  mx=8.0,  step=0.1,   default=3.0),
    ]),
    EffectDef("Relief directionnel","⬆", "fx_emboss_dir",   _FX_CONVOLVE_EMBOSS, [
        dict(name="uEmbossAngle",  label="Angle",        mn=0.0,  mx=1.0,  step=0.01,  default=0.125),
        dict(name="uEmbossStrength",label="Force",       mn=0.5,  mx=6.0,  step=0.1,   default=2.0),
    ]),
    EffectDef("Infrarouge",       "🌡", "fx_infrared",       _FX_INFRARED, [
        dict(name="uInfrared",     label="Intensité",    mn=0.0,  mx=1.0,  step=0.02,  default=0.8),
    ]),
    EffectDef("Négatif",         "☯", "fx_negative",  """\
vec4 fx_negative(vec4 col, vec2 uv, vec2 C) {
    return vec4(1.-col.rgb, 1.);
}""", []),
]


# ─────────────────────────────────────────────────────────────────────────────
# FXComposer — génère un shader Shadertoy unique chaînant les effets actifs
# ─────────────────────────────────────────────────────────────────────────────

def compose_fx_shader(active_effects: list[EffectDef]) -> str | None:
    """
    Génère un shader Shadertoy complet qui applique en chaîne
    tous les effets actifs dans l'ordre de la liste.
    Retourne None si aucun effet n'est actif.
    """
    if not active_effects:
        return None

    parts = []
    # 1. Définitions des fonctions composables
    for fx in active_effects:
        parts.append(fx.glsl)

    # 2. mainImage qui chaîne les effets
    parts.append("void mainImage(out vec4 O, vec2 C) {")
    parts.append("    vec2 uv = C / iResolution.xy;")
    parts.append("    vec4 col = texture(iChannel0, uv);")
    for fx in active_effects:
        parts.append(f"    col = {fx.fn_name}(col, uv, C);")
    parts.append("    O = col;")
    parts.append("}")

    return "\n".join(parts)


class LeftPanel(QWidget):
    shader_file_requested    = pyqtSignal(str)
    audio_file_requested     = pyqtSignal(str)
    uniform_value_changed    = pyqtSignal(str, object)
    effect_changed           = pyqtSignal(object)  # str (glsl) ou None
    shader_save_requested    = pyqtSignal(str)      # format demandé : 'st', 'glsl', 'trans'
    export_requested         = pyqtSignal(dict)     # dict de paramètres d'export
    # v2.2 — Preview miniature
    thumbnail_requested      = pyqtSignal(str)      # chemin du shader dont on veut le thumbnail
    # v2.3 — Auto-paramétrage IA
    apply_param_to_shader    = pyqtSignal(object)   # ShaderParam — demande d'injection dans le code
    params_scan_requested    = pyqtSignal()         # demande de scan du shader courant

    # Cache des thumbnails : chemin → QPixmap (120×68)
    _thumbnail_cache: dict[str, "QPixmap"] = {}
    THUMB_W = 120
    THUMB_H = 68

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(248)
        self._fx_btns: list[QPushButton] = []
        self._fx_toggles: list[QPushButton] = []   # boutons ON/OFF par effet
        self._fx_active: list[bool] = [False] * len(EFFECTS)
        self._fx_param_widgets: list[QWidget | None] = [None] * len(EFFECTS)
        # _fx_spins[i] = liste des QDoubleSpinBox de l'effet i (dans l'ordre de fx.params)
        self._fx_spins: list[list] = [[] for _ in EFFECTS]
        self._params_layout = None
        self._ai_detector   = AIParamDetector()
        self._ai_params: list[ShaderParam] = []     # paramètres détectés
        self._param_sliders: dict[str, QSlider] = {}  # name → slider
        self._param_spins:   dict[str, QDoubleSpinBox] = {}  # name → spinbox
        self._params_list_layout = None              # layout dynamique des params
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        tabs = QTabWidget()
        tabs.setStyleSheet(_TAB_STYLE)
        tabs.addTab(self._build_fx_tab(),     "✦ FX")
        tabs.addTab(self._build_files_tab(),  "Files")
        tabs.addTab(self._build_params_tab(), "⚙ Params")
        tabs.addTab(self._build_export_tab(), "⬇ Export")
        tabs.addTab(self._build_info_tab(),   "Info")
        root.addWidget(tabs)

    # ── FX Tab ───────────────────────────────────────────────────────────────

    def _build_fx_tab(self) -> QWidget:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── En-tête ──────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet("background:#111318;")
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(8, 6, 8, 4)

        lbl = QLabel("POST-PROCESSING")
        lbl.setStyleSheet("color:#3a3f58; font:bold 8px 'Segoe UI';")
        hdr_l.addWidget(lbl)
        hdr_l.addStretch()

        btn_all_off = QPushButton("✕ Tout off")
        btn_all_off.setFixedHeight(18)
        btn_all_off.setStyleSheet(_FX_OFF_BTN)
        btn_all_off.clicked.connect(self._disable_all_fx)
        hdr_l.addWidget(btn_all_off)
        outer_layout.addWidget(hdr)

        # ── Liste scrollable des effets ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea{border:none;background:#111318;}"
            "QScrollBar:vertical{background:#111318;width:5px;}"
            "QScrollBar::handle:vertical{background:#2a2d3a;border-radius:2px;}"
        )

        fx_container = QWidget()
        fx_container.setStyleSheet("background:#111318;")
        fx_list_layout = QVBoxLayout(fx_container)
        fx_list_layout.setContentsMargins(5, 4, 5, 8)
        fx_list_layout.setSpacing(2)

        self._fx_btns    = []
        self._fx_toggles = []
        self._fx_param_widgets = []

        for i, fx in enumerate(EFFECTS):
            # ── Ligne effet ──────────────────────────────────────────────────
            row_widget = QWidget()
            row_widget.setStyleSheet("background:transparent;")
            row_l = QHBoxLayout(row_widget)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)

            # Bouton toggle ON/OFF
            tog = QPushButton("OFF")
            tog.setFixedSize(34, 22)
            tog.setCheckable(True)
            tog.setStyleSheet(_FX_TOGGLE_OFF)
            row_l.addWidget(tog)
            self._fx_toggles.append(tog)

            # Bouton principal (nom de l'effet) — cliquable pour expand params
            btn = QPushButton(f"{fx.icon}  {fx.name}")
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet(_FX_BTN)
            row_l.addWidget(btn)
            self._fx_btns.append(btn)

            fx_list_layout.addWidget(row_widget)

            # ── Zone paramètres (cachée par défaut) ──────────────────────────
            params_widget = self._build_fx_params_widget(fx, i)
            params_widget.setVisible(False)
            fx_list_layout.addWidget(params_widget)
            self._fx_param_widgets.append(params_widget)

            # Connexions (closure propre sur i)
            def _make_handlers(idx: int, toggle: QPushButton, name_btn: QPushButton,
                                param_w: QWidget):
                def on_toggle(checked: bool):
                    self._fx_active[idx] = checked
                    toggle.setText("ON" if checked else "OFF")
                    toggle.setStyleSheet(_FX_TOGGLE_ON if checked else _FX_TOGGLE_OFF)
                    # Afficher les params si ON, cacher si OFF
                    param_w.setVisible(checked or name_btn.isChecked())
                    self._emit_composed_shader()

                def on_name_btn(checked: bool):
                    # Expand/collapse les paramètres
                    param_w.setVisible(checked or self._fx_active[idx])

                return on_toggle, on_name_btn

            on_toggle, on_name = _make_handlers(i, tog, btn, params_widget)
            tog.toggled.connect(on_toggle)
            btn.toggled.connect(on_name)

        fx_list_layout.addStretch()
        scroll.setWidget(fx_container)
        outer_layout.addWidget(scroll, 1)
        return outer

    def _build_fx_params_widget(self, fx: EffectDef, fx_idx: int) -> QWidget:
        """Construit le widget de paramètres pour un effet donné."""
        w = QWidget()
        w.setStyleSheet("background:#181a22; border-left:2px solid #1e2232; margin-left:8px;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 6, 6, 6)
        lay.setSpacing(4)

        self._fx_spins[fx_idx] = []   # reset pour cet effet

        for p in fx.params:
            pbox = QWidget()
            pbox.setStyleSheet("background:transparent;")
            pl = QVBoxLayout(pbox)
            pl.setContentsMargins(0, 0, 0, 2)
            pl.setSpacing(1)

            row = QHBoxLayout()
            lbl = QLabel(p['label'])
            lbl.setStyleSheet("color:#7880a0; font:8px 'Segoe UI'; min-width:60px;")
            spin = QDoubleSpinBox()
            spin.setRange(p['mn'], p['mx'])
            spin.setSingleStep(p['step'])
            spin.setValue(p['default'])
            decimals = 3 if p['step'] < 0.01 else (2 if p['step'] < 0.1 else 1)
            spin.setDecimals(decimals)
            spin.setFixedWidth(68)
            spin.setFixedHeight(18)
            spin.setStyleSheet(_SPIN_STYLE)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(spin)
            pl.addLayout(row)

            # Stocker la référence pour get/restore fx state
            self._fx_spins[fx_idx].append(spin)

            slider = QSlider(Qt.Orientation.Horizontal)
            steps  = max(1, int((p['mx'] - p['mn']) / p['step']))
            slider.setRange(0, steps)
            slider.setValue(int((p['default'] - p['mn']) / p['step']))
            slider.setStyleSheet(_SLIDER_STYLE)
            pl.addWidget(slider)

            # Sync spin ↔ slider
            pdef = p
            def make_handlers(sp, sl, pd):
                def on_spin(v):
                    sl.blockSignals(True)
                    sl.setValue(int(round((v - pd['mn']) / pd['step'])))
                    sl.blockSignals(False)
                    self.uniform_value_changed.emit(pd['name'], float(v))
                def on_slide(v):
                    val = pd['mn'] + v * pd['step']
                    sp.blockSignals(True)
                    sp.setValue(val)
                    sp.blockSignals(False)
                    self.uniform_value_changed.emit(pd['name'], float(val))
                return on_spin, on_slide
            on_spin, on_slide = make_handlers(spin, slider, pdef)
            spin.valueChanged.connect(on_spin)
            slider.valueChanged.connect(on_slide)

            # Envoyer valeur initiale
            self.uniform_value_changed.emit(p['name'], float(p['default']))

            lay.addWidget(pbox)

        return w

    def _emit_composed_shader(self):
        """Recompose et émet le shader combiné des effets actifs."""
        active = [fx for i, fx in enumerate(EFFECTS) if self._fx_active[i]]
        shader = compose_fx_shader(active)
        if shader:
            active_names = " + ".join(fx.name for fx in active)
            log.debug("FX composé : %s", active_names)
        self.effect_changed.emit(shader)

    def _disable_all_fx(self):
        """Désactive tous les effets d'un coup."""
        for i, (tog, pw) in enumerate(zip(self._fx_toggles, self._fx_param_widgets)):
            self._fx_active[i] = False
            tog.blockSignals(True)
            tog.setChecked(False)
            tog.setText("OFF")
            tog.setStyleSheet(_FX_TOGGLE_OFF)
            tog.blockSignals(False)
            # Garder les params visibles si l'expand est ouvert
            name_btn = self._fx_btns[i]
            pw.setVisible(name_btn.isChecked())
        self.effect_changed.emit(None)
        log.debug("Tous les effets FX désactivés")

    # ── API publique — état FX par shader ────────────────────────────────────

    def get_fx_state(self) -> dict:
        """
        Retourne l'état complet du panneau FX : quels effets sont actifs
        et les valeurs courantes de leurs paramètres.

        Format retourné :
        {
            "active": [True, False, ...],          # bool par effet (ordre EFFECTS)
            "params": {                             # valeurs des spinboxes
                "uChromatic": 0.006,
                "uBloom": 0.4,
                ...
            }
        }
        """
        params: dict[str, float] = {}
        for i, fx in enumerate(EFFECTS):
            for j, p in enumerate(fx.params):
                if j < len(self._fx_spins[i]):
                    params[p['name']] = self._fx_spins[i][j].value()
                else:
                    params[p['name']] = p['default']
        return {
            "active": list(self._fx_active),
            "params": params,
        }

    def restore_fx_state(self, state: dict, emit: bool = True):
        """
        Restaure l'état FX depuis un dict produit par get_fx_state().
        Si emit=True, émet effect_changed et uniform_value_changed
        pour appliquer immédiatement le rendu.
        """
        active_list: list[bool] = state.get("active", [False] * len(EFFECTS))
        params: dict[str, float] = state.get("params", {})

        # 1. Restaurer les valeurs de paramètres (silencieux)
        for i, fx in enumerate(EFFECTS):
            for j, p in enumerate(fx.params):
                val = params.get(p['name'], p['default'])
                if j < len(self._fx_spins[i]):
                    sp = self._fx_spins[i][j]
                    sp.blockSignals(True)
                    sp.setValue(val)
                    sp.blockSignals(False)
                if emit:
                    self.uniform_value_changed.emit(p['name'], float(val))

        # 2. Restaurer les états ON/OFF
        for i, tog in enumerate(self._fx_toggles):
            checked = active_list[i] if i < len(active_list) else False
            self._fx_active[i] = checked
            tog.blockSignals(True)
            tog.setChecked(checked)
            tog.setText("ON" if checked else "OFF")
            tog.setStyleSheet(_FX_TOGGLE_ON if checked else _FX_TOGGLE_OFF)
            tog.blockSignals(False)
            # Visibilité des params : ON = visible, expand fermé = caché
            pw = self._fx_param_widgets[i]
            if pw:
                pw.setVisible(checked or self._fx_btns[i].isChecked())

        # 3. Recomposer et émettre le shader
        if emit:
            self._emit_composed_shader()

        log.debug("État FX restauré — actifs : %s",
                  [EFFECTS[i].name for i, a in enumerate(active_list) if a])

    # ── Files Tab ─────────────────────────────────────────────────────────────

    def _build_files_tab(self) -> QWidget:
        outer = QWidget()
        outer.setStyleSheet("background:#0d0f15;")
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar supérieure ────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(38)
        toolbar.setStyleSheet(
            "background:#0d0f15;"
            "border-bottom:1px solid #13162a;"
        )
        tb_l = QHBoxLayout(toolbar)
        tb_l.setContentsMargins(8, 0, 8, 0)
        tb_l.setSpacing(4)

        # Bouton Nouveau shader
        btn_new = QPushButton("+ Nouveau")
        btn_new.setFixedHeight(24)
        btn_new.setStyleSheet(_EXP_BTN_PRIMARY)
        btn_new.setToolTip("Créer un nouveau shader")
        btn_new.clicked.connect(self._on_new_shader)
        tb_l.addWidget(btn_new)

        # Bouton Sauvegarder
        btn_save = QPushButton("Sauvegarder")
        btn_save.setFixedHeight(24)
        btn_save.setStyleSheet(_EXP_BTN_SECONDARY)
        btn_save.setToolTip("Sauvegarder le shader actif")
        btn_save.clicked.connect(self._on_save_shader)
        tb_l.addWidget(btn_save)

        tb_l.addStretch()

        # Bouton Ouvrir fichier
        btn_open = QToolButton()
        btn_open.setText("📄")
        btn_open.setFixedSize(24, 24)
        btn_open.setStyleSheet(_EXP_TOOL_BTN)
        btn_open.setToolTip("Ouvrir un fichier")
        btn_open.clicked.connect(self._open_dialog)
        tb_l.addWidget(btn_open)

        # Bouton Ouvrir dossier
        btn_folder = QToolButton()
        btn_folder.setText("📁")
        btn_folder.setFixedSize(24, 24)
        btn_folder.setStyleSheet(_EXP_TOOL_BTN)
        btn_folder.setToolTip("Ouvrir un dossier")
        btn_folder.clicked.connect(self._open_folder_dialog)
        tb_l.addWidget(btn_folder)

        # Bouton Rafraîchir
        btn_refresh = QToolButton()
        btn_refresh.setText("⟳")
        btn_refresh.setFixedSize(24, 24)
        btn_refresh.setStyleSheet(_EXP_TOOL_BTN)
        btn_refresh.setToolTip("Rafraîchir")
        btn_refresh.clicked.connect(self.refresh_tree)
        tb_l.addWidget(btn_refresh)

        layout.addWidget(toolbar)

        # ── Barre de recherche + filtres ──────────────────────────────────────
        search_bar = QWidget()
        search_bar.setFixedHeight(34)
        search_bar.setStyleSheet(
            "background:#0a0c14;"
            "border-bottom:1px solid #13162a;"
        )
        sb_l = QHBoxLayout(search_bar)
        sb_l.setContentsMargins(8, 4, 8, 4)
        sb_l.setSpacing(6)

        # Icône loupe
        lupe = QLabel("⌕")
        lupe.setStyleSheet("color:#303458; font:13px; background:transparent;")
        sb_l.addWidget(lupe)

        # Champ de recherche
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Rechercher…")
        self._search_edit.setStyleSheet(_EXP_SEARCH_STYLE)
        self._search_edit.textChanged.connect(self._on_search_changed)
        sb_l.addWidget(self._search_edit, 1)

        # Filtre type
        self._type_filter = QComboBox()
        self._type_filter.addItems(["Tous", "Shaders", "Audio"])
        self._type_filter.setFixedWidth(72)
        self._type_filter.setStyleSheet(_EXP_COMBO_STYLE)
        self._type_filter.currentIndexChanged.connect(self._on_filter_changed)
        sb_l.addWidget(self._type_filter)

        layout.addWidget(search_bar)

        # ── Section stats ─────────────────────────────────────────────────────
        self._stats_lbl = QLabel("")
        self._stats_lbl.setStyleSheet(
            "color:#252840; font:8px 'Segoe UI';"
            "background:#08090f;"
            "padding:2px 10px;"
        )
        layout.addWidget(self._stats_lbl)

        # ── Arbre de fichiers ─────────────────────────────────────────────────
        self._tree = FileTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setIconSize(QSize(16, 16))
        self._tree.setIndentation(14)
        self._tree.setAnimated(True)
        self._tree.setStyleSheet(_EXP_TREE_STYLE)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        self._tree.itemClicked.connect(self._on_single_click)
        self._tree.setDragEnabled(True)
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        layout.addWidget(self._tree, 1)

        # ── Panneau de prévisualisation ───────────────────────────────────────
        self._preview_panel = QWidget()
        self._preview_panel.setFixedHeight(0)   # masqué par défaut
        self._preview_panel.setStyleSheet(
            "background:#080a12;"
            "border-top:1px solid #13162a;"
        )
        pp_l = QVBoxLayout(self._preview_panel)
        pp_l.setContentsMargins(8, 6, 8, 6)
        pp_l.setSpacing(4)

        # Thumbnail
        self._preview_thumb = QLabel()
        self._preview_thumb.setFixedSize(self.THUMB_W, self.THUMB_H)
        self._preview_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_thumb.setStyleSheet(
            "background:#0c0e18; border:1px solid #13162a; border-radius:3px;"
        )
        # Nom du fichier
        self._preview_name = QLabel()
        self._preview_name.setStyleSheet(
            "color:#606890; font:bold 9px 'Segoe UI'; background:transparent;"
        )
        self._preview_name.setWordWrap(True)
        # Infos
        self._preview_info = QLabel()
        self._preview_info.setStyleSheet(
            "color:#303458; font:8px 'Segoe UI'; background:transparent;"
        )

        pp_inner = QHBoxLayout()
        pp_inner.setSpacing(8)
        pp_inner.addWidget(self._preview_thumb)
        pp_vr = QVBoxLayout()
        pp_vr.setSpacing(3)
        pp_vr.addWidget(self._preview_name)
        pp_vr.addWidget(self._preview_info)
        pp_vr.addStretch()
        pp_inner.addLayout(pp_vr, 1)
        pp_l.addLayout(pp_inner)
        layout.addWidget(self._preview_panel)

        self.refresh_tree()
        return outer

    def _on_new_shader(self):
        """Crée un nouveau shader via dialogue de choix de format."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Nouveau shader")
        dlg.setFixedWidth(340)
        dlg.setStyleSheet(_EXP_DIALOG_STYLE)

        lay = QVBoxLayout(dlg)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)

        # Header
        hdr = QWidget()
        hdr.setStyleSheet("background:#0c0e18; border-bottom:1px solid #13162a;")
        hdr.setFixedHeight(46)
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(16, 0, 16, 0)
        lbl_h = QLabel("Nouveau shader")
        lbl_h.setStyleSheet("color:#c0c8e8; font:bold 12px 'Segoe UI';")
        hdr_l.addWidget(lbl_h)
        lay.addWidget(hdr)

        body = QWidget()
        body.setStyleSheet("background:#0d0f18;")
        body_l = QVBoxLayout(body)
        body_l.setContentsMargins(16, 14, 16, 14)
        body_l.setSpacing(6)

        formats = [
            ("🎨", "Shadertoy  (.st)",   "st",
             "Format Shadertoy — mainImage(), iTime, iResolution"),
            ("💡", "GLSL standard  (.glsl)", "glsl",
             "GLSL pur — uniforms uTime, uResolution"),
            ("🔀", "Transition  (.trans)", "trans",
             "Fondu entre scènes — iChannel0, iChannel1, iProgress"),
        ]

        grp = QButtonGroup(dlg)
        self._new_fmt_radios = []
        for icon, label, fmt, tip in formats:
            card = QWidget()
            card.setStyleSheet(
                "background:#0c0e18; border:1px solid #14182e;"
                "border-radius:5px; margin-bottom:1px;"
            )
            card_l = QHBoxLayout(card)
            card_l.setContentsMargins(10, 8, 10, 8)
            card_l.setSpacing(10)
            ic = QLabel(icon)
            ic.setStyleSheet("font:15px; background:transparent;")
            ic.setFixedWidth(20)
            tv = QVBoxLayout()
            tv.setSpacing(1)
            t1 = QLabel(label)
            t1.setStyleSheet("color:#8898c8; font:bold 9px 'Segoe UI';")
            t2 = QLabel(tip)
            t2.setStyleSheet("color:#303458; font:8px 'Segoe UI';")
            tv.addWidget(t1); tv.addWidget(t2)
            rb = QRadioButton()
            rb.setStyleSheet("""
                QRadioButton::indicator { width:14px; height:14px; }
                QRadioButton::indicator:checked {
                    background:#2a4090; border:2px solid #4a70e0; border-radius:7px;
                }
                QRadioButton::indicator:unchecked {
                    background:#12141e; border:1px solid #2a2e48; border-radius:7px;
                }
            """)
            grp.addButton(rb)
            self._new_fmt_radios.append((rb, fmt))
            card_l.addWidget(ic)
            card_l.addLayout(tv, 1)
            card_l.addWidget(rb)
            body_l.addWidget(card)

        self._new_fmt_radios[0][0].setChecked(True)

        # Nom
        body_l.addSpacing(8)
        lbl_n = QLabel("Nom du fichier")
        lbl_n.setStyleSheet("color:#303458; font:9px 'Segoe UI';")
        body_l.addWidget(lbl_n)
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("mon_shader")
        name_edit.setText("nouveau_shader")
        name_edit.setStyleSheet(_EXP_SEARCH_STYLE)
        body_l.addWidget(name_edit)

        lay.addWidget(body)

        # Footer
        ftr = QWidget()
        ftr.setStyleSheet("background:#0c0e18; border-top:1px solid #13162a;")
        ftr.setFixedHeight(48)
        ftr_l = QHBoxLayout(ftr)
        ftr_l.setContentsMargins(16, 0, 16, 0)
        ftr_l.setSpacing(8)
        ftr_l.addStretch()
        btn_cancel = QPushButton("Annuler")
        btn_cancel.setStyleSheet(_EXP_BTN_SECONDARY)
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok = QPushButton("Créer")
        btn_ok.setStyleSheet(_EXP_BTN_PRIMARY)
        btn_ok.clicked.connect(dlg.accept)
        btn_ok.setDefault(True)
        ftr_l.addWidget(btn_cancel)
        ftr_l.addWidget(btn_ok)
        lay.addWidget(ftr)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        fmt = "st"
        for rb, f in self._new_fmt_radios:
            if rb.isChecked():
                fmt = f
                break

        name = name_edit.text().strip() or "nouveau_shader"
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_map = {
            'st':    os.path.join(base, 'shaders', 'stoy'),
            'glsl':  os.path.join(base, 'shaders', 'glsl'),
            'trans': os.path.join(base, 'shaders', 'trans'),
        }
        folder = folder_map[fmt]
        os.makedirs(folder, exist_ok=True)
        if not name.endswith(f'.{fmt}'):
            name += f'.{fmt}'
        path = os.path.join(folder, name)

        templates = {
            'st':   "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    vec2 uv = fragCoord / iResolution.xy;\n    fragColor = vec4(uv, 0.5 + 0.5*sin(iTime), 1.0);\n}\n",
            'glsl': "#version 330 core\nuniform vec2 uResolution;\nuniform float uTime;\nout vec4 fragColor;\nvoid main() {\n    vec2 uv = gl_FragCoord.xy / uResolution.xy;\n    fragColor = vec4(uv, 0.5+0.5*sin(uTime), 1.0);\n}\n",
            'trans': "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    vec2 uv = fragCoord / iResolution.xy;\n    vec4 a = texture(iChannel0, uv);\n    vec4 b = texture(iChannel1, uv);\n    fragColor = mix(a, b, iProgress);\n}\n",
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(templates[fmt])
            self.refresh_tree()
            self.shader_file_requested.emit(path)
        except OSError as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de créer :\n{e}")

    def _on_save_shader(self):
        """Dialogue pour choisir format + emplacement et sauvegarder le shader actif."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Sauvegarder le shader")
        dlg.setFixedWidth(340)
        dlg.setStyleSheet(_EXP_DIALOG_STYLE)

        lay = QVBoxLayout(dlg)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)

        hdr = QWidget()
        hdr.setStyleSheet("background:#0c0e18; border-bottom:1px solid #13162a;")
        hdr.setFixedHeight(46)
        hdr_l = QHBoxLayout(hdr)
        hdr_l.setContentsMargins(16, 0, 16, 0)
        lbl_h = QLabel("Sauvegarder le shader")
        lbl_h.setStyleSheet("color:#c0c8e8; font:bold 12px 'Segoe UI';")
        hdr_l.addWidget(lbl_h)
        lay.addWidget(hdr)

        body = QWidget()
        body.setStyleSheet("background:#0d0f18;")
        body_l = QVBoxLayout(body)
        body_l.setContentsMargins(16, 14, 16, 14)
        body_l.setSpacing(6)

        lbl = QLabel("Format de sauvegarde")
        lbl.setStyleSheet("color:#303458; font:9px 'Segoe UI';")
        body_l.addWidget(lbl)

        formats = [
            ("🎨", "Shadertoy  (.st)",      "st",
             "mainImage(), iTime, iResolution"),
            ("💡", "GLSL standard  (.glsl)","glsl",
             "Uniforms uTime, uResolution"),
            ("🔀", "Transition  (.trans)",  "trans",
             "iChannel0, iChannel1, iProgress"),
        ]
        grp = QButtonGroup(dlg)
        radios = []
        for icon, label, fmt, tip in formats:
            card = QWidget()
            card.setStyleSheet(
                "background:#0c0e18; border:1px solid #14182e;"
                "border-radius:5px; margin-bottom:1px;"
            )
            card_l = QHBoxLayout(card)
            card_l.setContentsMargins(10, 8, 10, 8)
            card_l.setSpacing(10)
            ic = QLabel(icon)
            ic.setStyleSheet("font:15px; background:transparent;")
            ic.setFixedWidth(20)
            tv = QVBoxLayout()
            tv.setSpacing(1)
            t1 = QLabel(label)
            t1.setStyleSheet("color:#8898c8; font:bold 9px 'Segoe UI';")
            t2 = QLabel(tip)
            t2.setStyleSheet("color:#303458; font:8px 'Segoe UI';")
            tv.addWidget(t1); tv.addWidget(t2)
            rb = QRadioButton()
            rb.setStyleSheet("""
                QRadioButton::indicator { width:14px; height:14px; }
                QRadioButton::indicator:checked {
                    background:#2a4090; border:2px solid #4a70e0; border-radius:7px;
                }
                QRadioButton::indicator:unchecked {
                    background:#12141e; border:1px solid #2a2e48; border-radius:7px;
                }
            """)
            grp.addButton(rb)
            radios.append((rb, fmt))
            card_l.addWidget(ic)
            card_l.addLayout(tv, 1)
            card_l.addWidget(rb)
            body_l.addWidget(card)
        radios[0][0].setChecked(True)
        lay.addWidget(body)

        ftr = QWidget()
        ftr.setStyleSheet("background:#0c0e18; border-top:1px solid #13162a;")
        ftr.setFixedHeight(48)
        ftr_l = QHBoxLayout(ftr)
        ftr_l.setContentsMargins(16, 0, 16, 0)
        ftr_l.setSpacing(8)
        ftr_l.addStretch()
        btn_cancel = QPushButton("Annuler")
        btn_cancel.setStyleSheet(_EXP_BTN_SECONDARY)
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok = QPushButton("Choisir l'emplacement…")
        btn_ok.setStyleSheet(_EXP_BTN_PRIMARY)
        btn_ok.clicked.connect(dlg.accept)
        btn_ok.setDefault(True)
        ftr_l.addWidget(btn_cancel)
        ftr_l.addWidget(btn_ok)
        lay.addWidget(ftr)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        fmt = "st"
        for rb, f in radios:
            if rb.isChecked():
                fmt = f
                break

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_map = {
            'st':    (os.path.join(base, 'shaders', 'stoy'),  "Shadertoy (*.st)",    ".st"),
            'glsl':  (os.path.join(base, 'shaders', 'glsl'),  "GLSL (*.glsl)",       ".glsl"),
            'trans': (os.path.join(base, 'shaders', 'trans'), "Transition (*.trans)",".trans"),
        }
        default_dir, filter_str, suffix = folder_map[fmt]
        os.makedirs(default_dir, exist_ok=True)

        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder le shader", default_dir, filter_str)
        if not path:
            return
        if not path.lower().endswith(suffix):
            path += suffix

        self.shader_save_requested.emit(path)
        self.refresh_tree()

    def _open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Ouvrir un dossier")
        if folder:
            self._browse_external_folder(folder)

    def _browse_external_folder(self, folder: str):
        label = os.path.basename(folder)
        root_item = QTreeWidgetItem([f"📂  {label}"])
        root_item.setFont(0, QFont("Segoe UI", 8, QFont.Weight.Bold))
        self._tree.addTopLevelItem(root_item)
        exts = ('.st', '.glsl', '.trans', '.wav', '.mp3', '.ogg')
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
        for fname in files:
            fpath = os.path.join(folder, fname)
            ext = os.path.splitext(fname)[1].lower()
            icon = _EXT_ICONS.get(ext, "📄")
            child = QTreeWidgetItem([f"{icon}  {fname}"])
            child.setData(0, Qt.ItemDataRole.UserRole, fpath)
            root_item.addChild(child)
        root_item.setText(0, f"📂  {label}  [{len(files)}]")
        root_item.setExpanded(True)

    def _on_search_changed(self, text: str):
        self._apply_filter(text, self._type_filter.currentIndex())

    def _on_filter_changed(self, idx: int):
        self._apply_filter(self._search_edit.text(), idx)

    def _apply_filter(self, search: str, type_idx: int):
        """Filtre l'arbre selon le texte de recherche et le type."""
        search = search.strip().lower()
        # type_idx: 0=Tous, 1=Shaders, 2=Audio
        shader_exts = {'.st', '.glsl', '.trans'}
        audio_exts  = {'.wav', '.mp3', '.ogg'}

        root = self._tree.invisibleRootItem()
        total_visible = 0

        for i in range(root.childCount()):
            section = root.child(i)
            visible_count = 0

            for j in range(section.childCount()):
                item = section.child(j)
                path = item.data(0, Qt.ItemDataRole.UserRole)
                if not path:
                    continue
                ext  = os.path.splitext(path)[1].lower()
                name = os.path.basename(path).lower()

                # Filtre type
                if type_idx == 1 and ext not in shader_exts:
                    item.setHidden(True)
                    continue
                if type_idx == 2 and ext not in audio_exts:
                    item.setHidden(True)
                    continue

                # Filtre texte
                if search and search not in name:
                    item.setHidden(True)
                    continue

                item.setHidden(False)
                visible_count += 1

            section.setHidden(visible_count == 0 and (search or type_idx > 0))
            total_visible += visible_count

        if search or type_idx > 0:
            self._stats_lbl.setText(f"  {total_visible} résultat(s)")
        else:
            self._stats_lbl.setText("")

    def refresh_tree(self):
        self._tree.clear()
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        sections = [
            ("SHADERTOY",   "🎨", os.path.join(base, 'shaders', 'stoy'),  ('.st',)),
            ("GLSL",        "💡", os.path.join(base, 'shaders', 'glsl'),  ('.glsl',)),
            ("TRANSITIONS", "🔀", os.path.join(base, 'shaders', 'trans'), ('.trans',)),
            ("AUDIO",       "🔊", os.path.join(base, 'audio'),            ('.wav', '.mp3', '.ogg')),
        ]

        total_files = 0
        for label, icon, folder, exts in sections:
            # Header de section
            root_item = QTreeWidgetItem()
            root_item.setData(0, Qt.ItemDataRole.UserRole + 1, folder)
            root_item.setData(0, Qt.ItemDataRole.UserRole + 2, exts)
            self._tree.addTopLevelItem(root_item)

            os.makedirs(folder, exist_ok=True)
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))

            for fname in files:
                fpath = os.path.join(folder, fname)
                ext   = os.path.splitext(fname)[1].lower()
                ficon = _EXT_ICONS.get(ext, "📄")
                child = QTreeWidgetItem([f"{ficon}  {fname}"])
                child.setData(0, Qt.ItemDataRole.UserRole, fpath)
                child.setToolTip(0, fpath)

                # Thumbnail si disponible
                if fpath in self._thumbnail_cache:
                    from PyQt6.QtGui import QIcon
                    child.setIcon(0, QIcon(self._thumbnail_cache[fpath]))
                root_item.addChild(child)

            count = len(files)
            total_files += count
            root_item.setText(0, f"{icon}  {label}  ({count})")
            root_item.setFont(0, QFont("Segoe UI", 7, QFont.Weight.Bold))
            root_item.setForeground(0, QColor("#2e3460"))
            root_item.setExpanded(count > 0)

        # Appliquer le filtre en cours
        self._apply_filter(
            self._search_edit.text() if hasattr(self, '_search_edit') else "",
            self._type_filter.currentIndex() if hasattr(self, '_type_filter') else 0,
        )

    def _on_single_click(self, item, col):
        """Affiche le panneau de prévisualisation au clic simple."""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path or not os.path.isfile(path):
            self._preview_panel.setFixedHeight(0)
            return

        ext  = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        size = os.path.getsize(path)
        size_str = f"{size:,} octets" if size < 1024 else f"{size//1024} Ko"

        self._preview_name.setText(name)
        self._preview_info.setText(f"{ext.upper()[1:]}  ·  {size_str}")

        # Thumbnail
        if path in self._thumbnail_cache:
            self._preview_thumb.setPixmap(self._thumbnail_cache[path])
        else:
            self._preview_thumb.setText(_EXT_ICONS.get(ext, "📄"))
            self._preview_thumb.setStyleSheet(
                "background:#0c0e18; border:1px solid #13162a;"
                "border-radius:3px; font:28px; color:#252840;"
            )
            if ext in ('.st', '.glsl', '.trans'):
                self.thumbnail_requested.emit(path)

        self._preview_panel.setFixedHeight(88)

    # ── v2.2 — Thumbnails ────────────────────────────────────────────────────

    def set_thumbnail(self, shader_path: str, pixmap: "QPixmap"):
        if pixmap.width() != self.THUMB_W or pixmap.height() != self.THUMB_H:
            pixmap = pixmap.scaled(
                self.THUMB_W, self.THUMB_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._thumbnail_cache[shader_path] = pixmap

        # Mettre à jour la preview si ce fichier est sélectionné
        if hasattr(self, '_preview_thumb'):
            sel = self._tree.currentItem()
            if sel and sel.data(0, Qt.ItemDataRole.UserRole) == shader_path:
                self._preview_thumb.setPixmap(pixmap)
                self._preview_thumb.setStyleSheet(
                    "background:#0c0e18; border:1px solid #13162a; border-radius:3px;"
                )

        # Mettre à jour l'icône dans l'arbre
        it = self._tree.invisibleRootItem()
        for root_idx in range(it.childCount()):
            root_child = it.child(root_idx)
            for file_idx in range(root_child.childCount()):
                item = root_child.child(file_idx)
                if item.data(0, Qt.ItemDataRole.UserRole) == shader_path:
                    from PyQt6.QtGui import QIcon
                    item.setIcon(0, QIcon(pixmap))
                    return

    def request_thumbnail(self, shader_path: str):
        self.thumbnail_requested.emit(shader_path)

    def _on_double_click(self, item, col):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path or not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.st', '.glsl', '.trans'):
            self.shader_file_requested.emit(path)
        elif ext in ('.wav', '.mp3', '.ogg'):
            self.audio_file_requested.emit(path)

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir",  "",
            "Shaders & Audio (*.st *.glsl *.trans *.wav *.mp3 *.ogg)")
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.st', '.glsl', '.trans'):
                self.shader_file_requested.emit(path)
            elif ext in ('.wav', '.mp3', '.ogg'):
                self.audio_file_requested.emit(path)




    # ── Params Tab (Auto-paramétrage IA) ─────────────────────────────────────

    def _build_params_tab(self) -> QWidget:
        """Construit l'onglet ⚙ Params — détection IA des uniforms exposables."""
        outer = QWidget()
        outer.setStyleSheet("background:#111318;")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── En-tête ──────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet("background:#111318; border-bottom:1px solid #1a1d2a;")
        hdr_l = QVBoxLayout(hdr)
        hdr_l.setContentsMargins(8, 8, 8, 6)
        hdr_l.setSpacing(4)

        title_row = QHBoxLayout()
        lbl_title = QLabel("AUTO-PARAMS IA")
        lbl_title.setStyleSheet(
            "color:#4a6fa5; font:bold 9px 'Segoe UI'; letter-spacing:1px;"
        )
        title_row.addWidget(lbl_title)
        title_row.addStretch()

        # Badge compteur
        self._params_count_badge = QLabel("0")
        self._params_count_badge.setFixedSize(18, 18)
        self._params_count_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._params_count_badge.setStyleSheet(
            "background:#1a2a3a; color:#4a7ab5; border:1px solid #2a3a5a;"
            " border-radius:9px; font:bold 8px 'Segoe UI';"
        )
        title_row.addWidget(self._params_count_badge)
        hdr_l.addLayout(title_row)

        # Description
        lbl_desc = QLabel("Analyse le shader et détecte\nles paramètres exposables.")
        lbl_desc.setStyleSheet("color:#3a3f58; font:8px 'Segoe UI';")
        hdr_l.addWidget(lbl_desc)

        # Bouton Scan
        self._btn_scan = QPushButton("⟳  Scanner le shader")
        self._btn_scan.setFixedHeight(26)
        self._btn_scan.setStyleSheet(
            "QPushButton { background:#1a2a3a; color:#5080b8; border:1px solid #2a3a5a;"
            " border-radius:4px; font:bold 9px 'Segoe UI'; }"
            "QPushButton:hover { background:#1e3248; color:#80b0e8; border-color:#3a5a9a; }"
            "QPushButton:pressed { background:#162030; }"
        )
        self._btn_scan.clicked.connect(self._on_scan_requested)
        hdr_l.addWidget(self._btn_scan)

        outer_layout.addWidget(hdr)

        # ── Zone scrollable des params ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea{border:none;background:#111318;}"
            "QScrollBar:vertical{background:#111318;width:5px;}"
            "QScrollBar::handle:vertical{background:#2a2d3a;border-radius:2px;}"
        )

        self._params_container = QWidget()
        self._params_container.setStyleSheet("background:#111318;")
        self._params_list_layout = QVBoxLayout(self._params_container)
        self._params_list_layout.setContentsMargins(5, 6, 5, 8)
        self._params_list_layout.setSpacing(2)
        self._params_list_layout.addStretch()

        scroll.setWidget(self._params_container)
        outer_layout.addWidget(scroll, 1)

        # ── Pied de page — bouton Tout appliquer ─────────────────────────
        footer = QWidget()
        footer.setStyleSheet("background:#0e1016; border-top:1px solid #1a1d2a;")
        footer_l = QHBoxLayout(footer)
        footer_l.setContentsMargins(8, 5, 8, 5)

        self._btn_apply_all = QPushButton("✦  Appliquer tous")
        self._btn_apply_all.setFixedHeight(24)
        self._btn_apply_all.setEnabled(False)
        self._btn_apply_all.setStyleSheet(
            "QPushButton { background:#162a16; color:#3a6a3a; border:1px solid #2a4a2a;"
            " border-radius:4px; font:bold 8px 'Segoe UI'; }"
            "QPushButton:enabled { color:#5dd88a; border-color:#3a6a3a; background:#1a3a1a; }"
            "QPushButton:hover:enabled { background:#223222; border-color:#4a8a4a; color:#80f8a0; }"
            "QPushButton:disabled { color:#2a3a2a; border-color:#1a2a1a; }"
        )
        self._btn_apply_all.clicked.connect(self._on_apply_all_params)
        footer_l.addWidget(self._btn_apply_all)

        btn_clear = QPushButton("✕")
        btn_clear.setFixedSize(24, 24)
        btn_clear.setStyleSheet(
            "QPushButton { background:#1a1c24; color:#3a3f58; border:1px solid #22252e;"
            " border-radius:4px; font:bold 9px 'Segoe UI'; }"
            "QPushButton:hover { background:#2a1020; color:#c06070; border-color:#503040; }"
        )
        btn_clear.setToolTip("Effacer les paramètres détectés")
        btn_clear.clicked.connect(self._clear_params)
        footer_l.addWidget(btn_clear)

        outer_layout.addWidget(footer)

        return outer

    def _on_scan_requested(self):
        """Le bouton Scan a été cliqué : demande à main_window le shader courant."""
        self._btn_scan.setText("⟳  Scan en cours…")
        self._btn_scan.setEnabled(False)
        self.params_scan_requested.emit()

    def scan_shader(self, glsl: str):
        """
        API publique — appelée par main_window avec le code du shader courant.
        Lance la détection et peuple le panneau.
        """
        self._btn_scan.setText("⟳  Scanner le shader")
        self._btn_scan.setEnabled(True)

        params = self._ai_detector.detect(glsl)
        self._ai_params = params
        self._rebuild_params_list(params)

    def _rebuild_params_list(self, params: list[ShaderParam]):
        """Reconstruit dynamiquement la liste des paramètres dans le panneau."""
        self._param_sliders.clear()
        self._param_spins.clear()

        # Vider le layout existant (sauf le stretch final)
        while self._params_list_layout.count() > 1:
            item = self._params_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Badge
        self._params_count_badge.setText(str(len(params)))
        self._btn_apply_all.setEnabled(len(params) > 0)

        if not params:
            lbl_empty = QLabel("Aucun paramètre détecté.\nCliquez sur Scanner.")
            lbl_empty.setStyleSheet(
                "color:#3a3f58; font:9px 'Segoe UI'; padding:12px;"
            )
            lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._params_list_layout.insertWidget(0, lbl_empty)
            return

        # Regrouper par catégorie
        from collections import defaultdict
        by_cat: dict[str, list[ShaderParam]] = defaultdict(list)
        for p in params:
            by_cat[p.category or "misc"].append(p)

        CAT_LABELS = {
            "animation": "⏱ Animation",
            "color":     "🎨 Couleur",
            "geometry":  "◻ Géométrie",
            "audio":     "♪ Audio",
            "misc":      "⚙ Divers",
        }

        insert_idx = 0
        for cat_key in ["animation", "color", "geometry", "audio", "misc"]:
            cat_params = by_cat.get(cat_key, [])
            if not cat_params:
                continue

            # En-tête catégorie
            cat_lbl = QLabel(CAT_LABELS.get(cat_key, cat_key.upper()))
            cat_lbl.setStyleSheet(
                "color:#3a3f58; font:bold 8px 'Segoe UI';"
                " padding:4px 6px 2px 6px; background:#0e1016;"
            )
            self._params_list_layout.insertWidget(insert_idx, cat_lbl)
            insert_idx += 1

            for param in cat_params:
                widget = self._build_param_row(param)
                self._params_list_layout.insertWidget(insert_idx, widget)
                insert_idx += 1

    def _build_param_row(self, param: ShaderParam) -> QWidget:
        """Construit un widget de ligne pour un paramètre détecté."""
        row = QWidget()
        row.setStyleSheet(
            "background:#13151d; border:1px solid #1a1d28;"
            " border-radius:4px; margin:1px 0;"
        )
        rl = QVBoxLayout(row)
        rl.setContentsMargins(8, 5, 8, 5)
        rl.setSpacing(3)

        # ── Ligne titre ───────────────────────────────────────────────────
        title_row = QHBoxLayout()

        # Source badge
        src_color = "#2a4a2a" if param.source == "declared" else "#2a2a4a"
        src_text  = "déclaré" if param.source == "declared" else "auto"
        src_lbl = QLabel(src_text)
        src_lbl.setFixedSize(40, 14)
        src_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        src_lbl.setStyleSheet(
            f"background:{src_color}; color:#7080a0; border-radius:2px;"
            " font:7px 'Segoe UI';"
        )
        title_row.addWidget(src_lbl)

        # Nom de l'uniform
        name_lbl = QLabel(f"<b>{param.name}</b>")
        name_lbl.setStyleSheet("color:#8898c0; font:9px 'Segoe UI';")
        title_row.addWidget(name_lbl)
        title_row.addStretch()

        # Bouton Appliquer (seulement pour magic numbers)
        if param.source == "magic_number":
            btn_apply = QPushButton("→ Injecter")
            btn_apply.setFixedSize(60, 16)
            btn_apply.setStyleSheet(
                "QPushButton { background:#1a2a1a; color:#3a7a4a; border:1px solid #2a4a2a;"
                " border-radius:3px; font:7px 'Segoe UI'; }"
                "QPushButton:hover { color:#70d890; border-color:#3a7a3a; background:#1e3a1e; }"
            )
            btn_apply.setToolTip(
                f"Transforme {param.original} en uniform {param.name}"
            )
            btn_apply.clicked.connect(lambda checked, p=param: self.apply_param_to_shader.emit(p))
            title_row.addWidget(btn_apply)

        rl.addLayout(title_row)

        # ── Slider + valeur ───────────────────────────────────────────────
        ctrl_row = QHBoxLayout()

        label = QLabel(param.label)
        label.setStyleSheet("color:#505878; font:8px 'Segoe UI'; min-width:55px;")
        ctrl_row.addWidget(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        steps  = max(1, int((param.max_val - param.min_val) / param.step))
        slider.setRange(0, steps)
        def_pos = max(0, min(steps, int((param.default - param.min_val) / param.step)))
        slider.setValue(def_pos)
        slider.setStyleSheet(_SLIDER_STYLE)
        ctrl_row.addWidget(slider, 1)

        spin = QDoubleSpinBox()
        spin.setRange(param.min_val, param.max_val)
        spin.setSingleStep(param.step)
        spin.setValue(param.default)
        spin.setDecimals(3 if param.step < 0.01 else (2 if param.step < 0.1 else 1))
        spin.setFixedWidth(62)
        spin.setFixedHeight(18)
        spin.setStyleSheet(_SPIN_STYLE)
        ctrl_row.addWidget(spin)

        rl.addLayout(ctrl_row)

        # ── Contexte (magic number uniquement) ───────────────────────────
        if param.source == "magic_number" and param.context:
            ctx_lbl = QLabel(f"← {param.context[:60]}")
            ctx_lbl.setStyleSheet(
                "color:#2a3050; font:7px 'Consolas', monospace;"
                " padding-left:4px;"
            )
            ctx_lbl.setWordWrap(True)
            rl.addWidget(ctx_lbl)

        # ── Connexions slider ↔ spin ──────────────────────────────────────
        def on_slider(v, p=param, sp=spin):
            val = p.min_val + v * p.step
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
            self.uniform_value_changed.emit(p.name, float(val))

        def on_spin(v, p=param, sl=slider):
            pos = max(0, min(steps, int((v - p.min_val) / p.step)))
            sl.blockSignals(True)
            sl.setValue(pos)
            sl.blockSignals(False)
            self.uniform_value_changed.emit(p.name, float(v))

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

        self._param_sliders[param.name] = slider
        self._param_spins[param.name]   = spin

        # Émettre la valeur par défaut immédiatement pour initialiser le shader
        self.uniform_value_changed.emit(param.name, float(param.default))

        return row

    def _on_apply_all_params(self):
        """Émet apply_param_to_shader pour tous les magic numbers non encore déclarés."""
        for param in self._ai_params:
            if param.source == "magic_number":
                self.apply_param_to_shader.emit(param)

    def _clear_params(self):
        """Efface la liste des paramètres détectés."""
        self._ai_params = []
        self._rebuild_params_list([])


    # ── Export Tab ────────────────────────────────────────────────────────────

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:#12141a;}"
                             "QScrollBar:vertical{background:#12141a;width:6px;}"
                             "QScrollBar::handle:vertical{background:#2a2d3a;border-radius:3px;}")

        inner = QWidget()
        inner.setStyleSheet("background:#12141a;")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── Helpers UI locaux ─────────────────────────────────────────────────
        def section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet("color:#3a3f58; font:bold 8px 'Segoe UI';"
                              "padding:6px 0 2px 0;")
            return lbl

        def row_label(text):
            l = QLabel(text)
            l.setStyleSheet("color:#7880a0; font:9px 'Segoe UI'; min-width:80px;")
            return l

        spin_style = ("QDoubleSpinBox,QSpinBox{background:#12141a;color:#c8ccd8;"
                      "border:1px solid #2a2d3a;border-radius:3px;padding:2px 4px;"
                      "font:9px 'Segoe UI';}"
                      "QDoubleSpinBox::up-button,QDoubleSpinBox::down-button,"
                      "QSpinBox::up-button,QSpinBox::down-button"
                      "{background:#1a1c24;border:none;width:13px;}")
        combo_style = ("QComboBox{background:#12141a;color:#c8ccd8;border:1px solid #2a2d3a;"
                       "border-radius:3px;padding:2px 6px;font:9px 'Segoe UI';}"
                       "QComboBox::drop-down{border:none;width:16px;}"
                       "QComboBox QAbstractItemView{background:#1a1c24;color:#c8ccd8;"
                       "border:1px solid #2a2d3a;selection-background-color:#2a3a5a;}")

        # ── Format ────────────────────────────────────────────────────────────
        lay.addWidget(section("FORMAT"))

        self._exp_format_btns = {}
        fmt_row = QHBoxLayout(); fmt_row.setSpacing(4)
        for fmt, icon, tooltip in [
            ("mp4",  "🎬 MP4",  "H.264/AAC — Meilleure compatibilité"),
            ("webm", "🌐 WebM", "VP9/Opus — Web & Open source"),
            ("gif",  "🖼 GIF",  "Animé — Sans audio, 256 couleurs"),
        ]:
            btn = QPushButton(icon)
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(_EXP_FMT_BTN)
            self._exp_format_btns[fmt] = btn
            fmt_row.addWidget(btn)
        self._exp_format_btns["mp4"].setChecked(True)

        # Groupe exclusif
        for fmt, btn in self._exp_format_btns.items():
            btn.clicked.connect(lambda checked, f=fmt: self._on_fmt_select(f))
        lay.addLayout(fmt_row)

        # ── Résolution ────────────────────────────────────────────────────────
        lay.addWidget(section("RÉSOLUTION"))

        self._exp_preset_combo = QComboBox()
        self._exp_preset_combo.setStyleSheet(combo_style)
        for label in ["1920×1080 (Full HD)", "1280×720 (HD)", "854×480 (480p)",
                      "3840×2160 (4K)", "Personnalisé…"]:
            self._exp_preset_combo.addItem(label)
        lay.addWidget(self._exp_preset_combo)

        res_row = QHBoxLayout(); res_row.setSpacing(4)
        self._exp_w = QSpinBox(); self._exp_w.setRange(16, 7680)
        self._exp_w.setValue(1920); self._exp_w.setStyleSheet(spin_style)
        self._exp_w.setFixedHeight(22)
        self._exp_h = QSpinBox(); self._exp_h.setRange(16, 4320)
        self._exp_h.setValue(1080); self._exp_h.setStyleSheet(spin_style)
        self._exp_h.setFixedHeight(22)
        lbl_x = QLabel("×"); lbl_x.setStyleSheet("color:#505470;font:10px;")
        res_row.addWidget(self._exp_w); res_row.addWidget(lbl_x)
        res_row.addWidget(self._exp_h)
        lay.addLayout(res_row)

        # Sync preset → résolution
        def _on_preset(idx):
            presets = [(1920,1080),(1280,720),(854,480),(3840,2160)]
            if idx < len(presets):
                self._exp_w.setValue(presets[idx][0])
                self._exp_h.setValue(presets[idx][1])
        self._exp_preset_combo.currentIndexChanged.connect(_on_preset)

        # ── FPS & Durée ───────────────────────────────────────────────────────
        lay.addWidget(section("SÉQUENCE"))

        fps_row = QHBoxLayout(); fps_row.setSpacing(8)
        fps_row.addWidget(row_label("FPS :"))
        self._exp_fps = QSpinBox()
        self._exp_fps.setRange(1, 120); self._exp_fps.setValue(60)
        self._exp_fps.setStyleSheet(spin_style); self._exp_fps.setFixedHeight(22)
        self._exp_fps.setFixedWidth(56)
        fps_row.addWidget(self._exp_fps); fps_row.addStretch()
        lay.addLayout(fps_row)

        dur_row = QHBoxLayout(); dur_row.setSpacing(8)
        dur_row.addWidget(row_label("Durée :"))
        self._exp_dur = QDoubleSpinBox()
        self._exp_dur.setRange(0.1, 3600); self._exp_dur.setValue(60.0)
        self._exp_dur.setSuffix(" s"); self._exp_dur.setDecimals(1)
        self._exp_dur.setStyleSheet(spin_style); self._exp_dur.setFixedHeight(22)
        self._exp_dur.setFixedWidth(74)
        self._exp_btn_dur_tl = QPushButton("⟳ timeline")
        self._exp_btn_dur_tl.setFixedHeight(22)
        self._exp_btn_dur_tl.setStyleSheet(_BTN_STYLE)
        self._exp_btn_dur_tl.setToolTip("Utiliser la durée de la timeline")
        # La connexion est faite depuis main_window via set_export_duration
        self._exp_btn_dur_tl.clicked.connect(
            lambda: self._exp_dur.setValue(self._exp_dur.value()))  # placeholder overridé
        dur_row.addWidget(self._exp_dur)
        dur_row.addWidget(self._exp_btn_dur_tl); dur_row.addStretch()
        lay.addLayout(dur_row)

        # ── Qualité ───────────────────────────────────────────────────────────
        self._exp_quality_group = QWidget()
        self._exp_quality_group.setStyleSheet("background:transparent;")
        qlay = QVBoxLayout(self._exp_quality_group)
        qlay.setContentsMargins(0, 0, 0, 0); qlay.setSpacing(4)
        qlay.addWidget(section("QUALITÉ VIDÉO"))

        q_row = QHBoxLayout(); q_row.setSpacing(8)
        q_row.addWidget(row_label("CRF / Qualité :"))
        self._exp_crf = QSpinBox()
        self._exp_crf.setRange(0, 63); self._exp_crf.setValue(18)
        self._exp_crf.setStyleSheet(spin_style); self._exp_crf.setFixedHeight(22)
        self._exp_crf.setFixedWidth(50)
        self._exp_crf.setToolTip("0=lossless, 18=haute qualité, 28=bonne compression\n"
                                 "Valeur plus basse = meilleure qualité + fichier plus lourd")
        q_row.addWidget(self._exp_crf); q_row.addStretch()
        qlay.addLayout(q_row)

        lbl_crf_hint = QLabel("0 = lossless  ·  18 = HQ  ·  28 = compact")
        lbl_crf_hint.setStyleSheet("color:#303450; font:italic 8px 'Segoe UI';")
        qlay.addWidget(lbl_crf_hint)
        lay.addWidget(self._exp_quality_group)

        # Qualité GIF
        self._exp_gif_group = QWidget()
        self._exp_gif_group.setStyleSheet("background:transparent;")
        glay = QVBoxLayout(self._exp_gif_group)
        glay.setContentsMargins(0,0,0,0); glay.setSpacing(4)
        glay.addWidget(section("OPTIONS GIF"))
        gif_fps_row = QHBoxLayout(); gif_fps_row.setSpacing(8)
        gif_fps_row.addWidget(row_label("FPS GIF :"))
        self._exp_gif_fps = QSpinBox()
        self._exp_gif_fps.setRange(1, 50); self._exp_gif_fps.setValue(15)
        self._exp_gif_fps.setStyleSheet(spin_style); self._exp_gif_fps.setFixedHeight(22)
        self._exp_gif_fps.setFixedWidth(50)
        self._exp_gif_fps.setToolTip("GIF max ~50fps pratique, 15fps recommandé")
        gif_fps_row.addWidget(self._exp_gif_fps); gif_fps_row.addStretch()
        glay.addLayout(gif_fps_row)
        gif_loop_row = QHBoxLayout(); gif_loop_row.setSpacing(8)
        gif_loop_row.addWidget(row_label("Boucle :"))
        self._exp_gif_loop = QComboBox()
        self._exp_gif_loop.setStyleSheet(combo_style)
        self._exp_gif_loop.addItems(["Infinie (0)", "1 fois", "2 fois", "5 fois"])
        gif_loop_row.addWidget(self._exp_gif_loop); gif_loop_row.addStretch()
        glay.addLayout(gif_loop_row)
        lay.addWidget(self._exp_gif_group)
        self._exp_gif_group.hide()

        # ── Audio ─────────────────────────────────────────────────────────────
        self._exp_audio_group = QWidget()
        self._exp_audio_group.setStyleSheet("background:transparent;")
        alay = QVBoxLayout(self._exp_audio_group)
        alay.setContentsMargins(0,0,0,0); alay.setSpacing(4)
        alay.addWidget(section("AUDIO"))
        self._exp_audio_check = QCheckBox("Inclure l'audio")
        self._exp_audio_check.setChecked(True)
        self._exp_audio_check.setStyleSheet(
            "QCheckBox{color:#8890b0;font:9px 'Segoe UI';}"
            "QCheckBox::indicator{width:13px;height:13px;background:#1e2030;"
            "border:1px solid #2a2d3a;border-radius:3px;}"
            "QCheckBox::indicator:checked{background:#3a5888;border-color:#5080c0;}")
        alay.addWidget(self._exp_audio_check)
        lay.addWidget(self._exp_audio_group)

        # ── Séparateur ────────────────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#1e2030; max-height:1px; margin:4px 0;")
        lay.addWidget(sep)

        # ── Statut FFmpeg ─────────────────────────────────────────────────────
        ffmpeg_ok = bool(shutil.which("ffmpeg"))
        self._ffmpeg_status = QLabel()
        if ffmpeg_ok:
            self._ffmpeg_status.setText("✓ FFmpeg détecté")
            self._ffmpeg_status.setStyleSheet("color:#4a9a5a; font:bold 8px 'Segoe UI';")
        else:
            self._ffmpeg_status.setText("⚠ FFmpeg introuvable — requis pour MP4/WebM/GIF")
            self._ffmpeg_status.setWordWrap(True)
            self._ffmpeg_status.setStyleSheet("color:#c07030; font:8px 'Segoe UI';")
        lay.addWidget(self._ffmpeg_status)

        if not ffmpeg_ok:
            btn_dl = QPushButton("📥 Télécharger FFmpeg…")
            btn_dl.setFixedHeight(22)
            btn_dl.setStyleSheet(_BTN_STYLE)
            btn_dl.clicked.connect(lambda: self._open_ffmpeg_page())
            lay.addWidget(btn_dl)

        # ── Barre de progression ──────────────────────────────────────────────
        self._exp_progress_bar = QLabel("")
        self._exp_progress_bar.setFixedHeight(18)
        self._exp_progress_bar.setStyleSheet(
            "background:#1a1c24; color:#5dd88a; font:bold 9px 'Segoe UI';"
            "border-radius:3px; padding:0 6px;")
        self._exp_progress_bar.hide()
        lay.addWidget(self._exp_progress_bar)

        self._exp_progress_detail = QLabel("")
        self._exp_progress_detail.setStyleSheet("color:#404860; font:8px 'Segoe UI';")
        self._exp_progress_detail.setWordWrap(True)
        self._exp_progress_detail.hide()
        lay.addWidget(self._exp_progress_detail)

        lay.addStretch()

        # ── Bouton Exporter ───────────────────────────────────────────────────
        self._exp_btn = QPushButton("⬇  Exporter la vidéo")
        self._exp_btn.setFixedHeight(32)
        self._exp_btn.setStyleSheet(_BTN_EXPORT_STYLE)
        self._exp_btn.clicked.connect(self._on_export_click)
        lay.addWidget(self._exp_btn)

        scroll.setWidget(inner)
        root_lay = QVBoxLayout(w)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        root_lay.addWidget(scroll)

        return w

    def _on_fmt_select(self, fmt: str):
        for f, b in self._exp_format_btns.items():
            b.setChecked(f == fmt)
        # Afficher/masquer sections selon le format
        is_gif  = fmt == "gif"
        is_vid  = fmt in ("mp4", "webm")
        self._exp_quality_group.setVisible(is_vid)
        self._exp_gif_group.setVisible(is_gif)
        self._exp_audio_group.setVisible(is_vid)
        # Changer libellé bouton
        labels = {"mp4": "⬇  Exporter en MP4", "webm": "⬇  Exporter en WebM",
                  "gif": "⬇  Exporter en GIF"}
        self._exp_btn.setText(labels.get(fmt, "⬇  Exporter"))

    def _open_ffmpeg_page(self):
        webbrowser.open("https://ffmpeg.org/download.html")

    def get_export_params(self) -> dict:
        """Retourne les paramètres d'export courants."""
        fmt = next((f for f, b in self._exp_format_btns.items() if b.isChecked()), "mp4")
        loop_map = {"Infinie (0)": 0, "1 fois": 1, "2 fois": 2, "5 fois": 5}
        return {
            "format":      fmt,
            "width":       self._exp_w.value(),
            "height":      self._exp_h.value(),
            "fps":         self._exp_fps.value(),
            "duration":    self._exp_dur.value(),
            "crf":         self._exp_crf.value(),
            "gif_fps":     self._exp_gif_fps.value(),
            "gif_loop":    loop_map.get(self._exp_gif_loop.currentText(), 0),
            "include_audio": self._exp_audio_check.isChecked(),
        }

    def set_export_duration(self, dur: float):
        """Synchronise la durée depuis la timeline."""
        self._exp_dur.setValue(dur)

    def set_export_progress(self, pct: float, detail: str = ""):
        """Met à jour la barre de progression (0.0–1.0). pct=-1 = masque."""
        if pct < 0:
            self._exp_progress_bar.hide()
            self._exp_progress_detail.hide()
            self._exp_btn.setEnabled(True)
            self._exp_btn.setText(self._exp_btn.text().replace(" ⏳", ""))
        else:
            filled = int(pct * 20)
            bar = "█" * filled + "░" * (20 - filled)
            self._exp_progress_bar.setText(f"{bar}  {int(pct*100)}%")
            self._exp_progress_bar.show()
            if detail:
                self._exp_progress_detail.setText(detail)
                self._exp_progress_detail.show()

    def _on_export_click(self):
        params = self.get_export_params()
        self._exp_btn.setEnabled(False)
        self.export_requested.emit(params)

    # ── Info Tab ─────────────────────────────────────────────────────────────

    def _build_info_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        c = QWidget()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        grp_sh = QGroupBox("Shader actif")
        grp_sh.setStyleSheet(_GRP_STYLE)
        fi = QFormLayout(grp_sh)
        fi.setContentsMargins(8, 12, 8, 8)
        self._lbl_shader_name = QLabel("—")
        self._lbl_shader_type = QLabel("—")
        self._lbl_compile_st  = QLabel("—")
        for lbl in (self._lbl_shader_name, self._lbl_shader_type, self._lbl_compile_st):
            lbl.setStyleSheet("color:#c8ccd8; font:8px 'Segoe UI';")
        fi.addRow("Fichier :", self._lbl_shader_name)
        fi.addRow("Format :",  self._lbl_shader_type)
        fi.addRow("Statut :",  self._lbl_compile_st)
        lay.addWidget(grp_sh)

        grp_fps = QGroupBox("Performance")
        grp_fps.setStyleSheet(_GRP_STYLE)
        fp = QFormLayout(grp_fps)
        fp.setContentsMargins(8, 12, 8, 8)
        self._lbl_fps = QLabel("—")
        self._lbl_fps.setStyleSheet("color:#5dd88a; font:bold 10px 'Segoe UI';")
        fp.addRow("FPS :", self._lbl_fps)
        lay.addWidget(grp_fps)

        grp_tex = QGroupBox("Textures (iChannel)")
        grp_tex.setStyleSheet(_GRP_STYLE)
        tl = QFormLayout(grp_tex)
        tl.setContentsMargins(8, 12, 8, 8)
        self._tex_labels = []
        for i in range(4):
            lbl = QLabel("—")
            lbl.setStyleSheet("color:#505470; font:8px 'Segoe UI';")
            tl.addRow(f"iChannel{i} :", lbl)
            self._tex_labels.append(lbl)
        lay.addWidget(grp_tex)
        lay.addStretch()
        scroll.setWidget(c)
        return scroll

    # ── Public API ────────────────────────────────────────────────────────────

    def update_shader_info(self, name: str, stype: str, ok: bool, error: str = ""):
        self._lbl_shader_name.setText(name or "—")
        self._lbl_shader_type.setText(stype.upper() if stype else "—")
        if ok:
            self._lbl_compile_st.setText("✓ OK")
            self._lbl_compile_st.setStyleSheet("color:#5dd88a; font:8px 'Segoe UI';")
        else:
            self._lbl_compile_st.setText("✗ Erreur")
            self._lbl_compile_st.setStyleSheet("color:#e06060; font:8px 'Segoe UI';")

    def update_fps(self, fps: float):
        self._lbl_fps.setText(f"{fps:.1f}")

    def update_texture_label(self, channel: int, name: str):
        if 0 <= channel < 4:
            self._tex_labels[channel].setText(name)
            self._tex_labels[channel].setStyleSheet("color:#5dd88a; font:8px 'Segoe UI';")


# ── FileTreeWidget ─────────────────────────────────────────────────────────────

_EXT_ICONS = {
    '.st':    '🎨',
    '.glsl':  '💡',
    '.trans': '🔀',
    '.wav':   '🔊',
    '.mp3':   '🎵',
    '.ogg':   '🎶',
}

class FileTreeWidget(QTreeWidget):
    def __init__(self, left_panel, parent=None):
        super().__init__(parent)
        self._lp = left_panel

    def startDrag(self, actions):
        item = self.currentItem()
        if not item: return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path or not os.path.isfile(path): return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(path)])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(actions)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if not item: return
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_STYLE)

        path = item.data(0, Qt.ItemDataRole.UserRole)

        if path and os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()

            if ext in ('.st', '.glsl', '.trans'):
                act_open = menu.addAction("Ouvrir dans l'éditeur")
                act_open.triggered.connect(lambda: self._lp.shader_file_requested.emit(path))
                menu.addSeparator()
            elif ext in ('.wav', '.mp3', '.ogg'):
                act_load = menu.addAction("Charger l'audio")
                act_load.triggered.connect(lambda: self._lp.audio_file_requested.emit(path))
                menu.addSeparator()

            act_rename = menu.addAction("Renommer…")
            act_rename.triggered.connect(lambda: self._rename_file(item, path))

            act_dup = menu.addAction("Dupliquer")
            act_dup.triggered.connect(lambda: self._duplicate_file(path))

            menu.addSeparator()

            act_reveal = menu.addAction("Révéler dans l'explorateur")
            act_reveal.triggered.connect(lambda: self._reveal_file(path))

            menu.addSeparator()

            act_del = menu.addAction("Supprimer")
            act_del.triggered.connect(lambda: self._delete_file(item, path))

        else:
            folder = item.data(0, Qt.ItemDataRole.UserRole + 1)
            exts   = item.data(0, Qt.ItemDataRole.UserRole + 2)
            if folder:
                menu.addAction("Ouvrir le dossier").triggered.connect(
                    lambda: self._reveal_file(folder))
                if exts:
                    menu.addSeparator()
                    for ext_str, label in [
                        ('.st',    'Nouveau shader Shadertoy…'),
                        ('.glsl',  'Nouveau shader GLSL…'),
                        ('.trans', 'Nouvelle transition…'),
                    ]:
                        if ext_str in exts:
                            menu.addAction(label).triggered.connect(
                                lambda checked=False, f=folder, e=ext_str: self._new_file(f, e))

        if menu.actions():
            menu.exec(event.globalPos())

    # ── Actions ──────────────────────────────────────────────────────────────

    def _rename_file(self, item, path: str):
        old_name = os.path.basename(path)
        new_name, ok = QInputDialog.getText(
            self, "Renommer", "Nouveau nom :", text=old_name)
        if not ok or not new_name or new_name == old_name:
            return
        new_path = os.path.join(os.path.dirname(path), new_name)
        try:
            os.rename(path, new_path)
            item.setText(0, new_name)
            item.setData(0, Qt.ItemDataRole.UserRole, new_path)
        except OSError as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de renommer :\n{e}")

    def _duplicate_file(self, path: str):
        base, ext = os.path.splitext(path)
        new_path = f"{base}_copy{ext}"
        count = 1
        while os.path.exists(new_path):
            new_path = f"{base}_copy{count}{ext}"
            count += 1
        try:
            shutil.copy2(path, new_path)
            self._lp.refresh_tree()
        except OSError as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de dupliquer :\n{e}")

    def _delete_file(self, item, path: str):
        reply = QMessageBox.question(
            self, "Supprimer",
            f"Supprimer définitivement :\n{os.path.basename(path)} ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(path)
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    self.takeTopLevelItem(self.indexOfTopLevelItem(item))
                self._lp.refresh_tree()
            except OSError as e:
                QMessageBox.warning(self, "Erreur", f"Impossible de supprimer :\n{e}")

    def _reveal_file(self, path: str):
        target = path if os.path.isdir(path) else os.path.dirname(path)
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', target])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', target])
        else:
            subprocess.Popen(['xdg-open', target])

    def _new_file(self, folder: str, ext: str):
        templates = {
            '.st':    "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    vec2 uv = fragCoord / iResolution.xy;\n    fragColor = vec4(uv, 0.5 + 0.5 * sin(iTime), 1.0);\n}\n",
            '.glsl':  "#version 330 core\nuniform vec2  uResolution;\nuniform float uTime;\nout vec4 fragColor;\nvoid main() {\n    vec2 uv = gl_FragCoord.xy / uResolution.xy;\n    fragColor = vec4(uv, 0.5 + 0.5 * sin(uTime), 1.0);\n}\n",
            '.trans': "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    vec2 uv = fragCoord / iResolution.xy;\n    vec4 a = texture(iChannel0, uv);\n    vec4 b = texture(iChannel1, uv);\n    fragColor = mix(a, b, iProgress);\n}\n",
        }
        name, ok = QInputDialog.getText(
            self, f"Nouveau fichier {ext}", f"Nom du fichier (sans extension) :")
        if not ok or not name:
            return
        if not name.endswith(ext):
            name += ext
        path = os.path.join(folder, name)
        if os.path.exists(path):
            QMessageBox.warning(self, "Erreur", f"Le fichier existe déjà :\n{name}")
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(templates.get(ext, ""))
            self._lp.refresh_tree()
            self._lp.shader_file_requested.emit(path)
        except OSError as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de créer :\n{e}")
# ── Styles ────────────────────────────────────────────────────────────────────

_TAB_STYLE = """
QTabWidget::pane { background:#1a1c24; border:none; }
QTabBar::tab { background:#111318; color:#505470; padding:5px 12px;
               border:none; font:bold 8px 'Segoe UI'; }
QTabBar::tab:selected { background:#1a1c24; color:#c8ccd8;
                        border-bottom:2px solid #4a6fa5; }
QTabBar::tab:hover:!selected { background:#161820; }
"""

_GRP_STYLE = """
QGroupBox { color:#505470; font:bold 8px 'Segoe UI';
            border:1px solid #222530; border-radius:4px; margin-top:10px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }
"""

_FX_BTN = """
QPushButton { background:#161820; color:#7880a0; border:1px solid #1e2030;
              border-radius:3px; text-align:left; padding:0 8px;
              font:9px 'Segoe UI'; }
QPushButton:hover   { background:#1e2232; color:#c0c8e0; border-color:#2e3448; }
QPushButton:checked { background:#1a2232; color:#9098b8; border-color:#2a3048;
                      font:9px 'Segoe UI'; }
"""

_FX_TOGGLE_OFF = """
QPushButton { background:#1a1c24; color:#3a3f58; border:1px solid #22252e;
              border-radius:3px; font:bold 8px 'Segoe UI'; }
QPushButton:hover { background:#201824; color:#805060; border-color:#402030; }
"""

_FX_TOGGLE_ON = """
QPushButton { background:#1a3a1a; color:#50d870; border:1px solid #2a5a2a;
              border-radius:3px; font:bold 8px 'Segoe UI'; }
QPushButton:hover { background:#1a4a1a; color:#70ff90; border-color:#3a7a3a; }
"""

_FX_OFF_BTN = """
QPushButton { background:#161820; color:#402030; border:1px solid #201820;
              border-radius:3px; padding:0 6px;
              font:8px 'Segoe UI'; }
QPushButton:hover { background:#201020; color:#c06070; border-color:#503040; }
"""

_SPIN_STYLE = """
QDoubleSpinBox { background:#12141a; color:#c8ccd8; border:1px solid #222530;
                 border-radius:3px; padding:1px 3px; font:9px 'Segoe UI'; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button
              { background:#1a1c24; border:none; width:12px; }
"""

_SLIDER_STYLE = """
QSlider::groove:horizontal { background:#1e2030; height:4px; border-radius:2px; }
QSlider::handle:horizontal { background:#3a5888; width:10px; height:10px;
                              margin:-3px 0; border-radius:5px; }
QSlider::sub-page:horizontal { background:#2a4070; border-radius:2px; }
QSlider::handle:horizontal:hover { background:#5080c0; }
"""

_BTN_SAVE_STYLE = """
QPushButton { background:#1a2a1a; color:#5dd88a; border:1px solid #2a4a2a;
              border-radius:4px; font:bold 9px 'Segoe UI'; padding:2px 8px; }
QPushButton:hover  { background:#223222; border-color:#3a6a3a; }
QPushButton:pressed{ background:#2a3a2a; }
"""

_BTN_EXPORT_STYLE = """
QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1e3a6a, stop:1 #1a2a4a);
              color:#80b0e8; border:1px solid #2a4a7a;
              border-radius:5px; font:bold 10px 'Segoe UI'; padding:4px 8px; }
QPushButton:hover  { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                      stop:0 #28489a, stop:1 #223260);
                     color:#a0c8f8; border-color:#3a5a9a; }
QPushButton:pressed{ background:#182038; }
QPushButton:disabled{ background:#141620; color:#303450; border-color:#1e2030; }
"""

_EXP_FMT_BTN = """
QPushButton { background:#16181e; color:#6068a0; border:1px solid #1e2030;
              border-radius:4px; font:bold 9px 'Segoe UI'; padding:4px 2px; }
QPushButton:hover   { background:#1e2232; color:#9098c0; border-color:#2a2d3a; }
QPushButton:checked { background:#1a2a4a; color:#70a0e0; border-color:#2a4a7a;
                      font:bold 9px 'Segoe UI'; }
"""

_BTN_STYLE = """
QPushButton { background:#1a1c24; color:#9098b0; border:1px solid #222530;
              border-radius:3px; font:9px 'Segoe UI'; padding:2px 6px; }
QPushButton:hover { background:#222536; }
"""

_EXP_BTN_PRIMARY = """
QPushButton {
    background: #1a2a60;
    color: #8aacf0;
    border: 1px solid #2a4090;
    border-radius: 4px;
    font: bold 9px 'Segoe UI';
    padding: 3px 10px;
}
QPushButton:hover { background: #203580; color: #b0ccff; border-color: #3a55b8; }
QPushButton:pressed { background: #162050; }
"""

_EXP_BTN_SECONDARY = """
QPushButton {
    background: #0e1020;
    color: #4a5280;
    border: 1px solid #181c30;
    border-radius: 4px;
    font: 9px 'Segoe UI';
    padding: 3px 10px;
}
QPushButton:hover { background: #141828; color: #7080b0; border-color: #242840; }
QPushButton:pressed { background: #0c0e18; }
"""

_EXP_TOOL_BTN = """
QToolButton {
    background: transparent;
    color: #303458;
    border: none;
    border-radius: 3px;
    font: 12px;
    padding: 0px;
}
QToolButton:hover { background: #0e1020; color: #6070a8; }
QToolButton:pressed { background: #0c0e18; }
"""

_EXP_SEARCH_STYLE = """
QLineEdit {
    background: #0a0c14;
    color: #8088b0;
    border: 1px solid #14182a;
    border-radius: 3px;
    font: 9px 'Segoe UI';
    padding: 2px 6px;
    selection-background-color: #1e2a50;
}
QLineEdit:focus { border-color: #1e2a50; color: #a0a8d0; }
QLineEdit::placeholder { color: #252840; }
"""

_EXP_COMBO_STYLE = """
QComboBox {
    background: #0a0c14;
    color: #404870;
    border: 1px solid #14182a;
    border-radius: 3px;
    font: 8px 'Segoe UI';
    padding: 2px 4px;
}
QComboBox:hover { border-color: #1e2a50; color: #6070a0; }
QComboBox::drop-down { border: none; width: 14px; }
QComboBox QAbstractItemView {
    background: #0c0e18;
    color: #7080a8;
    border: 1px solid #14182a;
    selection-background-color: #1e2a50;
    font: 8px 'Segoe UI';
}
"""

_EXP_TREE_STYLE = """
QTreeWidget {
    background: #080a12;
    color: #6070a0;
    border: none;
    font: 9px 'Segoe UI';
    outline: none;
}
QTreeWidget::item {
    padding: 4px 2px;
    border-radius: 3px;
}
QTreeWidget::item:selected {
    background: #141c38;
    color: #a8b4e0;
}
QTreeWidget::item:hover:!selected {
    background: #0c0e1c;
    color: #8090c8;
}
QTreeWidget::branch {
    background: #080a12;
}
QTreeWidget::branch:has-children:!has-siblings:closed,
QTreeWidget::branch:closed:has-children:has-siblings {
    color: #252840;
}
"""

_EXP_DIALOG_STYLE = """
QDialog { background: #0d0f18; }
QLabel  { background: transparent; color: #c0c8e8; }
"""

_TREE_STYLE = """
QTreeWidget { background:#12141a; color:#c0c4d0; border:1px solid #1e2030;
              border-radius:3px; font:9px 'Segoe UI'; outline:none; }
QTreeWidget::item { padding:3px 0; }
QTreeWidget::item:selected { background:#1e2a44; color:#fff; }
QTreeWidget::item:hover { background:#181c2c; }
QTreeWidget::branch { background:#12141a; }
"""

_MENU_STYLE = """
QMenu { background:#1c1e24; color:#c8ccd8; border:1px solid #2a2d3a;
        border-radius:4px; padding:4px; }
QMenu::item { padding:5px 18px; border-radius:3px; }
QMenu::item:selected { background:#2a3050; }
"""

