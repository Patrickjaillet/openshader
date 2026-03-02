"""
asset_store.py
--------------
Asset Store intégré — Shaders communautaires  (v5.0)

Architecture :
  - Index JSON hébergé à distance (GitHub Pages ou serveur dédié)
  - AssetStoreIndex    : fetch + cache + parse de l'index
  - AssetStoreManager  : QObject, threading, signaux Qt
  - AssetPublisher     : dialog de publication (token API)
  - AssetStoreBrowser  : widget PyQt6 Browse / Preview / Import / Publish

Format de l'index (shader_index.json) :
{
  "version": 1,
  "generated": "2025-01-01T00:00:00Z",
  "shaders": [
    {
      "id":          "plasma-nova",
      "name":        "Plasma Nova",
      "description": "Fond plasma coloré animé",
      "author":      "demoscener42",
      "version":     "1.0.0",
      "category":    "backgrounds",     // backgrounds | transitions | post-process | utilities
      "tags":        ["plasma","color","animated"],
      "glsl_url":    "https://raw.githubusercontent.com/.../plasma_nova.glsl",
      "preview_url": "https://raw.githubusercontent.com/.../plasma_nova.png",
      "sha256":      "abc123...",
      "license":     "CC0",
      "rating":      4.5,
      "ratings_count": 22,
      "downloads":   580,
      "updated":     "2025-04-10T12:00:00Z"
    }
  ]
}

Stockage local :
  ~/.openshader/asset_store/cache.json     — cache index
  ~/.openshader/asset_store/ratings.json   — notes locales
  ~/.openshader/asset_store/previews/      — images preview téléchargées

API de publication (POST /shaders) :
  Authorization: Bearer <token>
  body: { name, description, category, tags, glsl_source, preview_base64, license }
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal, Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap, QColor, QFont, QPainter, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QComboBox, QProgressBar,
    QTabWidget, QTextEdit, QDialog, QDialogButtonBox,
    QFileDialog, QSizePolicy, QMessageBox, QCheckBox,
    QGridLayout, QSplitter, QStackedWidget,
)

from .logger import get_logger

log = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

INDEX_URL        = "https://raw.githubusercontent.com/openshader-org/asset-store/main/shader_index.json"
PUBLISH_API_URL  = "https://api.openshader.io/v1/shaders"
CACHE_TTL_S      = 3600   # 1 h

# ── Index embarqué (fallback offline) ─────────────────────────────────────────
# Activé automatiquement quand le fetch distant échoue et qu'il n'existe pas
# encore de cache local. Contient 12 shaders GLSL complets prêts à l'emploi.
_BUILTIN_INDEX: dict = {
    "version": 1,
    "generated": "2026-01-01T00:00:00Z",
    "_source": "builtin",
    "shaders": [
        # ── backgrounds ──────────────────────────────────────────────────────
        {
            "id": "plasma-nova", "name": "Plasma Nova",
            "description": "Fond plasma coloré animé. Classique demoscene.",
            "author": "iq / adapté OpenShader", "version": "1.0.0",
            "category": "backgrounds",
            "tags": ["plasma", "color", "animated", "classic"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.5, "ratings_count": 38, "downloads": 920,
            "updated": "2025-11-01T00:00:00Z",
            "_builtin_glsl": "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    vec2 uv = fragCoord / iResolution.xy;\n    float t = iTime * 0.4;\n    float v = sin(uv.x*8.+t)+sin(uv.y*6.-t*1.3)+sin((uv.x+uv.y)*7.+t*.8)+sin(length(uv-.5)*12.-t*2.);\n    v = v*.25+.5;\n    vec3 col = .5+.5*cos(v*6.28318+vec3(0.,2.094,4.189));\n    fragColor = vec4(col,1.);\n}"
        },
        {
            "id": "starfield", "name": "Starfield Warp",
            "description": "Tunnel étoilé avec effet warp speed. Fond idéal pour intros.",
            "author": "OpenShader Team", "version": "1.1.0",
            "category": "backgrounds",
            "tags": ["stars", "space", "warp", "tunnel", "animated"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.7, "ratings_count": 54, "downloads": 1340,
            "updated": "2025-12-15T00:00:00Z",
            "_builtin_glsl": "float hash(float n){return fract(sin(n)*43758.5453123);}\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord){\n    vec2 uv=(fragCoord-iResolution.xy*.5)/iResolution.y;\n    vec3 col=vec3(0.);\n    float t=iTime*.5;\n    for(int i=0;i<80;i++){\n        float fi=float(i);\n        vec2 pos=vec2(hash(fi*1.7)-.5,hash(fi*2.3)-.5);\n        float z=fract(hash(fi*.9)+t);\n        float sz=(1.-z)*.005;\n        float br=pow(1.-z,3.);\n        col+=br*vec3(1.,.9,.8)*smoothstep(sz,0.,length(uv-pos/z));\n    }\n    fragColor=vec4(col,1.);\n}"
        },
        {
            "id": "voronoi-cells", "name": "Voronoi Cells",
            "description": "Cellules de Voronoi animées avec bords lumineux. Ambiance futuriste.",
            "author": "iq", "version": "1.0.0",
            "category": "backgrounds",
            "tags": ["voronoi", "cells", "geometric", "neon"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.3, "ratings_count": 27, "downloads": 610,
            "updated": "2025-10-05T00:00:00Z",
            "_builtin_glsl": "vec2 hash2(vec2 p){return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);}\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.y*2.5;\n    float t=iTime*.3,md=8.;\n    for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){\n        vec2 cell=floor(uv)+vec2(i,j);\n        vec2 off=hash2(cell);\n        off=.5+.45*sin(t+6.28318*off);\n        md=min(md,length(uv-cell-off));\n    }\n    vec3 col=vec3(.1,.6,1.)*.8*exp(-12.*md)+vec3(0.,.1,.2)*.2;\n    fragColor=vec4(col,1.);\n}"
        },
        {
            "id": "aurora-gradient", "name": "Aurora Gradient",
            "description": "Aurore boréale procédurale avec couches de bruit.",
            "author": "OpenShader Team", "version": "1.0.0",
            "category": "backgrounds",
            "tags": ["aurora", "gradient", "noise", "atmospheric"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.6, "ratings_count": 19, "downloads": 445,
            "updated": "2026-01-10T00:00:00Z",
            "_builtin_glsl": "float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}\nfloat noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    float t=iTime*.1;\n    float n=noise(uv*3.+vec2(t,0.))+.5*noise(uv*6.+vec2(0.,t))+.25*noise(uv*12.-vec2(t*.5));\n    n/=1.75;\n    vec3 c1=vec3(0.,.8,.5),c2=vec3(.1,.2,.9),c3=vec3(0.,.05,.15);\n    vec3 col=mix(c3,mix(c1,c2,uv.x),pow(n*(1.-uv.y*.6),1.5));\n    fragColor=vec4(col,1.);\n}"
        },
        # ── transitions ──────────────────────────────────────────────────────
        {
            "id": "dissolve-noise", "name": "Dissolve Noise",
            "description": "Transition par dissolution avec seuil de bruit. iProgress=[0,1].",
            "author": "OpenShader Team", "version": "1.2.0",
            "category": "transitions",
            "tags": ["dissolve", "noise", "transition", "wipe"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.4, "ratings_count": 31, "downloads": 730,
            "updated": "2025-09-20T00:00:00Z",
            "_builtin_glsl": "// iChannel0=sceneA iChannel1=sceneB iProgress in [0,1]\nfloat hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}\nfloat noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    float p=clamp(iProgress,0.,1.);\n    float n=noise(uv*6.)*.5+noise(uv*12.)*.25+noise(uv*24.)*.125;n/=.875;\n    float mask=smoothstep(p-.05,p+.05,n);\n    fragColor=mix(texture(iChannel1,uv),texture(iChannel0,uv),mask);\n}"
        },
        {
            "id": "wipe-swirl", "name": "Wipe Swirl",
            "description": "Transition spirale centripète. iProgress=[0,1].",
            "author": "OpenShader Team", "version": "1.0.0",
            "category": "transitions",
            "tags": ["swirl", "spiral", "wipe", "transition"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.2, "ratings_count": 14, "downloads": 290,
            "updated": "2025-08-05T00:00:00Z",
            "_builtin_glsl": "// iChannel0=sceneA iChannel1=sceneB iProgress in [0,1]\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    vec2 ctr=uv-.5;\n    float p=clamp(iProgress,0.,1.);\n    float swirl=atan(ctr.y,ctr.x)/(2.*3.14159)+.5+length(ctr)*2.;\n    float mask=smoothstep(p-.05,p+.05,fract(swirl));\n    fragColor=mix(texture(iChannel1,uv),texture(iChannel0,uv),mask);\n}"
        },
        {
            "id": "pixelate-transition", "name": "Pixelate In/Out",
            "description": "Pixelisation croissante puis décroissante. iProgress=[0,1].",
            "author": "OpenShader Team", "version": "1.0.0",
            "category": "transitions",
            "tags": ["pixelate", "transition", "retro", "8bit"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.0, "ratings_count": 11, "downloads": 205,
            "updated": "2025-07-12T00:00:00Z",
            "_builtin_glsl": "// iChannel0=sceneA iChannel1=sceneB iProgress in [0,1]\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    float p=clamp(iProgress,0.,1.);\n    float pix=mix(1.,64.,sin(p*3.14159));\n    vec2 res=iResolution.xy/pix;\n    vec2 uv=floor(fragCoord/iResolution.xy*res)/res+.5/res;\n    fragColor=mix(texture(iChannel0,uv),texture(iChannel1,uv),step(.5,p));\n}"
        },
        # ── post-process ─────────────────────────────────────────────────────
        {
            "id": "chromatic-aberration", "name": "Chromatic Aberration",
            "description": "Aberration chromatique RGB réaliste avec distorsion radiale.",
            "author": "OpenShader Team", "version": "1.3.0",
            "category": "post-process",
            "tags": ["chromatic", "aberration", "rgb", "lens", "post-fx"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.8, "ratings_count": 67, "downloads": 1820,
            "updated": "2025-12-01T00:00:00Z",
            "_builtin_glsl": "// iChannel0=scene\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    vec2 ctr=uv-.5;\n    float amount=.006*dot(ctr,ctr)*4.;\n    vec2 dir=normalize(ctr+.0001)*amount;\n    fragColor=vec4(texture(iChannel0,uv+dir).r,texture(iChannel0,uv).g,texture(iChannel0,uv-dir).b,1.);\n}"
        },
        {
            "id": "crt-scanlines", "name": "CRT Scanlines",
            "description": "Effet moniteur CRT vintage : scanlines, courbure barrel et vignette.",
            "author": "OpenShader Team", "version": "2.0.0",
            "category": "post-process",
            "tags": ["crt", "scanlines", "retro", "vhs", "vintage"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.6, "ratings_count": 44, "downloads": 1110,
            "updated": "2025-11-20T00:00:00Z",
            "_builtin_glsl": "// iChannel0=scene\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    vec2 crt=uv*2.-1.;\n    vec2 off=crt.yx/6.;\n    crt+=crt*off*off;\n    if(clamp(crt,-1.,1.)!=crt){fragColor=vec4(0.);return;}\n    uv=(crt+1.)*.5;\n    vec3 col=texture(iChannel0,uv).rgb;\n    float sl=sin(uv.y*iResolution.y*3.14159)*.5+.5;\n    col*=mix(.6,1.,pow(sl,.25));\n    vec2 vig=uv*(1.-uv);\n    col*=pow(vig.x*vig.y*15.,.2);\n    col+=col*col*.15;\n    fragColor=vec4(col,1.);\n}"
        },
        {
            "id": "bloom-glow", "name": "Bloom / Glow",
            "description": "Bloom multi-sample avec seuil de luminosité configurable.",
            "author": "OpenShader Team", "version": "1.1.0",
            "category": "post-process",
            "tags": ["bloom", "glow", "hdr", "luminance", "post-fx"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.5, "ratings_count": 29, "downloads": 780,
            "updated": "2025-10-30T00:00:00Z",
            "_builtin_glsl": "// iChannel0=scene\nfloat lum(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    vec2 tx=1./iResolution.xy;\n    vec3 base=texture(iChannel0,uv).rgb,bloom=vec3(0.);\n    float w=0.;\n    for(int x=-4;x<=4;x++)for(int y=-4;y<=4;y++){\n        vec3 s=texture(iChannel0,uv+vec2(x,y)*tx*2.).rgb;\n        float wt=exp(-float(x*x+y*y)*.08);\n        bloom+=s*max(0.,lum(s)-.7)*wt;w+=wt;\n    }\n    fragColor=vec4(base+bloom/w*1.5,1.);\n}"
        },
        # ── utilities ────────────────────────────────────────────────────────
        {
            "id": "uv-debugger", "name": "UV Debugger",
            "description": "Affiche les coordonnées UV en couleur + grille de référence.",
            "author": "OpenShader Team", "version": "1.0.0",
            "category": "utilities",
            "tags": ["uv", "debug", "grid", "utility", "diagnostic"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.2, "ratings_count": 22, "downloads": 530,
            "updated": "2025-09-01T00:00:00Z",
            "_builtin_glsl": "void mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    vec2 grid=fract(uv*10.);\n    float line=1.-smoothstep(0.,.04,min(grid.x,grid.y));\n    vec3 col=mix(vec3(uv.x,uv.y,.3),vec3(1.),line*.4);\n    col=mix(col,vec3(1.,.2,.1),smoothstep(.012,.008,length(uv-.5)));\n    fragColor=vec4(col,1.);\n}"
        },
        {
            "id": "color-palette", "name": "Color Palette cos()",
            "description": "Génère des palettes infinies via cos(). Utilitaire iq classique.",
            "author": "iq", "version": "1.0.0",
            "category": "utilities",
            "tags": ["color", "palette", "utility", "math"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.9, "ratings_count": 88, "downloads": 2100,
            "updated": "2025-06-01T00:00:00Z",
            "_builtin_glsl": "vec3 palette(float t,vec3 a,vec3 b,vec3 c,vec3 d){return a+b*cos(6.28318*(c*t+d));}\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=fragCoord/iResolution.xy;\n    float t=iTime*.2;\n    vec3 col;\n    float x=uv.x+t;\n    if(uv.y<.25)      col=palette(x,vec3(.5),vec3(.5),vec3(1.),vec3(0.,.33,.67));\n    else if(uv.y<.5)  col=palette(x,vec3(.5),vec3(.5),vec3(1.),vec3(0.,.10,.20));\n    else if(uv.y<.75) col=palette(x,vec3(.5),vec3(.5),vec3(1.,1.,.5),vec3(.8,.9,.3));\n    else              col=palette(x,vec3(.5),vec3(.5),vec3(2.,1.,0.),vec3(.5,.2,.25));\n    if(fract(uv.y*4.)<.02)col=vec3(0.);\n    fragColor=vec4(col,1.);\n}"
        },
        {
            "id": "sdf-shapes", "name": "SDF Shapes Library",
            "description": "Bibliothèque de fonctions SDF 2D (cercle, boîte, étoile…) avec scène de démo.",
            "author": "iq / OpenShader Team", "version": "1.0.0",
            "category": "utilities",
            "tags": ["sdf", "shapes", "2d", "math", "utility", "library"],
            "glsl_url": "", "preview_url": "", "sha256": "", "license": "CC0",
            "rating": 4.7, "ratings_count": 55, "downloads": 1450,
            "updated": "2025-11-05T00:00:00Z",
            "_builtin_glsl": "float sdCircle(vec2 p,float r){return length(p)-r;}\nfloat sdBox(vec2 p,vec2 b){vec2 d=abs(p)-b;return length(max(d,0.))+min(max(d.x,d.y),0.);}\nfloat sdStar(vec2 p,float r,int n,float m){float an=3.14159/float(n),en=3.14159/m;vec2 acs=vec2(cos(an),sin(an)),ecs=vec2(cos(en),sin(en));float bn=mod(atan(p.y,p.x),2.*an)-an;p=length(p)*vec2(cos(bn),abs(sin(bn)));p-=r*acs;p+=ecs*clamp(-dot(p,ecs),0.,r*acs.y/ecs.y);return length(p)*sign(p.x);}\nvec3 sdfRender(float d,vec3 fill,vec3 bg){return mix(bg,fill,1.-smoothstep(-fwidth(d),fwidth(d),d));}\nvoid mainImage(out vec4 fragColor,in vec2 fragCoord){\n    vec2 uv=(fragCoord-iResolution.xy*.5)/iResolution.y;\n    vec3 col=vec3(.08,.08,.12);\n    col=sdfRender(sdCircle(uv-vec2(.3,.1),.12),vec3(.2,.6,1.),col);\n    col=sdfRender(sdBox(uv-vec2(-.3,.1),vec2(.10,.08)),vec3(1.,.5,.1),col);\n    col=sdfRender(sdStar(uv-vec2(0.,-.2),.18,5,2.5),vec3(1.,.9,.2),col);\n    fragColor=vec4(col,1.);\n}"
        },
    ]
}
OPENSHADER_DIR  = os.path.join(os.path.expanduser("~"), ".openshader")
ASSET_STORE_DIR  = os.path.join(OPENSHADER_DIR, "asset_store")
PREVIEWS_DIR     = os.path.join(ASSET_STORE_DIR, "previews")

CATEGORIES = ["Tous", "backgrounds", "transitions", "post-process", "utilities"]
CATEGORY_ICONS = {
    "backgrounds":  "🌌",
    "transitions":  "🔀",
    "post-process": "✨",
    "utilities":    "🔧",
}

try:
    import urllib.request
    import urllib.error
    _HTTP_OK = True
except ImportError:
    _HTTP_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Modèles de données
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShaderAsset:
    id:            str
    name:          str
    description:   str
    author:        str
    version:       str
    category:      str
    tags:          list[str]
    glsl_url:      str
    preview_url:   str
    sha256:        str
    license:       str        = "CC0"
    rating:        float      = 0.0
    ratings_count: int        = 0
    downloads:     int        = 0
    updated:       str        = ""

    @property
    def category_icon(self) -> str:
        return CATEGORY_ICONS.get(self.category, "📄")

    @property
    def stars_str(self) -> str:
        full  = int(self.rating)
        half  = 1 if (self.rating - full) >= 0.5 else 0
        empty = 5 - full - half
        return "★" * full + "½" * half + "☆" * empty

    @classmethod
    def from_dict(cls, d: dict) -> "ShaderAsset":
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            author=d.get("author", ""),
            version=d.get("version", "1.0.0"),
            category=d.get("category", "utilities"),
            tags=d.get("tags", []),
            glsl_url=d.get("glsl_url", ""),
            preview_url=d.get("preview_url", ""),
            sha256=d.get("sha256", ""),
            license=d.get("license", "CC0"),
            rating=float(d.get("rating", 0.0)),
            ratings_count=int(d.get("ratings_count", 0)),
            downloads=int(d.get("downloads", 0)),
            updated=d.get("updated", ""),
        )


@dataclass
class LocalRating:
    asset_id:  str
    stars:     int
    comment:   str   = ""
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
#  AssetStoreIndex  —  fetch, cache, parse
# ═══════════════════════════════════════════════════════════════════════════════

class AssetStoreIndex:
    def __init__(self):
        os.makedirs(ASSET_STORE_DIR, exist_ok=True)
        os.makedirs(PREVIEWS_DIR,    exist_ok=True)
        self._cache_path   = os.path.join(ASSET_STORE_DIR, "cache.json")
        self._ratings_path = os.path.join(ASSET_STORE_DIR, "ratings.json")
        self._assets: list[ShaderAsset] = []
        self._ratings: dict[str, LocalRating] = {}
        self._builtin_active: bool = False
        self._load_ratings()

    # ── Ratings locaux ─────────────────────────────────────────────────────────

    def _load_ratings(self):
        if not os.path.isfile(self._ratings_path):
            return
        try:
            data = json.loads(open(self._ratings_path, encoding="utf-8").read())
            for aid, rd in data.items():
                self._ratings[aid] = LocalRating(
                    asset_id=aid, stars=rd.get("stars", 0),
                    comment=rd.get("comment", ""),
                    timestamp=rd.get("timestamp", 0),
                )
        except Exception:
            pass

    def _save_ratings(self):
        data = {aid: {"stars": r.stars, "comment": r.comment, "timestamp": r.timestamp}
                for aid, r in self._ratings.items()}
        with open(self._ratings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_rating(self, asset_id: str, stars: int, comment: str = ""):
        self._ratings[asset_id] = LocalRating(asset_id=asset_id, stars=stars, comment=comment)
        self._save_ratings()

    def get_rating(self, asset_id: str) -> Optional[LocalRating]:
        return self._ratings.get(asset_id)

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _is_cache_fresh(self) -> bool:
        if not os.path.isfile(self._cache_path):
            return False
        return (time.time() - os.path.getmtime(self._cache_path)) < CACHE_TTL_S

    def _load_from_cache(self) -> bool:
        if not os.path.isfile(self._cache_path):
            return False
        try:
            data = json.loads(open(self._cache_path, encoding="utf-8").read())
            self._parse(data)
            return True
        except Exception:
            return False

    def _save_cache(self, data: dict):
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── Fetch ──────────────────────────────────────────────────────────────────

    def fetch(self, force: bool = False) -> list[ShaderAsset]:
        if not force and self._is_cache_fresh():
            if self._load_from_cache():
                log.debug("AssetStore : index depuis cache (%d assets)", len(self._assets))
                return self._assets

        if not _HTTP_OK:
            if not self._load_from_cache():
                self._load_builtin()
            return self._assets

        try:
            req = urllib.request.Request(INDEX_URL, headers={"User-Agent": "OpenShader/5.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)
            self._save_cache(data)
            self._parse(data)
            log.info("AssetStore : index distant chargé (%d shaders)", len(self._assets))
        except Exception as e:
            # Fallback 1 : cache local
            log.debug("AssetStore : serveur distant inaccessible (%s)", e)
            if self._load_from_cache():
                log.info("AssetStore : index depuis cache local (%d assets)", len(self._assets))
            else:
                # Fallback 2 : index embarqué (toujours disponible offline)
                self._load_builtin()
                log.info("AssetStore : index embarqué chargé (%d shaders — mode offline)", len(self._assets))
                self._builtin_active = True

        return self._assets

    def _load_builtin(self):
        """Charge l'index embarqué (_BUILTIN_INDEX) — disponible sans réseau."""
        self._parse(_BUILTIN_INDEX)

    def _parse(self, data: dict):
        self._assets = [ShaderAsset.from_dict(d) for d in data.get("shaders", [])]

    # ── Requêtes ───────────────────────────────────────────────────────────────

    def get_assets(self, query: str = "", category: str = "") -> list[ShaderAsset]:
        q = query.lower()
        result = self._assets
        if q:
            result = [a for a in result if
                      q in a.name.lower() or q in a.description.lower()
                      or q in a.author.lower() or any(q in t for t in a.tags)]
        if category and category not in ("Tous", ""):
            result = [a for a in result if a.category == category]
        return result

    def all_tags(self) -> list[str]:
        tags: set[str] = set()
        for a in self._assets:
            tags.update(a.tags)
        return sorted(tags)

    # ── Preview local ──────────────────────────────────────────────────────────

    def preview_path(self, asset: ShaderAsset) -> str:
        ext = os.path.splitext(asset.preview_url)[-1] or ".png"
        return os.path.join(PREVIEWS_DIR, f"{asset.id}{ext}")

    def fetch_preview(self, asset: ShaderAsset, callback: Callable[[str], None]):
        """Télécharge l'image de prévisualisation en arrière-plan."""
        dest = self.preview_path(asset)
        if os.path.isfile(dest):
            callback(dest)
            return

        def _run():
            try:
                req = urllib.request.Request(
                    asset.preview_url, headers={"User-Agent": "OpenShader/5.0"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = r.read()
                with open(dest, "wb") as f:
                    f.write(data)
                callback(dest)
            except Exception as e:
                log.debug("Preview fetch failed for %s: %s", asset.id, e)

        threading.Thread(target=_run, daemon=True, name=f"PreviewFetch-{asset.id}").start()

    # ── Download source GLSL ───────────────────────────────────────────────────

    def fetch_glsl(self, asset: ShaderAsset,
                   on_done: Callable[[str, str | None], None]):
        """
        Retourne le source GLSL de l'asset.
        Priorité : 1) _builtin_glsl embarqué  2) téléchargement distant
        on_done(source_code, error_or_None)
        """
        # Cherche le GLSL embarqué dans l'index brut (_BUILTIN_INDEX)
        builtin_glsl = ""
        for entry in _BUILTIN_INDEX.get("shaders", []):
            if entry.get("id") == asset.id:
                builtin_glsl = entry.get("_builtin_glsl", "")
                break

        if builtin_glsl:
            # Disponible offline — on appelle directement on_done (thread pour cohérence)
            threading.Thread(
                target=lambda: on_done(builtin_glsl, None),
                daemon=True, name=f"GLSLBuiltin-{asset.id}",
            ).start()
            return

        if not asset.glsl_url:
            on_done("", "Pas de source GLSL disponible pour cet asset.")
            return

        def _run():
            try:
                req = urllib.request.Request(
                    asset.glsl_url, headers={"User-Agent": "OpenShader/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    raw = r.read()
                if asset.sha256:
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest != asset.sha256:
                        on_done("", f"SHA-256 mismatch\nAttendu : {asset.sha256}\nReçu    : {digest}")
                        return
                on_done(raw.decode("utf-8"), None)
            except Exception as e:
                on_done("", str(e))

        threading.Thread(target=_run, daemon=True, name=f"GLSLFetch-{asset.id}").start()


# ═══════════════════════════════════════════════════════════════════════════════
#  AssetStoreManager  —  QObject principal
# ═══════════════════════════════════════════════════════════════════════════════

class AssetStoreManager(QObject):
    """
    Signaux :
      index_ready(list[ShaderAsset])
      import_done(str, bool, str)    — (asset_id, ok, glsl_or_error)
      preview_ready(str, str)        — (asset_id, local_path)
      publish_done(bool, str)        — (ok, message)
    """

    index_ready   = pyqtSignal(list)
    import_done   = pyqtSignal(str, bool, str)
    preview_ready = pyqtSignal(str, str)
    publish_done  = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._index = AssetStoreIndex()
        self._api_token: str = self._load_token()

    # ── Token ──────────────────────────────────────────────────────────────────

    _TOKEN_PATH = os.path.join(ASSET_STORE_DIR, "token.txt")

    def _load_token(self) -> str:
        try:
            if os.path.isfile(self._TOKEN_PATH):
                return open(self._TOKEN_PATH, encoding="utf-8").read().strip()
        except Exception:
            pass
        return ""

    def set_token(self, token: str):
        self._api_token = token.strip()
        os.makedirs(ASSET_STORE_DIR, exist_ok=True)
        with open(self._TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(self._api_token)

    @property
    def has_token(self) -> bool:
        return bool(self._api_token)

    @property
    def api_token(self) -> str:
        return self._api_token

    # ── Index ──────────────────────────────────────────────────────────────────

    def refresh(self, force: bool = False):
        def _run():
            assets = self._index.fetch(force=force)
            self.index_ready.emit(assets)
        threading.Thread(target=_run, daemon=True, name="AssetStoreFetch").start()

    def get_assets(self, query: str = "", category: str = "") -> list[ShaderAsset]:
        return self._index.get_assets(query, category)

    def all_tags(self) -> list[str]:
        return self._index.all_tags()

    # ── Preview ────────────────────────────────────────────────────────────────

    def request_preview(self, asset: ShaderAsset):
        def _cb(path: str):
            self.preview_ready.emit(asset.id, path)
        self._index.fetch_preview(asset, _cb)

    # ── Import (téléchargement du GLSL) ────────────────────────────────────────

    def import_asset(self, asset: ShaderAsset):
        def _cb(source: str, err: Optional[str]):
            if err:
                self.import_done.emit(asset.id, False, err)
            else:
                self.import_done.emit(asset.id, True, source)
        self._index.fetch_glsl(asset, _cb)

    # ── Rating ─────────────────────────────────────────────────────────────────

    def rate(self, asset_id: str, stars: int, comment: str = ""):
        stars = max(1, min(5, stars))
        self._index.save_rating(asset_id, stars, comment)

    def get_rating(self, asset_id: str) -> Optional[LocalRating]:
        return self._index.get_rating(asset_id)

    # ── Publication ────────────────────────────────────────────────────────────

    def publish(self, name: str, description: str, category: str,
                tags: list[str], glsl_source: str,
                preview_path: str, license_: str):
        """Publie un shader sur l'API distante en arrière-plan."""
        threading.Thread(
            target=self._do_publish,
            args=(name, description, category, tags, glsl_source, preview_path, license_),
            daemon=True, name="AssetStorePublish",
        ).start()

    def _do_publish(self, name, description, category, tags,
                    glsl_source, preview_path, license_):
        if not self._api_token:
            self.publish_done.emit(False, "Token API manquant. Configurez-le dans les paramètres.")
            return

        if not _HTTP_OK:
            self.publish_done.emit(False, "Module urllib non disponible.")
            return

        # Encode preview en base64 si fournie
        preview_b64 = ""
        if preview_path and os.path.isfile(preview_path):
            try:
                with open(preview_path, "rb") as f:
                    preview_b64 = base64.b64encode(f.read()).decode("ascii")
            except Exception as e:
                log.warning("Preview encode error: %s", e)

        payload = json.dumps({
            "name": name,
            "description": description,
            "category": category,
            "tags": tags,
            "glsl_source": glsl_source,
            "preview_base64": preview_b64,
            "license": license_,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                PUBLISH_API_URL,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_token}",
                    "User-Agent": "OpenShader/5.0",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read().decode("utf-8"))
            self.publish_done.emit(True, f"✓ Shader « {name} » publié avec succès.\nID : {resp.get('id', '?')}")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            self.publish_done.emit(False, f"Erreur HTTP {e.code} : {e.reason}\n{body[:200]}")
        except Exception as e:
            self.publish_done.emit(False, f"Erreur de publication : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers UI  (palette cohérente avec le reste de l'app)
# ═══════════════════════════════════════════════════════════════════════════════

_BG    = "#0d0f16"
_SURF  = "#12141e"
_SURF2 = "#181a26"
_BORD  = "#1e2235"
_TEXT  = "#c0c8e0"
_MUTED = "#5a6080"
_ACC   = "#4e7fff"
_ACC2  = "#50e0a0"
_ACC3  = "#e07850"
_WARN  = "#e0804e"
_SANS  = "Segoe UI, Arial, sans-serif"
_MONO  = "Cascadia Code, Consolas, monospace"

_SS_BASE = f"""
    QWidget           {{ background:{_BG}; color:{_TEXT}; font:9px '{_SANS}'; }}
    QScrollBar:vertical   {{ background:{_SURF}; width:7px; border:none; }}
    QScrollBar::handle:vertical {{ background:{_BORD}; border-radius:3px; min-height:20px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
    QScrollBar:horizontal {{ background:{_SURF}; height:7px; border:none; }}
    QScrollBar::handle:horizontal {{ background:{_BORD}; border-radius:3px; min-width:20px; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
"""


def _btn(label: str, color: str = _ACC, small: bool = False) -> QPushButton:
    fs = "8px" if small else "9px"
    pad = "2px 8px" if small else "4px 12px"
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton{{background:{_SURF};color:{color};border:1px solid {color}44;"
        f"border-radius:3px;padding:{pad};font:{fs} '{_SANS}';}}"
        f"QPushButton:hover{{background:{color}22;border-color:{color}88;}}"
        f"QPushButton:pressed{{background:{color}33;}}"
        f"QPushButton:disabled{{color:{_MUTED};border-color:{_BORD};}}"
    )
    return b


def _label(text: str, color: str = _TEXT, size: str = "9px", bold: bool = False) -> QLabel:
    w = QLabel(text)
    fw = "bold " if bold else ""
    w.setStyleSheet(f"color:{color};font:{fw}{size} '{_SANS}';")
    return w


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{_BORD};background:{_BORD};max-height:1px;")
    return f


class _StarWidget(QWidget):
    rating_changed = pyqtSignal(int)

    def __init__(self, value: int = 0, interactive: bool = True, parent=None):
        super().__init__(parent)
        self._v = value
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self._btns: list[QPushButton] = []
        for i in range(1, 6):
            b = QPushButton("★")
            b.setFixedSize(18, 18)
            b.setFlat(True)
            b.setStyleSheet("border:none;padding:0;")
            if interactive:
                b.clicked.connect(lambda _, v=i: self._set(v))
            self._btns.append(b)
            lay.addWidget(b)
        self._paint()

    def _set(self, v: int):
        self._v = v
        self._paint()
        self.rating_changed.emit(v)

    def _paint(self):
        for i, b in enumerate(self._btns):
            c = "#f0c040" if i < self._v else _MUTED
            b.setStyleSheet(f"border:none;padding:0;font:13px;color:{c};")

    @property
    def value(self) -> int:
        return self._v


class _PreviewLabel(QLabel):
    """Label avec prévisualisation GLSL (image PNG ou placeholder généré)."""

    def __init__(self, size: QSize = QSize(200, 112), parent=None):
        super().__init__(parent)
        self.setFixedSize(size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._show_placeholder()

    def _show_placeholder(self):
        pm = QPixmap(self.size())
        pm.fill(QColor(_SURF2))
        p = QPainter(pm)
        p.setPen(QColor(_MUTED))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "preview\nunavailable")
        p.end()
        self.setPixmap(pm)

    def load_image(self, path: str):
        pm = QPixmap(path)
        if pm.isNull():
            self._show_placeholder()
        else:
            self.setPixmap(pm.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))


# ═══════════════════════════════════════════════════════════════════════════════
#  _AssetCard  —  carte compacte dans la grille
# ═══════════════════════════════════════════════════════════════════════════════

class _AssetCard(QFrame):
    clicked = pyqtSignal(object)   # ShaderAsset

    def __init__(self, asset: ShaderAsset, parent=None):
        super().__init__(parent)
        self._asset = asset
        self.setFixedSize(180, 190)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame{{background:{_SURF};border:1px solid {_BORD};"
            f"border-radius:6px;}} "
            f"QFrame:hover{{border-color:{_ACC}66;background:{_SURF2};}}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(4)

        # Thumbnail
        self._thumb = _PreviewLabel(QSize(178, 100))
        self._thumb.setStyleSheet(
            f"border-radius:5px 5px 0 0;background:{_SURF2};"
        )
        lay.addWidget(self._thumb)

        # Info zone
        info = QWidget()
        info.setStyleSheet("background:transparent;")
        info_lay = QVBoxLayout(info)
        info_lay.setContentsMargins(7, 0, 7, 0)
        info_lay.setSpacing(2)

        # Catégorie badge + nom
        row1 = QHBoxLayout()
        cat_badge = QLabel(f"{asset.category_icon} {asset.category}")
        cat_badge.setStyleSheet(
            f"color:{_ACC};background:{_ACC}18;border-radius:3px;"
            f"padding:1px 5px;font:7px '{_SANS}';"
        )
        row1.addWidget(cat_badge)
        row1.addStretch()
        info_lay.addLayout(row1)

        name_lbl = QLabel(asset.name)
        name_lbl.setStyleSheet(f"color:{_TEXT};font:bold 9px '{_SANS}';")
        name_lbl.setWordWrap(True)
        info_lay.addWidget(name_lbl)

        author_lbl = QLabel(f"par {asset.author}")
        author_lbl.setStyleSheet(f"color:{_MUTED};font:8px '{_SANS}';")
        info_lay.addWidget(author_lbl)

        # Étoiles
        stars_row = QHBoxLayout()
        sw = _StarWidget(round(asset.rating), interactive=False)
        sw.setFixedHeight(16)
        stars_row.addWidget(sw)
        cnt = QLabel(f"({asset.ratings_count})")
        cnt.setStyleSheet(f"color:{_MUTED};font:7px '{_SANS}';")
        stars_row.addWidget(cnt)
        stars_row.addStretch()
        dl = QLabel(f"⬇ {asset.downloads}")
        dl.setStyleSheet(f"color:{_MUTED};font:7px '{_SANS}';")
        stars_row.addWidget(dl)
        info_lay.addLayout(stars_row)

        lay.addWidget(info)

    def set_preview(self, path: str):
        self._thumb.load_image(path)

    @property
    def asset(self) -> ShaderAsset:
        return self._asset

    def mousePressEvent(self, ev):
        self.clicked.emit(self._asset)
        super().mousePressEvent(ev)


# ═══════════════════════════════════════════════════════════════════════════════
#  AssetPublishDialog  —  dialog de publication
# ═══════════════════════════════════════════════════════════════════════════════

class AssetPublishDialog(QDialog):
    """Dialog pour publier un shader depuis OpenShader."""

    def __init__(self, manager: AssetStoreManager,
                 initial_glsl: str = "", parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._preview_path = ""
        self.setWindowTitle("📤 Publier un shader")
        self.setMinimumSize(640, 540)
        self.setStyleSheet(f"QDialog{{ background:{_BG}; color:{_TEXT}; }}" + _SS_BASE)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # ── Token API ──────────────────────────────────────────────────────────
        token_group = QFrame()
        token_group.setStyleSheet(
            f"QFrame{{border:1px solid {_BORD};border-radius:4px;padding:6px;}}")
        tg_lay = QHBoxLayout(token_group)
        tg_lay.setContentsMargins(8, 6, 8, 6)
        tg_lay.addWidget(_label("Token API :", _MUTED))
        self._token_edit = QLineEdit(manager.api_token)
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("Votre token openshader.io…")
        self._token_edit.setStyleSheet(
            f"QLineEdit{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:3px 8px;font:9px '{_MONO}';}}"
            f"QLineEdit:focus{{border-color:{_ACC}88;}}"
        )
        tg_lay.addWidget(self._token_edit, 1)
        btn_save_tok = _btn("💾 Sauvegarder", _MUTED, small=True)
        btn_save_tok.clicked.connect(self._save_token)
        tg_lay.addWidget(btn_save_tok)
        root.addWidget(token_group)

        # ── Formulaire ─────────────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)

        def _row(r: int, label: str, widget: QWidget):
            grid.addWidget(_label(label, _MUTED), r, 0)
            grid.addWidget(widget, r, 1)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Nom du shader…")
        self._name_edit.setStyleSheet(
            f"background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:3px 8px;"
        )
        _row(0, "Nom *", self._name_edit)

        self._cat_combo = QComboBox()
        for cat in CATEGORIES[1:]:
            self._cat_combo.addItem(f"{CATEGORY_ICONS.get(cat,'')} {cat}", cat)
        self._cat_combo.setStyleSheet(
            f"QComboBox{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:3px 8px;}}"
        )
        _row(1, "Catégorie *", self._cat_combo)

        self._tags_edit = QLineEdit()
        self._tags_edit.setPlaceholderText("plasma, color, animated  (séparés par des virgules)")
        self._tags_edit.setStyleSheet(
            f"background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:3px 8px;"
        )
        _row(2, "Tags", self._tags_edit)

        self._license_combo = QComboBox()
        for lic in ["CC0", "CC-BY 4.0", "MIT", "GPL-3.0"]:
            self._license_combo.addItem(lic)
        self._license_combo.setStyleSheet(
            f"QComboBox{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:3px 8px;}}"
        )
        _row(3, "Licence", self._license_combo)

        root.addLayout(grid)

        # Description
        root.addWidget(_label("Description *", _MUTED))
        self._desc_edit = QTextEdit()
        self._desc_edit.setPlaceholderText("Décrivez votre shader…")
        self._desc_edit.setFixedHeight(60)
        self._desc_edit.setStyleSheet(
            f"QTextEdit{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:4px;font:9px '{_SANS}';}}"
            f"QTextEdit:focus{{border-color:{_ACC}88;}}"
        )
        root.addWidget(self._desc_edit)

        # Source GLSL
        root.addWidget(_label("Source GLSL *", _MUTED))
        self._glsl_edit = QTextEdit()
        self._glsl_edit.setPlaceholderText("// Collez ici votre code GLSL…")
        self._glsl_edit.setPlainText(initial_glsl)
        self._glsl_edit.setFixedHeight(90)
        self._glsl_edit.setStyleSheet(
            f"QTextEdit{{background:{_SURF2};color:{_ACC2};border:1px solid {_BORD};"
            f"border-radius:3px;padding:4px;font:9px '{_MONO}';}}"
            f"QTextEdit:focus{{border-color:{_ACC}88;}}"
        )
        root.addWidget(self._glsl_edit)

        # Preview image
        prev_row = QHBoxLayout()
        self._preview_lbl = _label("Aucune image sélectionnée", _MUTED)
        prev_row.addWidget(self._preview_lbl, 1)
        btn_prev = _btn("🖼 Choisir une preview…", _MUTED, small=True)
        btn_prev.clicked.connect(self._pick_preview)
        prev_row.addWidget(btn_prev)
        root.addLayout(prev_row)

        # ── Boutons ────────────────────────────────────────────────────────────
        root.addStretch()
        btns = QHBoxLayout()

        self._status_lbl = _label("", _MUTED)
        btns.addWidget(self._status_lbl, 1)

        btn_cancel = _btn("Annuler", _WARN)
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)

        self._btn_pub = _btn("📤 Publier", _ACC2)
        self._btn_pub.clicked.connect(self._submit)
        btns.addWidget(self._btn_pub)

        root.addLayout(btns)

        # Connexion signaux
        manager.publish_done.connect(self._on_publish_done)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _save_token(self):
        tok = self._token_edit.text().strip()
        self._mgr.set_token(tok)
        self._status_lbl.setText("Token sauvegardé.")
        self._status_lbl.setStyleSheet(f"color:{_ACC2};")

    def _pick_preview(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir une image de prévisualisation", "",
            "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self._preview_path = path
            self._preview_lbl.setText(os.path.basename(path))
            self._preview_lbl.setStyleSheet(f"color:{_TEXT};")

    def _submit(self):
        name = self._name_edit.text().strip()
        desc = self._desc_edit.toPlainText().strip()
        glsl = self._glsl_edit.toPlainText().strip()
        if not name or not desc or not glsl:
            self._status_lbl.setText("⚠ Nom, description et source GLSL obligatoires.")
            self._status_lbl.setStyleSheet(f"color:{_WARN};")
            return
        if not self._mgr.has_token:
            tok = self._token_edit.text().strip()
            if not tok:
                self._status_lbl.setText("⚠ Token API requis pour publier.")
                self._status_lbl.setStyleSheet(f"color:{_WARN};")
                return
            self._mgr.set_token(tok)

        tags = [t.strip() for t in self._tags_edit.text().split(",") if t.strip()]
        cat  = self._cat_combo.currentData() or "utilities"
        lic  = self._license_combo.currentText()

        self._btn_pub.setEnabled(False)
        self._btn_pub.setText("Publication…")
        self._status_lbl.setText("Envoi en cours…")
        self._status_lbl.setStyleSheet(f"color:{_MUTED};")

        self._mgr.publish(name, desc, cat, tags, glsl, self._preview_path, lic)

    def _on_publish_done(self, ok: bool, message: str):
        self._btn_pub.setEnabled(True)
        self._btn_pub.setText("📤 Publier")
        if ok:
            self._status_lbl.setText(message.split("\n")[0])
            self._status_lbl.setStyleSheet(f"color:{_ACC2};")
            QTimer.singleShot(2000, self.accept)
        else:
            self._status_lbl.setText("❌ Échec")
            self._status_lbl.setStyleSheet(f"color:{_WARN};")
            QMessageBox.warning(self, "Erreur de publication", message)


# ═══════════════════════════════════════════════════════════════════════════════
#  AssetStoreBrowser  —  widget principal (intégrable comme onglet/dock)
# ═══════════════════════════════════════════════════════════════════════════════

class AssetStoreBrowser(QWidget):
    """
    Widget complet Asset Store.

    Signaux :
      import_requested(str)   — source GLSL prêt à être importé dans l'éditeur
    """

    import_requested = pyqtSignal(str)   # source GLSL

    def __init__(self, manager: AssetStoreManager,
                 initial_glsl_getter: Optional[Callable[[], str]] = None,
                 parent=None):
        super().__init__(parent)
        self._mgr = manager
        self._get_current_glsl = initial_glsl_getter   # pour pré-remplir le dialog de publication
        self._all_assets: list[ShaderAsset] = []
        self._cards: dict[str, _AssetCard] = {}
        self._selected: Optional[ShaderAsset] = None

        self.setStyleSheet(_SS_BASE)
        self._build_ui()

        manager.index_ready.connect(self._on_index_ready)
        manager.import_done.connect(self._on_import_done)
        manager.preview_ready.connect(self._on_preview_ready)

        manager.refresh()

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barre supérieure ───────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setContentsMargins(10, 8, 10, 8)
        top.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 Rechercher un shader…")
        self._search.setStyleSheet(
            f"QLineEdit{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:4px;padding:4px 8px;font:9px '{_SANS}';}}"
            f"QLineEdit:focus{{border-color:{_ACC}88;}}"
        )
        self._search.textChanged.connect(self._apply_filter)
        top.addWidget(self._search, 1)

        self._cat_combo = QComboBox()
        for cat in CATEGORIES:
            icon = CATEGORY_ICONS.get(cat, "")
            self._cat_combo.addItem(f"{icon} {cat}".strip(), cat)
        self._cat_combo.setStyleSheet(
            f"QComboBox{{background:{_SURF};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:4px;padding:4px 8px;font:9px '{_SANS}';}}"
            f"QComboBox QAbstractItemView{{background:{_SURF};color:{_TEXT};"
            f"selection-background-color:{_ACC}33;}}"
        )
        self._cat_combo.currentTextChanged.connect(self._apply_filter)
        top.addWidget(self._cat_combo)

        btn_refresh = _btn("↻", _MUTED, small=True)
        btn_refresh.setToolTip("Rafraîchir l'index")
        btn_refresh.clicked.connect(lambda: self._mgr.refresh(force=True))
        top.addWidget(btn_refresh)

        btn_publish = _btn("📤 Publier…", _ACC2, small=True)
        btn_publish.setToolTip("Publier un shader sur l'Asset Store")
        btn_publish.clicked.connect(self._open_publish_dialog)
        top.addWidget(btn_publish)

        root.addLayout(top)

        root.addWidget(_sep())

        # ── Corps : grille + panneau de détail ────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            f"QSplitter::handle{{background:{_BORD};width:1px;}}")

        # Grille (scroll)
        grid_container = QWidget()
        grid_container.setStyleSheet(f"background:{_BG};")
        grid_v = QVBoxLayout(grid_container)
        grid_v.setContentsMargins(8, 8, 8, 8)
        grid_v.setSpacing(0)

        self._count_lbl = _label("Chargement…", _MUTED, "8px")
        grid_v.addWidget(self._count_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{_BG};}}")
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background:{_BG};")
        self._grid_lay = QGridLayout(self._grid_widget)
        self._grid_lay.setContentsMargins(0, 8, 0, 8)
        self._grid_lay.setSpacing(8)
        scroll.setWidget(self._grid_widget)
        grid_v.addWidget(scroll, 1)

        splitter.addWidget(grid_container)

        # Panneau de détail
        self._detail_panel = self._build_detail_panel()
        splitter.addWidget(self._detail_panel)

        splitter.setSizes([420, 300])
        root.addWidget(splitter, 1)

        # ── Barre de statut interne ───────────────────────────────────────────
        self._status_bar = _label("", _MUTED, "8px")
        self._status_bar.setContentsMargins(10, 4, 10, 4)
        root.addWidget(self._status_bar)

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(270)
        panel.setStyleSheet(f"background:{_SURF};border-left:1px solid {_BORD};")
        self._dp_lay = QVBoxLayout(panel)
        self._dp_lay.setContentsMargins(14, 14, 14, 14)
        self._dp_lay.setSpacing(8)

        ph = _label("Sélectionnez un shader\npour voir les détails", _MUTED)
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setWordWrap(True)
        self._dp_lay.addStretch()
        self._dp_lay.addWidget(ph)
        self._dp_lay.addStretch()
        return panel

    # ── Grille ─────────────────────────────────────────────────────────────────

    def _apply_filter(self):
        q   = self._search.text().strip()
        cat = self._cat_combo.currentData() or ""
        visible = self._mgr.get_assets(q, cat)
        self._repopulate_grid(visible)

    def _repopulate_grid(self, assets: list[ShaderAsset]):
        # Supprime les anciennes cartes
        while self._grid_lay.count():
            item = self._grid_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        COLS = 3
        for idx, asset in enumerate(assets):
            card = _AssetCard(asset)
            card.clicked.connect(self._show_detail)
            self._cards[asset.id] = card
            self._grid_lay.addWidget(card, idx // COLS, idx % COLS)
            # Demande la preview
            self._mgr.request_preview(asset)

        # Remplir la dernière ligne si incomplète
        remainder = len(assets) % COLS
        if remainder:
            for c in range(remainder, COLS):
                self._grid_lay.addWidget(QWidget(), len(assets) // COLS, c)

        n = len(assets)
        self._count_lbl.setText(
            f"{n} shader{'s' if n != 1 else ''}"
            + (f" sur {len(self._all_assets)}" if n != len(self._all_assets) else ""))

    # ── Détail ─────────────────────────────────────────────────────────────────

    def _show_detail(self, asset: ShaderAsset):
        self._selected = asset

        # Vide le panneau
        while self._dp_lay.count():
            item = self._dp_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Preview
        self._dp_preview = _PreviewLabel(QSize(242, 136))
        self._dp_preview.setStyleSheet(
            f"border-radius:4px;background:{_SURF2};border:1px solid {_BORD};")
        prev_path = self._mgr._index.preview_path(asset)
        if os.path.isfile(prev_path):
            self._dp_preview.load_image(prev_path)
        self._dp_lay.addWidget(self._dp_preview, 0, Qt.AlignmentFlag.AlignHCenter)

        # Nom + version
        name_lbl = _label(asset.name, _TEXT, "12px", bold=True)
        name_lbl.setWordWrap(True)
        self._dp_lay.addWidget(name_lbl)

        row_meta = QHBoxLayout()
        row_meta.addWidget(_label(f"v{asset.version}", _MUTED, "8px"))
        row_meta.addWidget(_label("·", _MUTED, "8px"))
        row_meta.addWidget(_label(f"par {asset.author}", _MUTED, "8px"))
        row_meta.addStretch()
        cat_badge = QLabel(f"{asset.category_icon} {asset.category}")
        cat_badge.setStyleSheet(
            f"color:{_ACC};background:{_ACC}18;border-radius:3px;"
            f"padding:1px 5px;font:7px '{_SANS}';"
        )
        row_meta.addWidget(cat_badge)
        self._dp_lay.addLayout(row_meta)

        # Tags
        tags_row = QHBoxLayout()
        for tag in asset.tags[:5]:
            t = QLabel(f"#{tag}")
            t.setStyleSheet(
                f"color:{_ACC};background:{_ACC}18;border-radius:3px;"
                f"padding:1px 5px;font:7px '{_SANS}';"
            )
            tags_row.addWidget(t)
        tags_row.addStretch()
        self._dp_lay.addLayout(tags_row)

        # Description
        desc_lbl = _label(asset.description, _TEXT, "9px")
        desc_lbl.setWordWrap(True)
        self._dp_lay.addWidget(desc_lbl)

        self._dp_lay.addWidget(_sep())

        # Stats
        stats_row = QHBoxLayout()
        stars_w = _StarWidget(round(asset.rating), interactive=False)
        stats_row.addWidget(stars_w)
        stats_row.addWidget(_label(f"{asset.rating:.1f} ({asset.ratings_count})", _MUTED, "8px"))
        stats_row.addStretch()
        stats_row.addWidget(_label(f"⬇ {asset.downloads}", _MUTED, "8px"))
        self._dp_lay.addLayout(stats_row)

        # Notation locale
        local = self._mgr.get_rating(asset.id)
        self._dp_lay.addWidget(_label("Votre note :", _MUTED, "8px"))
        my_stars = _StarWidget(local.stars if local else 0)
        my_comment = QLineEdit(local.comment if local else "")
        my_comment.setPlaceholderText("Commentaire (optionnel)…")
        my_comment.setStyleSheet(
            f"QLineEdit{{background:{_BG};color:{_TEXT};border:1px solid {_BORD};"
            f"border-radius:3px;padding:3px 6px;font:8px '{_SANS}';}}"
            f"QLineEdit:focus{{border-color:{_ACC}88;}}"
        )
        self._dp_lay.addWidget(my_stars)
        self._dp_lay.addWidget(my_comment)
        btn_rate = _btn("Enregistrer la note", _ACC, small=True)
        btn_rate.clicked.connect(
            lambda: self._mgr.rate(asset.id, my_stars.value, my_comment.text()))
        self._dp_lay.addWidget(btn_rate)

        self._dp_lay.addWidget(_sep())

        # Licence
        lic_row = QHBoxLayout()
        lic_row.addWidget(_label("Licence :", _MUTED, "8px"))
        lic_row.addWidget(_label(asset.license, _ACC2, "8px"))
        lic_row.addStretch()
        if asset.updated:
            lic_row.addWidget(_label(f"màj {asset.updated[:10]}", _MUTED, "8px"))
        self._dp_lay.addLayout(lic_row)

        self._dp_lay.addStretch()

        # ── Bouton Import ──────────────────────────────────────────────────────
        self._import_btn = _btn(f"⬇ Importer dans le projet", _ACC2)
        self._import_btn.clicked.connect(lambda: self._do_import(asset))
        self._dp_lay.addWidget(self._import_btn)

    # ── Import ─────────────────────────────────────────────────────────────────

    def _do_import(self, asset: ShaderAsset):
        self._import_btn.setEnabled(False)
        self._import_btn.setText("Téléchargement…")
        self._status_bar.setText(f"Import de « {asset.name} »…")
        self._mgr.import_asset(asset)

    def _on_import_done(self, asset_id: str, ok: bool, glsl_or_error: str):
        if self._selected and self._selected.id == asset_id:
            self._import_btn.setEnabled(True)
            self._import_btn.setText("⬇ Importer dans le projet")

        if ok:
            self._status_bar.setText(f"✓ Import réussi")
            self.import_requested.emit(glsl_or_error)
        else:
            self._status_bar.setText(f"✗ Erreur d'import")
            QMessageBox.warning(self, "Erreur d'import",
                                f"Impossible d'importer le shader :\n{glsl_or_error}")

    # ── Preview ────────────────────────────────────────────────────────────────

    def _on_preview_ready(self, asset_id: str, path: str):
        card = self._cards.get(asset_id)
        if card:
            card.set_preview(path)
        if self._selected and self._selected.id == asset_id:
            if hasattr(self, '_dp_preview'):
                self._dp_preview.load_image(path)

    # ── Index ──────────────────────────────────────────────────────────────────

    def _on_index_ready(self, assets: list):
        self._all_assets = assets
        self._apply_filter()
        n = len(assets)
        idx = self._mgr._index
        if getattr(idx, "_builtin_active", False):
            source_hint = " · <span style='color:#e07850;'>mode offline — index embarqué</span>"
        elif os.path.isfile(idx._cache_path):
            mtime = os.path.getmtime(idx._cache_path)
            import datetime
            age = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m %H:%M")
            src_str = _BUILTIN_INDEX.get("_source", "")
            if src_str == "builtin":
                source_hint = " · <span style='color:#e07850;'>index embarqué (offline)</span>"
            else:
                source_hint = f" · cache {age}"
        else:
            source_hint = " · index distant"
        self._status_bar.setText(
            f"{n} shader{'s' if n != 1 else ''} disponibles{source_hint}")
        self._status_bar.setTextFormat(Qt.TextFormat.RichText)

    # ── Publication ────────────────────────────────────────────────────────────

    def _open_publish_dialog(self):
        glsl = ""
        if self._get_current_glsl:
            try:
                glsl = self._get_current_glsl()
            except Exception:
                pass
        dlg = AssetPublishDialog(self._mgr, initial_glsl=glsl, parent=self)
        dlg.exec()


# ═══════════════════════════════════════════════════════════════════════════════
#  Fonction factory  —  appelée depuis MainWindow
# ═══════════════════════════════════════════════════════════════════════════════

def create_asset_store_browser(
    glsl_getter: Optional[Callable[[], str]] = None,
    parent=None,
) -> tuple[AssetStoreManager, AssetStoreBrowser]:
    """
    Crée et retourne (manager, browser).
    glsl_getter : callable () → str, pour pré-remplir le dialog de publication
                  avec le shader actif dans l'éditeur.
    """
    mgr     = AssetStoreManager(parent)
    browser = AssetStoreBrowser(mgr, initial_glsl_getter=glsl_getter, parent=parent)
    return mgr, browser
