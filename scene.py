"""OpenGL scene: displaced blob, wireframe shell, bar ring, particles, bloom."""
import math

import moderngl
import numpy as np

BAR_RADIUS = 4.2
BLOB_RADIUS = 1.5
BG = (0.02, 0.023, 0.05)

# ---------------------------------------------------------------- math helpers

def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, np.float32) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def scale(k):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = k
    return m


def gl(m):
    """numpy row-major matrix -> column-major bytes for a GLSL mat4."""
    return np.ascontiguousarray(m.T, np.float32).tobytes()


# ---------------------------------------------------------------- geometry

def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [tuple(np.array(v) / np.linalg.norm(v)) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache = {}

        def mid(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                p = (np.array(verts[a]) + np.array(verts[b])) / 2
                verts.append(tuple(p / np.linalg.norm(p)))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts, np.float32), np.array(faces, np.int32)


def edges_of(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0).astype(np.int32)


def bar_cube(w=0.11):
    """Unit-height box, base at y=0. Returns (36, 3) triangle vertices."""
    h = w / 2
    c = np.array([[x, y, z] for x in (-h, h) for y in (0, 1) for z in (-h, h)], np.float32)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, cc, d in quads:
        tris += [c[a], c[b], c[cc], c[a], c[cc], c[d]]
    return np.array(tris, np.float32)


def annulus(r, width, segments=256):
    a = np.linspace(0, 2 * np.pi, segments + 1)
    inner = np.stack([np.cos(a) * (r - width), np.zeros_like(a), np.sin(a) * (r - width)], 1)
    outer = np.stack([np.cos(a) * (r + width), np.zeros_like(a), np.sin(a) * (r + width)], 1)
    return np.stack([inner, outer], 1).reshape(-1, 3).astype(np.float32)  # triangle strip


# ---------------------------------------------------------------- shaders

HSL = """
vec3 hsl2rgb(vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.,4.,2.), 6.) - 3.) - 1., 0., 1.);
    return c.z + c.y*(rgb - 0.5)*(1. - abs(2.*c.z - 1.));
}
"""

BLOB_VS = """
#version 330
uniform mat4 uProj, uView, uModel;
uniform float uTime, uBass, uMid, uTreble, uPulse, uRadius;
in vec3 in_pos;
out float vDisp; out vec3 vNormal; out vec3 vView;
float wave(vec3 p){
    return sin(p.x*3.0 + uTime*1.7)*sin(p.y*3.3 + uTime*1.3)*sin(p.z*2.9 + uTime*1.1)
         + 0.5*sin(p.x*7.0 - uTime*2.3)*sin(p.y*6.5 + uTime*1.9)*sin(p.z*7.3 - uTime*2.1);
}
void main(){
    vec3 n = normalize(in_pos);
    float d = wave(n * (1.0 + uMid*0.8));
    float disp = d*(0.1 + uMid*0.55) + uBass*0.5 + uPulse*0.2
               + uTreble*0.12*sin(36.0*n.y + uTime*9.0);
    vDisp = disp;
    vec4 mv = uView * uModel * vec4(n*uRadius + n*disp*(uRadius/1.5), 1.0);
    vNormal = mat3(uView * uModel) * n;
    vView = -mv.xyz;
    gl_Position = uProj * mv;
}
"""

BLOB_FS = """
#version 330
uniform float uHue, uSat, uBass;
in float vDisp; in vec3 vNormal; in vec3 vView;
out vec4 frag;
""" + HSL + """
void main(){
    float fr = pow(1.0 - max(dot(normalize(vNormal), normalize(vView)), 0.0), 2.2);
    vec3 base = hsl2rgb(vec3(fract(uHue + vDisp*0.3), uSat, clamp(0.3 + vDisp*0.35, 0.05, 0.7)));
    vec3 rim  = hsl2rgb(vec3(fract(uHue + 0.12), uSat, 0.6));
    frag = vec4(base*(0.25 + 0.4*uBass) + rim*fr*1.3, 1.0);
}
"""

WIRE_FS = """
#version 330
uniform float uHue, uSat, uTreble;
in float vDisp; in vec3 vNormal; in vec3 vView;
out vec4 frag;
""" + HSL + """
void main(){
    frag = vec4(hsl2rgb(vec3(fract(uHue + 0.5 + vDisp*0.2), uSat, 0.5)) * (0.2 + uTreble*0.9), 1.0);
}
"""

BARS_VS = """
#version 330
uniform mat4 uProj, uView;
uniform float uHue, uSpread, uSat, uRadius, uRot;
in vec3 in_pos;
in vec4 inst;      // angle, signed height, value, band fraction
in float dim;      // 1 for bars, <1 for the reflection
out vec3 vColor;
""" + HSL + """
void main(){
    float a = inst.x + uRot;
    float c = cos(a), s = sin(a);
    vec3 p = vec3(in_pos.x, in_pos.y*inst.y, in_pos.z);
    vec3 w = vec3(c*p.x - s*p.z + c*uRadius, p.y, s*p.x + c*p.z + s*uRadius);
    gl_Position = uProj * uView * vec4(w, 1.0);
    vec3 col = hsl2rgb(vec3(fract(uHue + inst.w*uSpread), uSat, 0.14 + inst.z*0.32));
    vColor = col * dim * (0.4 + 0.7*in_pos.y);
}
"""

FLAT_FS = """
#version 330
in vec3 vColor;
out vec4 frag;
void main(){ frag = vec4(vColor, 1.0); }
"""

RING_VS = """
#version 330
uniform mat4 uProj, uView, uModel;
uniform vec3 uColor;
in vec3 in_pos;
out vec3 vColor;
void main(){ gl_Position = uProj*uView*uModel*vec4(in_pos, 1.0); vColor = uColor; }
"""

POINTS_VS = """
#version 330
uniform mat4 uProj, uView, uModel;
uniform float uSize, uScale;
in vec3 in_pos; in float in_seed;
out float vFade;
void main(){
    vec4 mv = uView * uModel * vec4(in_pos, 1.0);
    gl_Position = uProj * mv;
    float d = max(-mv.z, 0.1);
    gl_PointSize = uSize * uScale * (0.5 + in_seed) / d;
    vFade = exp(-d * 0.025);
}
"""

POINTS_FS = """
#version 330
uniform vec3 uColor;
in float vFade;
out vec4 frag;
void main(){
    float r = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.0, r); a *= a;
    frag = vec4(uColor * a * vFade, 1.0);
}
"""

QUAD_VS = """
#version 330
uniform vec4 uRect;   // x0, y0, x1, y1 in NDC
in vec2 in_pos;       // 0..1
out vec2 uv;
void main(){ uv = in_pos; gl_Position = vec4(mix(uRect.xy, uRect.zw, in_pos), 0.0, 1.0); }
"""

PREFILTER_FS = """
#version 330
uniform sampler2D uTex; uniform vec2 uTexel; uniform float uThreshold, uKnee;
in vec2 uv; out vec4 frag;
void main(){
    vec4 d = uTexel.xyxy * vec4(-1, -1, 1, 1);
    vec3 c = (texture(uTex, uv + d.xy).rgb + texture(uTex, uv + d.zy).rgb
            + texture(uTex, uv + d.xw).rgb + texture(uTex, uv + d.zw).rgb) * 0.25;
    float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - uThreshold + uKnee, 0.0, 2.0*uKnee);
    soft = soft*soft / (4.0*uKnee + 1e-4);
    frag = vec4(c * max(soft, br - uThreshold) / max(br, 1e-4), 1.0);
}
"""

DOWN_FS = """
#version 330
uniform sampler2D uTex; uniform vec2 uTexel;
in vec2 uv; out vec4 frag;
void main(){
    vec4 d = uTexel.xyxy * vec4(-1, -1, 1, 1);
    frag = vec4((texture(uTex, uv + d.xy).rgb + texture(uTex, uv + d.zy).rgb
               + texture(uTex, uv + d.xw).rgb + texture(uTex, uv + d.zw).rgb) * 0.25, 1.0);
}
"""

UP_FS = """
#version 330
uniform sampler2D uTex; uniform vec2 uTexel;
in vec2 uv; out vec4 frag;
void main(){
    vec4 d = uTexel.xyxy * vec4(1, 1, -1, 0);
    vec3 s = texture(uTex, uv - d.xy).rgb
           + texture(uTex, uv - d.wy).rgb * 2.0
           + texture(uTex, uv - d.zy).rgb
           + texture(uTex, uv + d.zw).rgb * 2.0
           + texture(uTex, uv).rgb * 4.0
           + texture(uTex, uv + d.xw).rgb * 2.0
           + texture(uTex, uv + d.zy).rgb
           + texture(uTex, uv + d.wy).rgb * 2.0
           + texture(uTex, uv + d.xy).rgb;
    frag = vec4(s / 16.0, 1.0);
}
"""

COMPOSITE_FS = """
#version 330
uniform sampler2D uScene, uBloom; uniform float uStrength;
in vec2 uv; out vec4 frag;
vec3 aces(vec3 x){ return clamp((x*(2.51*x + 0.03)) / (x*(2.43*x + 0.59) + 0.14), 0.0, 1.0); }
void main(){
    vec3 c = texture(uScene, uv).rgb + texture(uBloom, uv).rgb * uStrength;
    c = aces(c * 1.35);
    float v = smoothstep(1.25, 0.35, length(uv - 0.5));   // vignette
    frag = vec4(c * mix(0.55, 1.0, v), 1.0);
}
"""

OVERLAY_FS = """
#version 330
uniform sampler2D uTex; uniform float uAlpha;
in vec2 uv; out vec4 frag;
void main(){ vec4 c = texture(uTex, uv); frag = vec4(c.rgb, c.a * uAlpha); }
"""


def set_u(prog, name, value):
    """Set a uniform if the GLSL compiler kept it."""
    if name in prog:
        if isinstance(value, bytes):
            prog[name].write(value)
        else:
            prog[name].value = value


# ---------------------------------------------------------------- scene

class Scene:
    BAR_BANDS = 64

    def __init__(self, ctx: moderngl.Context, size):
        self.ctx = ctx
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.samples = min(4, ctx.max_samples)

        # blob + wire shell
        self.blob_prog = ctx.program(vertex_shader=BLOB_VS, fragment_shader=BLOB_FS)
        self.wire_prog = ctx.program(vertex_shader=BLOB_VS, fragment_shader=WIRE_FS)
        v, f = icosphere(6)
        self.blob_vao = ctx.vertex_array(self.blob_prog, [(ctx.buffer(v), "3f", "in_pos")],
                                         index_buffer=ctx.buffer(f), index_element_size=4)
        wv, wf = icosphere(3)
        self.wire_vao = ctx.vertex_array(self.wire_prog, [(ctx.buffer(wv), "3f", "in_pos")],
                                         index_buffer=ctx.buffer(edges_of(wf)), index_element_size=4)

        # bar ring: BAR_BANDS mirrored around the circle, plus a reflection below
        self.bar_count = self.BAR_BANDS * 2
        self.bars_prog = ctx.program(vertex_shader=BARS_VS, fragment_shader=FLAT_FS)
        self.bar_inst = ctx.buffer(reserve=self.bar_count * 2 * 5 * 4, dynamic=True)
        self.bars_vao = ctx.vertex_array(self.bars_prog, [
            (ctx.buffer(bar_cube()), "3f", "in_pos"),
            (self.bar_inst, "4f 1f/i", "inst", "dim"),
        ])
        n = self.bar_count
        self._bar_angles = (np.arange(n) / n * 2 * np.pi).astype(np.float32)
        self._bar_band = np.array([i if i < self.BAR_BANDS else n - 1 - i for i in range(n)])

        # ring on the floor
        self.ring_prog = ctx.program(vertex_shader=RING_VS, fragment_shader=FLAT_FS)
        self.ring_vao = ctx.vertex_array(self.ring_prog,
                                         [(ctx.buffer(annulus(BAR_RADIUS - 0.4, 0.025)), "3f", "in_pos")])

        # particles in a spherical shell
        rng = np.random.default_rng(7)
        d = rng.normal(size=(4000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts = d * rng.uniform(8, 50, size=(4000, 1))
        seeds = rng.random(4000)
        data = np.hstack([pts, seeds[:, None]]).astype(np.float32)
        self.pts_prog = ctx.program(vertex_shader=POINTS_VS, fragment_shader=POINTS_FS)
        self.pts_vao = ctx.vertex_array(self.pts_prog, [(ctx.buffer(data), "3f 1f", "in_pos", "in_seed")])

        # fullscreen / overlay quad
        quad = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], np.float32))
        self._post = {}
        for name, fs in [("prefilter", PREFILTER_FS), ("down", DOWN_FS), ("up", UP_FS),
                         ("composite", COMPOSITE_FS), ("overlay", OVERLAY_FS)]:
            p = ctx.program(vertex_shader=QUAD_VS, fragment_shader=fs)
            self._post[name] = (p, ctx.vertex_array(p, [(quad, "2f", "in_pos")]))
        self.overlay_tex = None

        self.resize(size)

    # ------------------------------------------------------------ targets
    def resize(self, size):
        ctx = self.ctx
        self.size = w, h = max(1, size[0]), max(1, size[1])
        for attr in ("msaa_fbo", "scene_fbo"):
            if hasattr(self, attr):
                getattr(self, attr).release()
        self.msaa_fbo = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((w, h), 4, samples=self.samples, dtype="f2")],
            depth_attachment=ctx.depth_renderbuffer((w, h), samples=self.samples))
        self.scene_tex = ctx.texture((w, h), 4, dtype="f2")
        self.scene_tex.repeat_x = self.scene_tex.repeat_y = False
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene_tex],
                                         depth_attachment=ctx.depth_renderbuffer((w, h)))
        self.levels = []
        lw, lh = w // 2, h // 2
        while len(self.levels) < 6 and min(lw, lh) >= 4:
            t = ctx.texture((lw, lh), 4, dtype="f2")
            t.repeat_x = t.repeat_y = False
            self.levels.append((t, ctx.framebuffer(color_attachments=[t])))
            lw, lh = lw // 2, lh // 2

    def set_overlay(self, rgba_bytes, size):
        if self.overlay_tex is None or self.overlay_tex.size != size:
            if self.overlay_tex is not None:
                self.overlay_tex.release()
            self.overlay_tex = self.ctx.texture(size, 4)
        self.overlay_tex.write(rgba_bytes)

    # ------------------------------------------------------------ draw
    def _quad(self, name, rect=(-1, -1, 1, 1), **uniforms):
        prog, vao = self._post[name]
        set_u(prog, "uRect", rect)
        for k, v in uniforms.items():
            set_u(prog, k, v)
        vao.render(moderngl.TRIANGLE_STRIP)

    def render(self, s):
        """`s` is a dict of per-frame state from main.py."""
        ctx = self.ctx
        w, h = self.size
        proj = perspective(s["fov"], w / h, 0.1, 300)
        view = look_at(s["eye"])
        P, V = gl(proj), gl(view)

        fbo = self.msaa_fbo
        fbo.use()
        fbo.clear(*BG, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        fbo.depth_mask = True

        common = dict(uProj=P, uView=V, uTime=s["t"], uBass=s["bass"], uMid=s["mid"],
                      uTreble=s["treble"], uPulse=s["pulse"], uHue=s["hue"], uSat=s["sat"])

        # blob
        for k, v in common.items():
            set_u(self.blob_prog, k, v)
        set_u(self.blob_prog, "uModel", gl(rot("y", s["core_ry"]) @ rot("x", s["core_rx"])))
        set_u(self.blob_prog, "uRadius", BLOB_RADIUS)
        self.blob_vao.render(moderngl.TRIANGLES)

        # bars (+ reflection)
        bands = s["bands"]
        n = self.bar_count
        v = bands[self._bar_band]
        heights = 0.05 + v * v * 2.2  # squared: quiet bands stay low, peaks pop
        frac = self._bar_band / self.BAR_BANDS
        top = np.stack([self._bar_angles, heights, v, frac, np.ones(n)], 1)
        bottom = np.stack([self._bar_angles, -heights * 0.55, v, frac, np.full(n, 0.22)], 1)
        self.bar_inst.write(np.vstack([top, bottom]).astype(np.float32).tobytes())
        for k, val in dict(uProj=P, uView=V, uHue=s["hue"], uSpread=s["spread"], uSat=s["sat"],
                           uRadius=BAR_RADIUS, uRot=s["t"] * 0.05).items():
            set_u(self.bars_prog, k, val)
        self.bars_vao.render(moderngl.TRIANGLES, instances=n * 2)

        # floor ring
        set_u(self.ring_prog, "uProj", P)
        set_u(self.ring_prog, "uView", V)
        set_u(self.ring_prog, "uModel", gl(scale(1 + s["pulse"] * 0.06)))
        set_u(self.ring_prog, "uColor", s["ring_color"])
        self.ring_vao.render(moderngl.TRIANGLE_STRIP)

        # additive layers: wire shell, particles
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        fbo.depth_mask = False
        for k, val in common.items():
            set_u(self.wire_prog, k, val)
        set_u(self.wire_prog, "uModel", gl(rot("y", s["wire_ry"]) @ rot("z", s["wire_rz"])))
        set_u(self.wire_prog, "uRadius", BLOB_RADIUS * 1.18)
        self.wire_vao.render(moderngl.LINES)

        set_u(self.pts_prog, "uProj", P)
        set_u(self.pts_prog, "uView", V)
        set_u(self.pts_prog, "uModel", gl(rot("y", s["pts_ry"]) @ rot("x", s["pts_rx"])))
        set_u(self.pts_prog, "uSize", s["pt_size"])
        set_u(self.pts_prog, "uScale", h * 0.5 * proj[1, 1])
        set_u(self.pts_prog, "uColor", s["pt_color"])
        self.pts_vao.render(moderngl.POINTS)
        fbo.depth_mask = True

        # resolve MSAA, then bloom
        ctx.copy_framebuffer(self.scene_fbo, self.msaa_fbo)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)

        src = self.scene_tex
        for i, (tex, lfbo) in enumerate(self.levels):
            lfbo.use()
            src.use(0)
            texel = (1 / src.size[0], 1 / src.size[1])
            if i == 0:
                self._quad("prefilter", uTex=0, uTexel=texel, uThreshold=0.55, uKnee=0.35)
            else:
                self._quad("down", uTex=0, uTexel=texel)
            src = tex
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.levels) - 1, 0, -1):
            lo_tex = self.levels[i][0]
            self.levels[i - 1][1].use()
            lo_tex.use(0)
            self._quad("up", uTex=0, uTexel=(1 / lo_tex.size[0], 1 / lo_tex.size[1]))
        ctx.disable(moderngl.BLEND)

        # composite to the window
        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        self.scene_tex.use(0)
        self.levels[0][0].use(1)
        self._quad("composite", uScene=0, uBloom=1, uStrength=s["bloom"])

        # HUD overlay
        if self.overlay_tex is not None and s["hud_alpha"] > 0.01:
            ow, oh = self.overlay_tex.size
            x0, y1 = -1 + 32 / w, 1 - 32 / h
            rect = (x0, y1 - 2 * oh / h, x0 + 2 * ow / w, y1)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.overlay_tex.use(0)
            self._quad("overlay", rect=rect, uTex=0, uAlpha=s["hud_alpha"])
            ctx.disable(moderngl.BLEND)
