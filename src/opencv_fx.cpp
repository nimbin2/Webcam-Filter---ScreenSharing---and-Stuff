/* SPDX-License-Identifier: GPL-2.0-only
 * vcam — Linux virtual webcam pipeline
 * Copyright (C) 2026 the vcam authors
 *
 * This program is free software under the terms of the GNU General
 * Public License version 2. See the LICENSE file for the full text.
 */

/*
 * opencv_fx.cpp — OpenCV 5 backed effects, exposed through the flat C
 * ABI declared in vcam_cv.h. Built only when OPENCV=1.
 *
 * Design notes for the laptop case:
 *   - Classic effects (cartoon/pencil/stylize/flow/bgsub) use only CPU
 *     imgproc/photo/video and run comfortably at 720p.
 *   - DNN effects (face/segmentation) run inference on a DOWNSCALED
 *     copy and cache the result, re-inferring only every Nth frame
 *     (see INFER_EVERY). Between inferences the cached mask/boxes are
 *     reused, so per-frame cost stays low and steady.
 *   - Every entry point is wrapped in try/catch. Anything unexpected
 *     (bad frame, missing/corrupt model, OpenCV throwing) degrades to
 *     leaving the frame untouched. Nothing crosses the C boundary as an
 *     exception, so the pipeline can never crash because of OpenCV.
 *   - Models are loaded lazily on first use and cached. A missing model
 *     prints one warning and thereafter no-ops.
 */

#include "vcam_cv.h"
#include "vcam_ort.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <memory>
#include <sys/stat.h>
#include <chrono>

/* ── OpenCV 5 feature gate ─────────────────────────────────────
 * When built against OpenCV 5 we light up the 5.x-only features:
 *
 *   - the NEW graph-based DNN engine (typed op graph, shape inference,
 *     constant folding, operator fusion; 80%+ ONNX operator coverage vs
 *     ~22% in 4.x). readNet() gained a 4th parameter
 *     `EngineType engine = ENGINE_AUTO`; we pass ENGINE_AUTO explicitly
 *     so 5.x tries the new engine first and falls back to the classic
 *     engine only if the model can't be loaded by it.
 *   - first-class FP16: cv::hfloat (CV_16F) became a full Mat type in
 *     5.x with arithmetic available even on hardware without native
 *     FP16. We run the person-matte feathered blend at half precision,
 *     halving the memory traffic of the heaviest per-frame step.
 *
 * Against OpenCV 4.x the same code compiles with the classic engine and
 * an FP32 blend. Everything is decided at compile time from CV_VERSION,
 * and cv_engine_info() reports which path is active. */
#if CV_VERSION_MAJOR >= 5
#  define VCAM_CV5 1
#else
#  define VCAM_CV5 0
#endif

namespace {

#if VCAM_CV5
/* Load an ONNX model on OpenCV 5's new graph engine (ENGINE_AUTO:
 * new engine first, classic as automatic fallback). */
inline cv::dnn::Net load_net(const std::string &path)
{
    return cv::dnn::readNet(path, "", "", cv::dnn::ENGINE_AUTO);
}
#else
inline cv::dnn::Net load_net(const std::string &path)
{
    return cv::dnn::readNet(path);
}
#endif

/* Wrap the caller's BGR24 buffer as a cv::Mat WITHOUT copying. The Mat
 * shares the caller's memory (respecting stride), so writes land back
 * in the frame. Valid only for the duration of the call. */
inline cv::Mat wrap_bgr(unsigned char *data, int w, int h, int stride)
{
    return cv::Mat(h, w, CV_8UC3, data, static_cast<size_t>(stride));
}

inline unsigned char clamp_int_u8(int v)
{
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* cheap deterministic 2D+time hash -> [0,1) for noise-driven effects */
inline float uhash(int x, int y, unsigned t)
{
    unsigned h = (unsigned)x * 374761393u + (unsigned)y * 668265263u + t * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)((h ^ (h >> 16)) & 0xFFFFFF) / 16777216.0f;
}

bool path_exists(const std::string &p)
{
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

/* ── model directory resolution ── */
std::string resolve_model_dir()
{
    if (const char *env = std::getenv("VCAM_MODEL_DIR"))
        if (path_exists(env)) return env;

    const char *home = std::getenv("HOME");
    if (home) {
        std::string u = std::string(home) + "/.local/share/vcam/models";
        if (path_exists(u)) return u;
    }
    if (path_exists("/usr/local/share/vcam/models"))
        return "/usr/local/share/vcam/models";
    if (path_exists("/usr/share/vcam/models"))
        return "/usr/share/vcam/models";

    /* Compiled default even if it doesn't exist yet — callers check for
     * the specific model file and warn if absent. */
    if (home) return std::string(home) + "/.local/share/vcam/models";
    return "/usr/local/share/vcam/models";
}

const std::string &model_dir()
{
    static const std::string d = resolve_model_dir();
    return d;
}

/* Print a given warning only once per process. */
void warn_once(const char *key, const std::string &msg)
{
    static std::mutex m;
    static std::string seen;
    std::lock_guard<std::mutex> lock(m);
    std::string tag = std::string(key) + "\n";
    if (seen.find(tag) != std::string::npos) return;
    seen += tag;
    std::fprintf(stderr, "[vcam/opencv] %s\n", msg.c_str());
}

/* A monotonically increasing per-effect frame counter, so DNN effects
 * can re-infer every Nth frame. */
inline unsigned tick(unsigned &counter) { return counter++; }

constexpr int INFER_EVERY = 3;   /* run DNN inference 1 in 3 frames */

/* Optional GPU inference: set VCAM_CV_OPENCL=1 to run DNN targets on
 * OpenCL (AMD Radeon iGPUs work through this). Falls back to CPU
 * automatically if the OpenCL path throws at load or first forward. */
inline void set_net_target(cv::dnn::Net &n)
{
    n.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    const char *ocl = std::getenv("VCAM_CV_OPENCL");
    if (ocl && *ocl == '1')
        n.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    else
        n.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

/* ────────────────────────────────────────────────────────────
 * DNN model holders (lazy, cached, guarded)
 * ──────────────────────────────────────────────────────────── */

struct FaceState {
    cv::Ptr<cv::FaceDetectorYN> det;
    bool tried = false;
    bool ok = false;
    int  net_w = 0, net_h = 0;
    unsigned counter = 0;
    cv::Mat faces;             /* cached detections between inferences */
    std::mutex mtx;
};
FaceState g_face;

struct SegState {
    cv::dnn::Net net;
    bool tried = false;
    bool ok = false;
    unsigned counter = 0;
    cv::Mat mask;             /* SOFT person alpha, CV_32F 0..1, EMA-smoothed */
    cv::Mat bg_est;           /* running background estimate (invisibility) */
    int bg_frames = 0;        /* how much background we've learned */
    float hx = 0.0f, hy = 0.0f;   /* EMA head offset, -0.5..0.5 (window3d) */
    int person_ch = -1;           /* which output channel is the person (-1=unlocked) */
    int ch_vote = 0, ch_frames = 0;   /* voting to lock person_ch */
    /* RVM (Robust Video Matting): a recurrent video-native matting net.
     * Far better temporal stability + hair detail than PP-HumanSeg.
     * Tried first if the model file exists; ANY failure falls back. */
    cv::dnn::Net rvm;
    int rvm_state = 0;            /* OpenCV-DNN RVM: 0=untried 1=ok -1=failed */
    int ort_state = 0;            /* ONNX-Runtime RVM: 0=untried 1=ok -1=failed */
    cv::Mat r1, r2, r3, r4;       /* recurrent states, cycled per frame */
    std::mutex mtx;
};
SegState g_seg;

/* User-supplied background image for personfx mode 3, set at runtime
 * via 'vcam-ctl filter bgimage <path>'. Cached at the last frame size
 * with cover-scaling (fill + center-crop, aspect preserved). */
struct BgImgState {
    cv::Mat orig;             /* as loaded */
    cv::Mat scaled;           /* cover-scaled to frame size */
    std::mutex mtx;
};
BgImgState g_bgimg;

/* User-supplied face-overlay image (PNG with alpha) for facefx mode 24,
 * set via 'vcam-ctl filter faceimage <path>'. Tracked to the face:
 * scaled by face width, rotated by eye angle. */
struct FaceImgState {
    cv::Mat rgba;             /* as loaded, 4-channel */
    std::mutex mtx;
};
FaceImgState g_faceimg;

/* paintflow: a persistent canvas advected by dense optical flow —
 * moving objects drag the paint with them. */
struct PaintState {
    cv::Mat canvas, prev_gray, mapx, mapy;
    std::mutex mtx;
};
PaintState g_paint;

struct ClassicState {
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2;
    cv::Mat prev_gray;        /* for optical flow */
    std::mutex mtx;
};
ClassicState g_classic;

/* ── Neural style transfer state ──
 * Five fast-neural-style networks (Johnson et al., instance-norm),
 * MIT-licensed ONNX exports. Fully convolutional: any input size works,
 * which gives us a live quality↔speed knob. Input/output: float RGB
 * 0..255, NCHW. */
static const char *STYLE_FILES[5] = {
    "mosaic-9.onnx", "candy-9.onnx", "rain-princess-9.onnx",
    "udnie-9.onnx", "pointilism-9.onnx",
};
struct StyleState {
    cv::dnn::Net net[5];
    bool tried[5] = {false,false,false,false,false};
    bool ok[5]    = {false,false,false,false,false};
    cv::Mat ema;              /* temporal smoothing of the stylized frame */
    int ema_style = -1;
    std::mutex mtx;
};
StyleState g_style;

bool ensure_style(int s)
{
    if (g_style.tried[s]) return g_style.ok[s];
    g_style.tried[s] = true;
    std::string path = model_dir() + "/" + STYLE_FILES[s];
    if (!path_exists(path)) {
        warn_once((std::string("style") + STYLE_FILES[s]).c_str(),
            "style model not found: " + path +
            "  — run 'vcam-setup --models-style' to download the styles. "
            "'styleai' passes through untouched.");
        return false;
    }
    try {
        g_style.net[s] = load_net(path);
        set_net_target(g_style.net[s]);
        g_style.ok[s] = true;
    } catch (const std::exception &e) {
        warn_once("style_load", std::string("failed to load style model: ") + e.what());
    }
    return g_style.ok[s];
}

/* ── Emotion (FER) state ──
 * MobileFaceNet expression classifier from the OpenCV Zoo (Apache-2.0),
 * 112x112 input, 7 outputs. ~2ms on a laptop CPU. Reuses the YuNet face
 * detector for the crop. Scores are EMA-smoothed to avoid flicker. */
static const char *EMO_NAME[7] =
    { "ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISED" };
/* BGR accent color per emotion (neutral stays colorless) */
static const int EMO_COL[7][3] = {
    {40,40,230},{60,170,60},{190,60,170},{40,200,255},
    {160,160,160},{230,140,50},{255,255,255},
};
struct MoodState {
    cv::dnn::Net net;
    bool tried = false, ok = false;
    float scores[7] = {0,0,0,0,0,0,0};   /* EMA-smoothed */
    unsigned counter = 0;
    /* party-mode particles: x,y,vx,vy,life,color-idx */
    float px[96], py[96], pvx[96], pvy[96], pl[96];
    int pc[96];
    std::mutex mtx;
};
MoodState g_mood;

bool ensure_mood()
{
    if (g_mood.tried) return g_mood.ok;
    g_mood.tried = true;
    std::string path = model_dir() +
        "/facial_expression_recognition_mobilefacenet_2022july.onnx";
    if (!path_exists(path)) {
        warn_once("fer_model",
            "expression model not found: " + path +
            "  — run 'vcam-setup --models' to download it. "
            "'moodfx' passes through untouched.");
        return false;
    }
    try {
        g_mood.net = load_net(path);
        set_net_target(g_mood.net);
        g_mood.ok = true;
    } catch (const std::exception &e) {
        warn_once("fer_load", std::string("failed to load FER model: ") + e.what());
    }
    return g_mood.ok;
}


/* Lazily construct the YuNet detector sized to the (downscaled) frame.
 * Returns false and warns once if the model file is absent/unloadable. */
bool ensure_face(int w, int h)
{
    if (g_face.tried && !g_face.ok) return false;
    if (g_face.ok && g_face.net_w == w && g_face.net_h == h) return true;

    std::string path = model_dir() + "/face_detection_yunet_2023mar.onnx";
    if (!path_exists(path)) {
        g_face.tried = true; g_face.ok = false;
        warn_once("face_model",
            "face model not found: " + path +
            "  — run 'vcam-setup --models' to download it. "
            "The 'facefx' filter will pass through untouched.");
        return false;
    }
    try {
        g_face.det = cv::FaceDetectorYN::create(
            path, "", cv::Size(w, h),
            0.7f, 0.3f, 5000);
        g_face.net_w = w; g_face.net_h = h;
        g_face.tried = true; g_face.ok = true;
        return true;
    } catch (const std::exception &e) {
        g_face.tried = true; g_face.ok = false;
        warn_once("face_load", std::string("failed to load face model: ") + e.what());
        return false;
    }
}

bool ensure_seg()
{
    if (g_seg.tried && !g_seg.ok) return false;
    if (g_seg.ok) return true;

    std::string path = model_dir() + "/human_segmentation_pphumanseg_2023mar.onnx";
    if (!path_exists(path)) {
        g_seg.tried = true; g_seg.ok = false;
        warn_once("seg_model",
            "segmentation model not found: " + path +
            "  — run 'vcam-setup --models' to download it. "
            "The 'personfx' filter will pass through untouched.");
        return false;
    }
    try {
        g_seg.net = load_net(path);   /* OpenCV 5: new graph engine (ENGINE_AUTO) */
        set_net_target(g_seg.net);
        g_seg.tried = true; g_seg.ok = true;
#if VCAM_CV5
        warn_once("seg_engine",
            "segmentation model loaded on the OpenCV 5 DNN graph engine");
#endif
        return true;
    } catch (const std::exception &e) {
        g_seg.tried = true; g_seg.ok = false;
        warn_once("seg_load", std::string("failed to load seg model: ") + e.what());
        return false;
    }
}

/* PP-HumanSeg: input 192x192, two-channel softmax output (bg, person).
 * Returns a full-frame CV_8U mask (255 = person) or empty on failure.
 * Preprocessing: (pixel/255 - 0.5)/0.5  ==  pixel/127.5 - 1, which
 * blobFromImage produces as scalefactor=1/127.5, mean=127.5. */
/* Try RVM once: load lazily; any exception marks it failed forever
 * (this session) and PP-HumanSeg takes over. Returns empty on failure.
 *
 * Backend priority:
 *   1. ONNX Runtime (ORT=1 builds) — the reference runtime; computes
 *      RVM correctly. Input 512-wide for best quality.
 *   2. OpenCV DNN — kept as a fallback; on current OpenCV 5 graph
 *      engines it loads but MISCOMPUTES (empty matte), which the
 *      dead-matte gate below detects, falling through to PP-HumanSeg. */
cv::Mat rvm_infer(const cv::Mat &bgr, int size_w)
{
    if (g_seg.rvm_state < 0) return cv::Mat();

    /* ── path 1: ONNX Runtime ── */
    if (ort_available() && g_seg.ort_state >= 0) {
        try {
            std::string path = model_dir() + "/rvm_mobilenetv3_fp32.onnx";
            if (!path_exists(path)) {
                g_seg.ort_state = -1;         /* no model; don't retry */
            } else if (g_seg.ort_state == 0) {
                g_seg.ort_state = ort_rvm_init(path.c_str()) ? 1 : -1;
                if (g_seg.ort_state == 1)
                    warn_once("rvm_ort",
                        "personfx: RVM running on the ONNX Runtime backend "
                        "(reference runtime — correct, temporally-stable mattes)");
            }
            if (g_seg.ort_state == 1) {
                /* size_w = refinement resolution (the quality knob).
                 * downsample_ratio pins the heavy encoder at ~256px
                 * regardless — RVM's intended realtime operation, which
                 * we can finally use because ORT computes it right. */
                int IW = size_w;
                double sc = (double)IW / bgr.cols;
                int iw = std::max(64, ((int)(bgr.cols*sc)/16)*16);
                int ih = std::max(64, ((int)(bgr.rows*sc)/16)*16);
                float ds = (float)std::min(1.0, std::max(0.25, 256.0 / iw));
                cv::Mat small2;
                cv::resize(bgr, small2, cv::Size(iw, ih), 0, 0, cv::INTER_AREA);
                cv::Mat src = cv::dnn::blobFromImage(small2, 1.0/255.0,
                                  cv::Size(iw, ih), cv::Scalar(), true, false);
                cv::Mat pha(ih, iw, CV_32F);
                if (ort_rvm_infer((const float*)src.ptr<float>(0), iw, ih, ds,
                                  pha.ptr<float>(0))) {
                    const char *dbg = std::getenv("VCAM_CV_DEBUG");
                    if (dbg && *dbg=='1' && g_seg.counter < 6)
                        std::fprintf(stderr, "[vcam/ort] pha mean=%.4f\n",
                                     cv::mean(pha)[0]);
                    cv::Mat full;
                    cv::resize(pha, full, bgr.size(), 0, 0, cv::INTER_LINEAR);
                    cv::min(full, 1.0, full);
                    cv::max(full, 0.0, full);
                    return full;
                }
                g_seg.ort_state = -1;         /* backend died; fall through */
            }
        } catch (const std::exception &e) {
            g_seg.ort_state = -1;
            warn_once("rvm_ort_fail",
                std::string("ORT RVM path failed (") + e.what() +
                ") — trying other backends");
        }
    }

    /* ── path 2: OpenCV DNN (may miscompute on 5.x; gated below) ── */
    try {
        if (g_seg.rvm_state == 0) {
            std::string path = model_dir() + "/rvm_mobilenetv3_fp32.onnx";
            if (!path_exists(path)) { g_seg.rvm_state = -1; return cv::Mat(); }
            g_seg.rvm = load_net(path);
            set_net_target(g_seg.rvm);
            g_seg.rvm_state = 1;
            warn_once("rvm_on",
                "personfx: RVM video matting model found — using it "
                "(temporally-stable, hair-level mattes)");
        }
        /* input ~384 wide with downsample_ratio=1.0: bypasses RVM's
         * internal dynamic-resize path entirely (prime suspect for
         * graph-engine miscompilation -> dead matte). Smaller src keeps
         * the compute cost similar to 512 @ ds 0.6. */
        int IW = 384;
        double sc = (double)IW / bgr.cols;
        int iw = std::max(64, ((int)(bgr.cols*sc)/16)*16);
        int ih = std::max(64, ((int)(bgr.rows*sc)/16)*16);
        cv::Mat small2;
        cv::resize(bgr, small2, cv::Size(iw, ih), 0, 0, cv::INTER_AREA);
        /* src: RGB, 0..1, NCHW fp32 */
        cv::Mat src = cv::dnn::blobFromImage(small2, 1.0/255.0, cv::Size(iw, ih),
                                             cv::Scalar(), true, false);
        if (g_seg.r1.empty()) {
            int shp[4] = {1,1,1,1};
            g_seg.r1 = cv::Mat(4, shp, CV_32F, cv::Scalar(0));
            g_seg.r2 = g_seg.r1.clone();
            g_seg.r3 = g_seg.r1.clone();
            g_seg.r4 = g_seg.r1.clone();
        }
        float dsr[1] = {1.0f};
        int dshp[1] = {1};
        cv::Mat ds(1, dshp, CV_32F, dsr);
        g_seg.rvm.setInput(src,      "src");
        g_seg.rvm.setInput(g_seg.r1, "r1i");
        g_seg.rvm.setInput(g_seg.r2, "r2i");
        g_seg.rvm.setInput(g_seg.r3, "r3i");
        g_seg.rvm.setInput(g_seg.r4, "r4i");
        g_seg.rvm.setInput(ds,       "downsample_ratio");
        /* OpenCV 5's graph engine requires requesting ALL 6 outputs.
         * BUT do not trust that outs[] order matches our name order —
         * identify tensors BY SHAPE, which is unambiguous:
         *   pha = 4D, channels==1, spatial == input size
         *   fgr = 4D, channels==3
         *   recurrents = everything else (4 tensors, C>1, smaller). */
        std::vector<cv::Mat> outs;
        std::vector<cv::String> names = {"fgr","pha","r1o","r2o","r3o","r4o"};
        g_seg.rvm.forward(outs, names);
        if (outs.size() < 6) { g_seg.rvm_state = -1; return cv::Mat(); }
        const char *dbg = std::getenv("VCAM_CV_DEBUG");
        int pha_i = -1;
        std::vector<int> rec_i;
        for (int oi = 0; oi < (int)outs.size(); oi++) {
            const cv::Mat &o = outs[oi];
            if (o.dims != 4) continue;
            if (dbg && *dbg=='1' && g_seg.counter < 3)
                std::fprintf(stderr, "[vcam/rvm] out[%d] shape %dx%dx%dx%d\n",
                    oi, o.size[0], o.size[1], o.size[2], o.size[3]);
            if (o.size[1] == 1 && o.size[2] == ih && o.size[3] == iw)
                pha_i = oi;
            else if (o.size[1] != 3)
                rec_i.push_back(oi);
        }
        if (pha_i < 0) { g_seg.rvm_state = -1; return cv::Mat(); }
        if (rec_i.size() == 4) {
            /* feed the recurrents back in the order found */
            outs[rec_i[0]].copyTo(g_seg.r1);
            outs[rec_i[1]].copyTo(g_seg.r2);
            outs[rec_i[2]].copyTo(g_seg.r3);
            outs[rec_i[3]].copyTo(g_seg.r4);
        } else {
            /* can't identify them safely -> run STATELESS: reset the
             * states to zeros every frame (loses temporal memory, but
             * the matte itself stays correct). */
            g_seg.r1.release(); g_seg.r2.release();
            g_seg.r3.release(); g_seg.r4.release();
            warn_once("rvm_stateless",
                "RVM: recurrent outputs ambiguous — running stateless");
        }
        int oh = outs[pha_i].size[2], ow = outs[pha_i].size[3];
        cv::Mat pha(oh, ow, CV_32F, (float*)outs[pha_i].ptr<float>(0));
        double pmean = cv::mean(pha)[0];
        if (dbg && *dbg=='1' && g_seg.counter < 6)
            std::fprintf(stderr, "[vcam/rvm] pha=out[%d] mean=%.4f\n", pha_i, pmean);
        /* AUTO QUALITY GATE: a matte that stays essentially empty for
         * 30 straight frames means the engine miscomputes this model.
         * Declare RVM broken and fall back — never a black screen. */
        static int dead = 0;
        if (pmean < 0.005) {
            if (++dead >= 30) {
                g_seg.rvm_state = -1;
                g_seg.r1.release(); g_seg.r2.release();
                g_seg.r3.release(); g_seg.r4.release();
                warn_once("rvm_dead",
                    "RVM produced an empty matte for 30 frames — this "
                    "OpenCV build miscomputes it. Falling back to "
                    "PP-HumanSeg. (ONNX Runtime backend would fix this.)");
                return cv::Mat();
            }
        } else dead = 0;
        cv::Mat full;
        cv::resize(pha, full, bgr.size(), 0, 0, cv::INTER_LINEAR);
        cv::min(full, 1.0, full);
        cv::max(full, 0.0, full);
        return full.clone();
    } catch (const std::exception &e) {
        g_seg.rvm_state = -1;
        g_seg.r1.release(); g_seg.r2.release(); g_seg.r3.release(); g_seg.r4.release();
        warn_once("rvm_fail",
            std::string("RVM could not run on this OpenCV build (") + e.what() +
            ") — falling back to PP-HumanSeg. On OpenCV 5 the new graph "
            "engine usually loads it; on 4.x it typically cannot.");
        return cv::Mat();
    }
}

cv::Mat seg_infer(const cv::Mat &bgr, const cv::Rect &body, float edge)
{
    /* Segment only the body ROI (from the face detector), LETTERBOXED
     * into the square input instead of aspect-squished. The person now
     * fills the model's view the way its selfie training data did —
     * this is the difference between a mushy mask and a crisp one. */
    const int N = 192;
    cv::Mat crop = bgr(body);
    double s2 = (double)N / std::max(crop.cols, crop.rows);
    int lw = std::max(8, (int)(crop.cols * s2));
    int lh = std::max(8, (int)(crop.rows * s2));
    cv::Mat resized(N, N, CV_8UC3, cv::Scalar(114,114,114));
    int lx = (N - lw)/2, ly = (N - lh)/2;
    {
        cv::Mat dst2 = resized(cv::Rect(lx, ly, lw, lh));
        cv::resize(crop, dst2, dst2.size(), 0, 0, cv::INTER_AREA);
    }
    cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / 127.5, cv::Size(N, N),
        cv::Scalar(127.5, 127.5, 127.5), true, false);

    g_seg.net.setInput(blob);
    cv::Mat out = g_seg.net.forward();   /* shape [1,2,192,192] */
    if (out.dims != 4 || out.size[1] < 2) return cv::Mat();

    /* Robust channel access: build a 2D header onto each channel plane
     * from the raw data pointer (out.ptr<float>(0,c) is fragile on a 4D
     * Mat). Plane stride is N*N floats. */
    const int HW = N * N;
    float *base = (float *)out.ptr<float>(0);
    cv::Mat ch0(N, N, CV_32F, base);
    cv::Mat ch1(N, N, CV_32F, base + HW);

    /* Which channel is the PERSON? PP-HumanSeg's channel order isn't
     * guaranteed, and getting it backwards inverts the mask (the exact
     * "half the body is cut" symptom). The order is FIXED for a given
     * model, so we vote over the first frames and then lock it: the
     * person channel is hotter in the CENTER of the letterboxed crop
     * (person is centered; background hugs the corners). */
    auto center_minus_border = [&](const cv::Mat &c) -> float {
        int m = N / 4;
        cv::Scalar cen = cv::mean(c(cv::Rect(m, m, N - 2*m, N - 2*m)));
        cv::Scalar whole = cv::mean(c);
        return (float)(cen[0] - whole[0]);
    };
    cv::Mat mx = cv::max(ch0, ch1);
    cv::Mat e0, e1;
    cv::exp(ch0 - mx, e0);
    cv::exp(ch1 - mx, e1);
    cv::Mat sum = e0 + e1;
    cv::Mat p0 = e0 / sum;
    cv::Mat p1 = e1 / sum;
    if (g_seg.person_ch < 0) {                 /* not locked yet: vote */
        if (center_minus_border(p1) >= center_minus_border(p0))
            g_seg.ch_vote++;
        else
            g_seg.ch_vote--;
        if (++g_seg.ch_frames >= 8)            /* lock after 8 frames */
            g_seg.person_ch = (g_seg.ch_vote >= 0) ? 1 : 0;
    }
    /* until locked, use the current best guess */
    int pc = (g_seg.person_ch >= 0) ? g_seg.person_ch
             : ((center_minus_border(p1) >= center_minus_border(p0)) ? 1 : 0);
    cv::Mat prob = (pc == 1) ? p1 : p0;
    /* un-letterbox → back into the body ROI → into a full-frame canvas.
     * NOTE: no blur here — we sharpen the alpha FIRST (below), then apply
     * a single narrow feather. Blurring before the opacity remap was what
     * spread the translucent halo around the whole silhouette. */
    cv::Mat inner = prob(cv::Rect(lx, ly, lw, lh));
    cv::Mat roi_mask;
    cv::resize(inner, roi_mask, body.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat full = cv::Mat::zeros(bgr.size(), CV_32F);
    roi_mask.copyTo(full(body));

    /* connected-component cleanup at 1/4 res: keep the person blob(s),
     * drop stray background false-positives (e.g. the leaking corner). */
    cv::Mat q, q8, lbl, stats, cent;
    cv::resize(full, q, cv::Size(), 0.25, 0.25, cv::INTER_LINEAR);
    q8 = q > 0.30;
    int n2 = cv::connectedComponentsWithStats(q8, lbl, stats, cent);
    if (n2 > 1) {
        int ba = 0;
        for (int i2 = 1; i2 < n2; i2++)
            ba = std::max(ba, stats.at<int>(i2, cv::CC_STAT_AREA));
        /* keep every component with >=15% of the largest's area (head +
         * torso + raised arm survive; small false blobs like the leaking
         * corner are discarded). */
        cv::Mat keep = cv::Mat::zeros(lbl.size(), CV_8U);
        for (int i2 = 1; i2 < n2; i2++) {
            int a2 = stats.at<int>(i2, cv::CC_STAT_AREA);
            if (a2 >= std::max(40, ba * 15 / 100))
                keep |= (lbl == i2);
        }
        cv::Mat keepfull;
        cv::resize(keep, keepfull, full.size(), 0, 0, cv::INTER_NEAREST);
        cv::Mat keepf; keepfull.convertTo(keepf, CV_32F, 1.0/255.0);
        full = full.mul(keepf);
    }

    /* OPACITY: remap so confident interior -> fully solid, with a NARROW
     * transition band. 'edge' sets the midpoint (lower = more hair kept);
     * the /0.14 makes the ramp steep, so the person is opaque and only a
     * thin rim is soft — no more see-through halo. */
    full = (full - edge) * (1.0 / 0.14);
    cv::min(full, 1.0, full);
    cv::max(full, 0.0, full);
    cv::Mat f2 = full.mul(full);
    full = f2 * 3.0 - f2.mul(full) * 2.0;      /* smoothstep 3t^2 - 2t^3 */

    /* Pull the edge inward by ~1px so the soft rim sits just INSIDE the
     * true silhouette (prevents the 1-2px background halo from bleeding
     * through), then a single tight feather for anti-aliasing. */
    cv::erode(full, full, cv::Mat(), cv::Point(-1,-1), 1);
    cv::GaussianBlur(full, full, cv::Size(0, 0), std::max(1.0, bgr.cols / 480.0));
    return full;
}

} // namespace

/* ════════════════════════════════════════════════════════════
 * C ABI
 * ════════════════════════════════════════════════════════════ */

extern "C" {

bool cv_available(void) { return true; }

const char *cv_version(void) { return CV_VERSION; }

const char *cv_engine_info(void)
{
#if VCAM_CV5
    return "OpenCV " CV_VERSION " — new DNN graph engine (ENGINE_AUTO), FP16 blend";
#else
    return "OpenCV " CV_VERSION " — classic DNN engine, FP32 blend "
           "(build against OpenCV 5 for the new graph engine)";
#endif
}

const char *cv_model_dir(void) { return model_dir().c_str(); }

/* ── classic effects ── */

void cv_fx_cartoon(unsigned char *data, int w, int h, int stride, float strength)
{
    if (!data || w < 4 || h < 4) return;
    if (strength <= 0.0f || strength > 1.0f) strength = 0.6f;
    try {
        cv::Mat img = wrap_bgr(data, w, h, stride);
        cv::Mat small, smooth;
        /* bilateral is expensive; run it at half-res then upscale */
        cv::resize(img, small, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        int d = 9;
        double sc = 50 + 120 * strength;
        cv::bilateralFilter(small, smooth, d, sc, sc);
        cv::resize(smooth, smooth, img.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat gray, edges;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::medianBlur(gray, gray, 7);
        cv::adaptiveThreshold(gray, edges, 255,
            cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 2);
        cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
        cv::bitwise_and(smooth, edges, img);   /* writes back into frame */
    } catch (const std::exception &e) {
        warn_once("cartoon", std::string("cartoon failed: ") + e.what());
    }
}

void cv_fx_pencil(unsigned char *data, int w, int h, int stride, int color)
{
    if (!data || w < 4 || h < 4) return;
    try {
        cv::Mat img = wrap_bgr(data, w, h, stride);
        cv::Mat gray1, color1;
        cv::pencilSketch(img, gray1, color1,
                         60.0f, 0.07f, 0.02f);
        if (color) color1.copyTo(img);
        else cv::cvtColor(gray1, img, cv::COLOR_GRAY2BGR);
    } catch (const std::exception &e) {
        warn_once("pencil", std::string("pencil failed: ") + e.what());
    }
}

void cv_fx_stylize(unsigned char *data, int w, int h, int stride, float strength)
{
    if (!data || w < 4 || h < 4) return;
    if (strength <= 0.0f || strength > 1.0f) strength = 0.6f;
    try {
        cv::Mat img = wrap_bgr(data, w, h, stride);
        cv::Mat out;
        cv::stylization(img, out, 60.0f, 0.45f * strength + 0.1f);
        out.copyTo(img);
    } catch (const std::exception &e) {
        warn_once("stylize", std::string("stylize failed: ") + e.what());
    }
}

void cv_fx_flow(unsigned char *data, int w, int h, int stride, float mix)
{
    if (!data || w < 8 || h < 8) return;
    if (mix <= 0.0f || mix > 1.0f) mix = 0.6f;
    try {
        std::lock_guard<std::mutex> lock(g_classic.mtx);
        cv::Mat img = wrap_bgr(data, w, h, stride);
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        /* flow on a downscaled gray for speed */
        cv::Mat gsmall;
        double s = 0.5;
        cv::resize(gray, gsmall, cv::Size(), s, s, cv::INTER_LINEAR);

        if (g_classic.prev_gray.empty() ||
            g_classic.prev_gray.size() != gsmall.size()) {
            gsmall.copyTo(g_classic.prev_gray);
            return;   /* need two frames */
        }

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(
            g_classic.prev_gray, gsmall, flow,
            0.5, 2, 13, 3, 5, 1.1, 0);
        gsmall.copyTo(g_classic.prev_gray);

        /* map flow → HSV (hue=direction, val=magnitude) */
        cv::Mat parts[2];
        cv::split(flow, parts);
        cv::Mat mag, ang;
        cv::cartToPolar(parts[0], parts[1], mag, ang, true);
        cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
        cv::Mat hsv(gsmall.size(), CV_8UC3);
        for (int y = 0; y < hsv.rows; y++) {
            for (int x = 0; x < hsv.cols; x++) {
                float a = ang.at<float>(y, x);
                float m = mag.at<float>(y, x);
                hsv.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(a / 2.0f),      /* H 0..180 */
                    255,
                    static_cast<uchar>(m));            /* V */
            }
        }
        cv::Mat flowbgr;
        cv::cvtColor(hsv, flowbgr, cv::COLOR_HSV2BGR);
        cv::resize(flowbgr, flowbgr, img.size(), 0, 0, cv::INTER_LINEAR);
        cv::addWeighted(flowbgr, mix, img, 1.0 - mix, 0.0, img);
    } catch (const std::exception &e) {
        warn_once("flow", std::string("flow failed: ") + e.what());
    }
}

void cv_fx_bgsub(unsigned char *data, int w, int h, int stride, float sensitivity)
{
    if (!data || w < 8 || h < 8) return;
    if (sensitivity <= 0.0f || sensitivity > 1.0f) sensitivity = 0.5f;
    try {
        std::lock_guard<std::mutex> lock(g_classic.mtx);
        cv::Mat img = wrap_bgr(data, w, h, stride);
        if (!g_classic.mog2) {
            g_classic.mog2 = cv::createBackgroundSubtractorMOG2(
                500, 16.0 + 40.0 * (1.0 - sensitivity), true);
        }
        cv::Mat fg;
        g_classic.mog2->apply(img, fg, -1);
        /* clean the mask */
        cv::threshold(fg, fg, 128, 255, cv::THRESH_BINARY);
        cv::medianBlur(fg, fg, 5);
        cv::Mat fg3;
        cv::cvtColor(fg, fg3, cv::COLOR_GRAY2BGR);

        /* dim + desaturate the background, keep foreground vivid */
        cv::Mat gray, dim;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, dim, cv::COLOR_GRAY2BGR);
        dim *= 0.35;

        /* out = fg ? img : dim */
        cv::Mat inv;
        cv::bitwise_not(fg3, inv);
        cv::Mat keep, drop;
        cv::bitwise_and(img, fg3, keep);
        cv::bitwise_and(dim, inv, drop);
        cv::add(keep, drop, img);
    } catch (const std::exception &e) {
        warn_once("bgsub", std::string("bgsub failed: ") + e.what());
    }
}

/* ── DNN effects ── */

void cv_fx_face(unsigned char *data, int w, int h, int stride, int mode,
                float amount, float speed)
{
    if (!data || w < 16 || h < 16) return;
    if (amount <= 0.0f) amount = 1.0f;       /* effect size/intensity */
    if (amount < 0.3f) amount = 0.3f;
    if (amount > 2.5f) amount = 2.5f;
    if (speed <= 0.0f) speed = 1.0f;         /* animation rate */
    if (speed < 0.2f) speed = 0.2f;
    if (speed > 3.0f) speed = 3.0f;
    float T = (float)g_face.counter * speed; /* knob-scaled time base */
    try {
        std::lock_guard<std::mutex> lock(g_face.mtx);

        /* run detection on a downscaled copy for speed, then scale
         * boxes back up */
        const int TARGET_W = 320;
        double s = (w > TARGET_W) ? (double)TARGET_W / w : 1.0;
        int dw = std::max(16, (int)(w * s));
        int dh = std::max(16, (int)(h * s));

        cv::Mat img = wrap_bgr(data, w, h, stride);

        if (!ensure_face(dw, dh)) return;   /* model missing → no-op */

        if (tick(g_face.counter) % INFER_EVERY == 0) {
            cv::Mat small;
            cv::resize(img, small, cv::Size(dw, dh));
            cv::Mat faces;
            g_face.det->detect(small, faces);
            faces.copyTo(g_face.faces);
        }
        if (g_face.faces.empty()) return;

        double inv = 1.0 / s;
        for (int i = 0; i < g_face.faces.rows; i++) {
            const float *f = g_face.faces.ptr<float>(i);
            int x = (int)(f[0] * inv), y = (int)(f[1] * inv);
            int fw = (int)(f[2] * inv), fh = (int)(f[3] * inv);
            /* clamp */
            x = std::max(0, std::min(x, w - 1));
            y = std::max(0, std::min(y, h - 1));
            fw = std::max(1, std::min(fw, w - x));
            fh = std::max(1, std::min(fh, h - y));
            cv::Rect roi(x, y, fw, fh);

            if (mode == 0) {
                /* privacy: pixelate the face box */
                cv::Mat face = img(roi);
                int bx = std::max(1, fw / 12), by = std::max(1, fh / 12);
                cv::Mat tiny;
                cv::resize(face, tiny, cv::Size(bx, by), 0, 0, cv::INTER_LINEAR);
                cv::resize(tiny, face, face.size(), 0, 0, cv::INTER_NEAREST);
            } else if (mode == 1) {
                /* glossy ring + landmark dots */
                cv::ellipse(img, cv::Point(x + fw/2, y + fh/2),
                            cv::Size(fw/2, fh/2), 0, 0, 360,
                            cv::Scalar(60, 220, 255), std::max(2, fw/40));
                /* landmarks start at index 4: re,le,nose,rmouth,lmouth */
                for (int k = 4; k + 1 < 14; k += 2) {
                    int lx = (int)(f[k] * inv), ly = (int)(f[k+1] * inv);
                    cv::circle(img, cv::Point(lx, ly), std::max(2, fw/30),
                               cv::Scalar(255, 255, 255), -1);
                }
            } else if (mode == 3 || mode == 6) {
                /* BIGHEAD (3) / BOBBLEHEAD (6: scale oscillates): magnify the head in place, tracked live.
                 * Grab a region around the face, enlarge it, paste it
                 * back centered with an elliptical feather. */
                float scale = 1.0f + 0.45f * amount;
                if (mode == 6)
                    scale = 1.0f + 0.30f * amount + 0.28f * amount * std::sin(T * 0.35f);
                int ex = std::max(0, x - fw/3), ey = std::max(0, y - fh/2);
                int ew = std::min(w - ex, fw + 2*fw/3);
                int eh = std::min(h - ey, fh + fh);
                if (ew > 8 && eh > 8) {
                    cv::Rect big(ex, ey, ew, eh);
                    int cw = (int)(ew / scale), ch = (int)(eh / scale);
                    int cx0 = ex + (ew - cw)/2, cy0 = ey + (eh - ch)/2;
                    cv::Mat zoom;
                    cv::resize(img(cv::Rect(cx0, cy0, cw, ch)), zoom,
                               big.size(), 0, 0, cv::INTER_LINEAR);
                    /* elliptical feather */
                    for (int yy = 0; yy < eh; yy++) {
                        cv::Vec3b *dst2 = img.ptr<cv::Vec3b>(ey+yy) + ex;
                        const cv::Vec3b *sp2 = zoom.ptr<cv::Vec3b>(yy);
                        float ny = (yy - eh/2.f) / (eh/2.f);
                        for (int xx = 0; xx < ew; xx++) {
                            float nx = (xx - ew/2.f) / (ew/2.f);
                            float d = nx*nx + ny*ny;
                            if (d >= 1.0f) continue;
                            float a = d < 0.55f ? 1.0f : (1.0f - d) / 0.45f;
                            dst2[xx][0]=(uchar)(dst2[xx][0]+(sp2[xx][0]-dst2[xx][0])*a);
                            dst2[xx][1]=(uchar)(dst2[xx][1]+(sp2[xx][1]-dst2[xx][1])*a);
                            dst2[xx][2]=(uchar)(dst2[xx][2]+(sp2[xx][2]-dst2[xx][2])*a);
                        }
                    }
                }
            } else if (mode == 4) {
                /* DEAL-WITH-IT: pixel sunglasses on the eye landmarks */
                cv::Point re((int)(f[4]*inv), (int)(f[5]*inv));
                cv::Point le((int)(f[6]*inv), (int)(f[7]*inv));
                int lw = std::max(8, fw/3);          /* lens width */
                int lh = std::max(4, lw/2);
                cv::Scalar black(10,10,10);
                for (const cv::Point &e : {re, le}) {
                    cv::rectangle(img, cv::Rect(e.x-lw/2, e.y-lh/2, lw, lh),
                                  black, cv::FILLED);
                    /* gloss */
                    cv::rectangle(img, cv::Rect(e.x-lw/2+2, e.y-lh/2+2,
                                  std::max(2,lw/4), std::max(1,lh/5)),
                                  cv::Scalar(120,120,120), cv::FILLED);
                }
                cv::line(img, cv::Point(re.x, re.y), cv::Point(le.x, le.y),
                         black, std::max(2, lh/4));  /* bridge */
                cv::line(img, cv::Point(std::min(re.x,le.x)-lw/2, (re.y+le.y)/2),
                         cv::Point(x, y + fh/4), black, std::max(2, lh/4));
                cv::line(img, cv::Point(std::max(re.x,le.x)+lw/2, (re.y+le.y)/2),
                         cv::Point(x+fw, y + fh/4), black, std::max(2, lh/4));
            } else if (mode == 7) {
                /* THIRD EYE: your right eye, cloned onto your forehead
                 * with a circular feather. Deeply unsettling. */
                cv::Point re((int)(f[4]*inv), (int)(f[5]*inv));
                int r3 = std::max(6, fw/5);
                cv::Point dst3(x + fw/2, std::max(r3, y + fh/8));
                if (re.x-r3 >= 0 && re.y-r3 >= 0 && re.x+r3 < w && re.y+r3 < h &&
                    dst3.x-r3 >= 0 && dst3.y-r3 >= 0 && dst3.x+r3 < w && dst3.y+r3 < h) {
                    cv::Mat patch = img(cv::Rect(re.x-r3, re.y-r3, 2*r3, 2*r3)).clone();
                    for (int yy = 0; yy < 2*r3; yy++) {
                        cv::Vec3b *dr = img.ptr<cv::Vec3b>(dst3.y-r3+yy) + (dst3.x-r3);
                        const cv::Vec3b *sr = patch.ptr<cv::Vec3b>(yy);
                        float ny = (yy - r3) / (float)r3;
                        for (int xx = 0; xx < 2*r3; xx++) {
                            float nx = (xx - r3) / (float)r3;
                            float d = nx*nx + ny*ny;
                            if (d >= 1.0f) continue;
                            float a = d < 0.5f ? 1.0f : (1.0f-d)/0.5f;
                            dr[xx][0]=(uchar)(dr[xx][0]+(sr[xx][0]-dr[xx][0])*a);
                            dr[xx][1]=(uchar)(dr[xx][1]+(sr[xx][1]-dr[xx][1])*a);
                            dr[xx][2]=(uchar)(dr[xx][2]+(sr[xx][2]-dr[xx][2])*a);
                        }
                    }
                }
            } else if (mode == 8) {
                /* CROWN: tracked golden crown with jewels above your head */
                int by2 = std::max(6, y - fh/6);          /* band bottom */
                int bh2 = std::max(4, fh/8);
                int ph = std::max(6, fh/4);               /* point height */
                cv::Scalar gold(0, 190, 250), dark(0, 120, 180);
                cv::rectangle(img, cv::Rect(x, by2-bh2, fw, bh2), gold, cv::FILLED);
                cv::rectangle(img, cv::Rect(x, by2-bh2, fw, bh2), dark, 2);
                for (int k2 = 0; k2 < 4; k2++) {
                    int px0 = x + (fw*k2)/4, px1 = x + (fw*(k2+1))/4;
                    cv::Point tri[3] = { {px0, by2-bh2}, {px1, by2-bh2},
                                         {(px0+px1)/2, by2-bh2-ph} };
                    cv::fillConvexPoly(img, tri, 3, gold);
                    cv::circle(img, cv::Point((px0+px1)/2, by2-bh2-ph+3),
                               std::max(2, fw/40),
                               k2%2 ? cv::Scalar(60,60,230) : cv::Scalar(230,90,60),
                               cv::FILLED);
                }
            } else if (mode == 5) {
                /* LASER EYES: beams from both eyes along the nose->eye
                 * axis, extended to the frame edge, with additive glow. */
                cv::Point nose((int)(f[8]*inv), (int)(f[9]*inv));
                for (int e = 0; e < 2; e++) {
                    cv::Point eye((int)(f[4+e*2]*inv), (int)(f[5+e*2]*inv));
                    float ddx = (float)(eye.x - nose.x);
                    float ddy = (float)(eye.y - nose.y) - fh * 0.25f;
                    float L = std::sqrt(ddx*ddx + ddy*ddy);
                    if (L < 1) { ddx = e ? -1.f : 1.f; ddy = -0.3f; L = 1; }
                    cv::Point tip(eye.x + (int)(ddx/L * (w+h)),
                                  eye.y + (int)(ddy/L * (w+h)));
                    cv::line(img, eye, tip, cv::Scalar(20,20,140), std::max(6, (int)(fw/8*amount)));
                    cv::line(img, eye, tip, cv::Scalar(40,40,255), std::max(3, (int)(fw/16*amount)));
                    cv::line(img, eye, tip, cv::Scalar(220,220,255), std::max(1, fw/40));
                    cv::circle(img, eye, std::max(4, fw/10),
                               cv::Scalar(60,60,255), cv::FILLED);
                    cv::circle(img, eye, std::max(2, fw/20),
                               cv::Scalar(255,255,255), cv::FILLED);
                }
            } else if (mode == 24) {
                /* FACE IMAGE: your uploaded PNG (animal face, mask,
                 * meme...) tracked to your face — scaled by face width,
                 * rotated by your eye angle, alpha-composited.
                 * Set it with: vcam-ctl filter faceimage <path>
                 * amount = overlay size. */
                std::lock_guard<std::mutex> fl2(g_faceimg.mtx);
                if (!g_faceimg.rgba.empty()) {
                    double ang = std::atan2((double)(f[7]-f[5]),
                                            (double)(f[6]-f[4])) * 57.2958;
                    /* ── AR life signs, all from YuNet's 5 landmarks ──
                     * jaw: nose→mouth distance vs. its running "closed"
                     * baseline -> mouth-open amount; yaw: nose offset
                     * from the eye midpoint -> head turn. */
                    float emx = (f[4]+f[6])*0.5f*inv;
                    float nsx = f[8]*inv, nsy = f[9]*inv;
                    float mmx = (f[10]+f[12])*0.5f*inv, mmy2 = (f[11]+f[13])*0.5f*inv;
                    float nm = std::sqrt((mmx-nsx)*(mmx-nsx)+(mmy2-nsy)*(mmy2-nsy))
                             / std::max(1.0f, (float)fh);
                    static float nm_base = -1;
                    if (nm_base < 0) nm_base = nm;
                    if (nm < nm_base) nm_base += (nm - nm_base) * 0.10f;
                    else              nm_base += (nm - nm_base) * 0.004f;
                    float open2 = std::max(0.0f, std::min(1.0f,
                                  (nm - nm_base*1.06f) / (nm_base*0.45f + 1e-4f)));
                    float yaw = (nsx - emx) / std::max(1.0f, (float)fw);  /* ± ~0.3 */
                    /* lighting match: your face region's light tints the mask */
                    float lr2=0, lg2=0, lb2=0; int ln2=0;
                    for (int yy = y; yy < y+fh; yy += 8)
                        for (int xx = x; xx < x+fw; xx += 8) {
                            if (xx<0||yy<0||xx>=w||yy>=h) continue;
                            const cv::Vec3b &pv = img.at<cv::Vec3b>(yy,xx);
                            lb2+=pv[0]; lg2+=pv[1]; lr2+=pv[2]; ln2++;
                        }
                    float kb=1, kg=1, kr=1;
                    if (ln2 > 4) {
                        kb = std::min(1.35f, std::max(0.55f, (lb2/ln2)/150.0f));
                        kg = std::min(1.35f, std::max(0.55f, (lg2/ln2)/150.0f));
                        kr = std::min(1.35f, std::max(0.55f, (lr2/ln2)/150.0f));
                    }
                    int S = std::max(24, (int)(fw * 1.9f * amount));
                    int cx3 = x + fw/2 + (int)(yaw * S * 0.45f);   /* turn */
                    int cy3 = y + fh/2 - fh/12;
                    int x0 = cx3 - S/2, y0 = cy3 - S/2;
                    if (x0 >= 0 && y0 >= 0 && x0+S < w && y0+S < h) {
                        /* affine: PNG -> S x S canvas, rotated (manual
                         * 2x3, version-proof), transparent border */
                        double sc2 = (double)S / std::max(g_faceimg.rgba.cols,
                                                          g_faceimg.rgba.rows);
                        double ra = -ang * 3.14159265 / 180.0;
                        double ca2 = std::cos(ra)*sc2, sa2 = std::sin(ra)*sc2;
                        double pcx = g_faceimg.rgba.cols/2.0;
                        double pcy = g_faceimg.rgba.rows/2.0;
                        double abuf[6] = { ca2, sa2, S/2.0 - ca2*pcx - sa2*pcy,
                                          -sa2, ca2, S/2.0 + sa2*pcx - ca2*pcy };
                        cv::Mat A2(2, 3, CV_64F, abuf);
                        cv::Mat canvas;
                        cv::warpAffine(g_faceimg.rgba, canvas, A2,
                                       cv::Size(S, S), cv::INTER_LINEAR,
                                       cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0));
                        /* composite with a live jaw: the bottom of the
                         * mask drops when YOUR mouth opens; the gap is
                         * a darkened mouth interior. */
                        int split = (int)(S * 0.60f);
                        int drop = (int)(open2 * S * 0.13f);
                        for (int yy = 0; yy < S; yy++) {
                            int sy3;                     /* source canvas row */
                            float dark = 1.0f;
                            if (yy < split)              sy3 = yy;
                            else if (yy < split + drop) { sy3 = split; dark = 0.35f; }
                            else                         sy3 = yy - drop;
                            const cv::Vec4b *sp = canvas.ptr<cv::Vec4b>(sy3);
                            cv::Vec3b *dp = img.ptr<cv::Vec3b>(y0+yy) + x0;
                            for (int xx = 0; xx < S; xx++) {
                                float al = sp[xx][3] / 255.0f;
                                if (al < 0.02f) continue;
                                float b3 = sp[xx][0]*kb*dark;
                                float g3 = sp[xx][1]*kg*dark;
                                float r3 = sp[xx][2]*kr*dark;
                                if (b3>255)b3=255;
                                if (g3>255)g3=255;
                                if (r3>255)r3=255;
                                dp[xx][0]=(uchar)(dp[xx][0]+(b3-dp[xx][0])*al);
                                dp[xx][1]=(uchar)(dp[xx][1]+(g3-dp[xx][1])*al);
                                dp[xx][2]=(uchar)(dp[xx][2]+(r3-dp[xx][2])*al);
                            }
                        }
                    }
                } else {
                    warn_once("faceimg_miss",
                        "facefx 24: no face image set — "
                        "vcam-ctl filter faceimage /path/animal.png "
                        "(PNG with transparency works best)");
                }
            } else if (mode == 23) {
                /* GOLDEN RATIO: the da-Vinci face-analysis meme —
                 * fibonacci spiral, phi grid, rotating measurement
                 * ticks and a suspiciously precise symmetry score. */
                cv::Scalar gold(60, 200, 245);
                float PHI = 1.618f;
                /* phi grid over the face box */
                int g1x = x + (int)(fw / PHI), g2x = x + fw - (int)(fw / PHI);
                int g1y = y + (int)(fh / PHI), g2y = y + fh - (int)(fh / PHI);
                cv::rectangle(img, cv::Rect(x, y, fw, fh), gold, 1, cv::LINE_AA);
                cv::line(img, {g1x, y}, {g1x, y+fh}, gold, 1, cv::LINE_AA);
                cv::line(img, {g2x, y}, {g2x, y+fh}, gold, 1, cv::LINE_AA);
                cv::line(img, {x, g1y}, {x+fw, g1y}, gold, 1, cv::LINE_AA);
                cv::line(img, {x, g2y}, {x+fw, g2y}, gold, 1, cv::LINE_AA);
                /* fibonacci spiral: nested quarter-arcs shrinking by phi */
                float rr3 = fw * 0.62f * amount;
                float sxq = (float)(x + fw/2), syq = (float)(y + fh/2);
                float aa0 = T * 0.01f;
                for (int k2 = 0; k2 < 7 && rr3 > 3; k2++) {
                    cv::ellipse(img, cv::Point((int)sxq,(int)syq),
                                cv::Size((int)rr3,(int)rr3),
                                aa0*57.3f + k2*90.0f, 0, 90, gold, 1, cv::LINE_AA);
                    float na = aa0 + k2*1.5708f;
                    sxq += std::cos(na) * rr3 * (1.0f - 1.0f/PHI);
                    syq += std::sin(na) * rr3 * (1.0f - 1.0f/PHI);
                    rr3 /= PHI;
                }
                /* landmark ticks + readouts */
                for (int lm = 0; lm < 5; lm++) {
                    int lx3 = (int)(f[4+lm*2]*inv), ly3 = (int)(f[5+lm*2]*inv);
                    cv::drawMarker(img, {lx3, ly3}, gold, cv::MARKER_CROSS, 9, 1);
                }
                int sym = 90 + (int)(uhash((int)(x/9), (int)(y/9), 0) * 9);
                char lab[48];
                snprintf(lab, sizeof lab, "PHI 1.618  SYM %d%%", sym);
                int ty2 = std::max(16, y - 10);
                cv::putText(img, lab, {x, ty2}, cv::FONT_HERSHEY_SIMPLEX,
                            0.55, cv::Scalar(0,0,0), 3);
                cv::putText(img, lab, {x, ty2}, cv::FONT_HERSHEY_SIMPLEX,
                            0.55, gold, 1);
            } else if (mode == 20) {
                /* PICASSO: cubism by patch surgery — your eyes, nose
                 * and mouth swap places with soft circular blends.
                 * amount = how far features stray. */
                struct FPt { float sx2, sy2, dx2, dy2, r2; };
                float k2 = 0.6f * amount;
                FPt parts[4] = {
                    /* src (landmark)      dst offset (face-relative) */
                    { f[4], f[5],  0.30f, -0.34f, fw/5.5f },  /* R eye -> forehead */
                    { f[6], f[7],  0.05f,  0.28f, fw/5.5f },  /* L eye -> cheek */
                    { f[8], f[9], -0.30f, -0.10f, fw/5.0f },  /* nose -> left */
                    { (f[10]+f[12])/2, (f[11]+f[13])/2,
                                   0.26f,  0.05f, fw/4.5f },  /* mouth -> right */
                };
                cv::Mat srcimg = img.clone();
                for (int p2 = 0; p2 < 4; p2++) {
                    int r3 = std::max(6, (int)parts[p2].r2);
                    int sx3 = (int)(parts[p2].sx2 * inv);
                    int sy3 = (int)(parts[p2].sy2 * inv);
                    int dx3 = sx3 + (int)(parts[p2].dx2 * fw * k2);
                    int dy3 = sy3 + (int)(parts[p2].dy2 * fh * k2);
                    if (sx3-r3<0||sy3-r3<0||sx3+r3>=w||sy3+r3>=h) continue;
                    if (dx3-r3<0||dy3-r3<0||dx3+r3>=w||dy3+r3>=h) continue;
                    for (int yy = -r3; yy < r3; yy++) {
                        const cv::Vec3b *sr = srcimg.ptr<cv::Vec3b>(sy3+yy);
                        cv::Vec3b *dr = img.ptr<cv::Vec3b>(dy3+yy);
                        float ny = (float)yy / r3;
                        for (int xx = -r3; xx < r3; xx++) {
                            float nx = (float)xx / r3;
                            float d2 = nx*nx + ny*ny;
                            if (d2 >= 1.0f) continue;
                            float al = d2 < 0.55f ? 1.0f : (1.0f-d2)/0.45f;
                            dr[dx3+xx][0]=(uchar)(dr[dx3+xx][0]+(sr[sx3+xx][0]-dr[dx3+xx][0])*al);
                            dr[dx3+xx][1]=(uchar)(dr[dx3+xx][1]+(sr[sx3+xx][1]-dr[dx3+xx][1])*al);
                            dr[dx3+xx][2]=(uchar)(dr[dx3+xx][2]+(sr[sx3+xx][2]-dr[dx3+xx][2])*al);
                        }
                    }
                }
            } else if (mode == 21) {
                /* TIME RIPPLE: pixels near your face live in the
                 * present; the further away, the deeper in the past —
                 * waves of time delay radiate from your head. amount =
                 * how deep the past reaches. */
                static cv::Mat ring[14];
                static int ri = 0, rqw = 0, rqh = 0;
                int QW = w/4, QH = h/4;
                if (rqw != QW || rqh != QH) {
                    for (int i = 0; i < 14; i++) ring[i].release();
                    rqw = QW; rqh = QH; ri = 0;
                }
                cv::Mat q;
                cv::resize(img, q, cv::Size(QW, QH), 0, 0, cv::INTER_AREA);
                q.copyTo(ring[ri]);
                float fcx = (x + fw/2) / 4.0f, fcy = (y + fh/2) / 4.0f;
                float rmax = std::sqrt((float)(QW*QW + QH*QH)) * 0.55f;
                cv::Mat outq(QH, QW, CV_8UC3);
                for (int yy = 0; yy < QH; yy++) {
                    cv::Vec3b *op = outq.ptr<cv::Vec3b>(yy);
                    float dy = yy - fcy;
                    for (int xx = 0; xx < QW; xx++) {
                        float dx = xx - fcx;
                        float d = std::sqrt(dx*dx + dy*dy) / rmax;
                        int delay = (int)(d * 13 * std::min(1.0f, amount));
                        if (delay > 13) delay = 13;
                        int gi = (ri - delay + 14) % 14;
                        const cv::Mat &g2 = ring[gi].empty() ? ring[ri] : ring[gi];
                        op[xx] = g2.at<cv::Vec3b>(yy, xx);
                    }
                }
                ri = (ri + 1) % 14;
                cv::resize(outq, img, img.size(), 0, 0, cv::INTER_LINEAR);
            } else if (mode == 22) {
                /* FISHEYE FACE: a barrel-lens bulge rides your face —
                 * the closer to your nose, the bigger. amount = bulge. */
                float R = fw * 0.95f;
                float k2 = 0.42f * amount;
                float fcx = (float)(x + fw/2), fcy = (float)(y + fh/2);
                int x0 = std::max(0, (int)(fcx - R)), x1 = std::min(w-1, (int)(fcx + R));
                int y0 = std::max(0, (int)(fcy - R)), y1 = std::min(h-1, (int)(fcy + R));
                if (x1 > x0+8 && y1 > y0+8) {
                    cv::Mat srcimg = img(cv::Rect(x0, y0, x1-x0, y1-y0)).clone();
                    for (int Y = y0; Y < y1; Y++) {
                        cv::Vec3b *ip = img.ptr<cv::Vec3b>(Y);
                        float dy = (Y - fcy) / R;
                        for (int X = x0; X < x1; X++) {
                            float dx = (X - fcx) / R;
                            float r2 = dx*dx + dy*dy;
                            if (r2 >= 1.0f) continue;
                            float f2 = 1.0f - k2 * (1.0f - r2);
                            int sx2 = (int)(fcx + dx*R*f2) - x0;
                            int sy2 = (int)(fcy + dy*R*f2) - y0;
                            if (sx2 < 0 || sy2 < 0 ||
                                sx2 >= srcimg.cols || sy2 >= srcimg.rows) continue;
                            ip[X] = srcimg.at<cv::Vec3b>(sy2, sx2);
                        }
                    }
                }
            } else if (mode == 16) {
                /* SPARKLE EYES: anime glints — 4-point stars twinkling
                 * at your eyes and cheeks. amount = density/size. */
                int nsp = (int)(3 * amount) + 2;
                for (int k2 = 0; k2 < nsp; k2++) {
                    /* anchor on eyes + cheek points, jittered per-star */
                    float bx2 = (k2 % 2) ? f[4] : f[6];
                    float by2 = (k2 % 2) ? f[5] : f[7];
                    if (k2 >= 2) { by2 += fh * 0.22f / inv; }
                    int sx2 = (int)(bx2 * inv) + (int)((uhash(k2,3,0)-0.5f)*fw*0.3f);
                    int sy2 = (int)(by2 * inv) + (int)((uhash(k2,7,0)-0.5f)*fh*0.2f);
                    float tw = 0.5f + 0.5f * std::sin(T * 0.22f + k2 * 1.9f);
                    if (tw < 0.35f) continue;              /* twinkle off-phase */
                    int r5 = std::max(2, (int)(fw / 22 * amount * tw));
                    if (sx2 < r5 || sy2 < r5 || sx2 >= w-r5 || sy2 >= h-r5) continue;
                    cv::Scalar wht(255,255,255);
                    cv::line(img, {sx2-r5,sy2}, {sx2+r5,sy2}, wht, 2);
                    cv::line(img, {sx2,sy2-r5}, {sx2,sy2+r5}, wht, 2);
                    cv::line(img, {sx2-r5/2,sy2-r5/2}, {sx2+r5/2,sy2+r5/2}, wht, 1);
                    cv::line(img, {sx2-r5/2,sy2+r5/2}, {sx2+r5/2,sy2-r5/2}, wht, 1);
                    cv::circle(img, {sx2,sy2}, std::max(1,r5/3), wht, cv::FILLED);
                }
            } else if (mode == 17) {
                /* HALO: a golden ring floats and bobs above your head,
                 * with a soft additive glow. amount = size. */
                int hx2 = x + fw/2;
                int hy2 = y - fh/3 + (int)(std::sin(T * 0.08f) * fh * 0.06f);
                int ra = (int)(fw * 0.42f * amount);
                int rb = std::max(3, ra / 4);
                if (hy2 - rb > 0) {
                    for (int g2 = 3; g2 >= 1; g2--)      /* glow layers */
                        cv::ellipse(img, cv::Point(hx2, hy2), cv::Size(ra+g2*3, rb+g2*2),
                                    0, 0, 360, cv::Scalar(30*g2, 60*g2, 70*g2), 2);
                    cv::ellipse(img, cv::Point(hx2, hy2), cv::Size(ra, rb),
                                0, 0, 360, cv::Scalar(90, 210, 255),
                                std::max(3, fw/28));
                    cv::ellipse(img, cv::Point(hx2, hy2), cv::Size(ra, rb),
                                0, 200, 340, cv::Scalar(200, 250, 255),
                                std::max(1, fw/56));
                }
            } else if (mode == 18) {
                /* CLOWN: red nose (tracks the nose landmark), white eye
                 * patches, rosy cheeks. amount = nose size. */
                cv::Point nz((int)(f[8]*inv), (int)(f[9]*inv));
                int nr = std::max(5, (int)(fw / 6 * amount));
                for (int e2 = 0; e2 < 2; e2++) {          /* eye patches */
                    cv::Point ey((int)(f[4+e2*2]*inv), (int)(f[5+e2*2]*inv));
                    cv::circle(img, ey, (int)(nr*0.9f), cv::Scalar(240,240,245), cv::FILLED);
                    cv::circle(img, ey, std::max(2,(int)(nr*0.35f)), cv::Scalar(40,40,40), cv::FILLED);
                }
                cv::circle(img, {x + fw/6, y + fh*2/3}, nr,           /* cheeks */
                           cv::Scalar(120,120,250), cv::FILLED);
                cv::circle(img, {x + fw*5/6, y + fh*2/3}, nr,
                           cv::Scalar(120,120,250), cv::FILLED);
                cv::circle(img, nz, nr, cv::Scalar(40,40,230), cv::FILLED);
                cv::circle(img, {nz.x - nr/3, nz.y - nr/3},
                           std::max(2, nr/4), cv::Scalar(120,120,255), cv::FILLED);
            } else if (mode == 19) {
                /* AUTO-DIRECTOR: the camera dollies to keep your face
                 * centered at a constant size, smoothly — Center Stage
                 * for any app. amount = framing tightness (how big your
                 * face is held), speed = follow responsiveness. */
                static float zcx = -1, zcy = -1, zfw = -1;
                float k2 = 0.03f * speed;                  /* follow EMA */
                float tcx = x + fw/2.f, tcy = y + fh/2.f, tfw = (float)fw;
                if (zcx < 0) { zcx = tcx; zcy = tcy; zfw = tfw; }
                zcx += (tcx - zcx) * k2;
                zcy += (tcy - zcy) * k2;
                zfw += (tfw - zfw) * k2 * 0.6f;
                /* target: face occupies ~ amount * 22% of frame width */
                float zoom = (0.22f * amount * w) / std::max(24.0f, zfw);
                if (zoom < 1.02f) zoom = 1.02f;
                if (zoom > 2.6f)  zoom = 2.6f;
                int cw = (int)(w / zoom), ch = (int)(h / zoom);
                int cx0 = (int)(zcx - cw/2), cy0 = (int)(zcy - ch*0.42f);
                cx0 = std::max(0, std::min(w - cw, cx0));
                cy0 = std::max(0, std::min(h - ch, cy0));
                cv::Mat crop = img(cv::Rect(cx0, cy0, cw, ch)).clone();
                cv::resize(crop, img, img.size(), 0, 0, cv::INTER_LINEAR);
            } else if (mode == 14) {
                /* FACE SATELLITES: three mini-copies of your face orbit
                 * your head like moons. */
                int cx3 = x + fw/2, cy3 = y + fh/2;
                int rad = (int)(std::max(fw, fh) * 1.05 * amount);
                int mw = std::max(12, fw*2/5), mh2 = std::max(12, fh*2/5);
                if (x >= 0 && y >= 0 && x+fw < w && y+fh < h) {
                    cv::Mat facecp = img(cv::Rect(x, y, fw, fh)).clone();
                    cv::Mat mini;
                    cv::resize(facecp, mini, cv::Size(mw, mh2));
                    float base = T * 0.05f;
                    for (int k2 = 0; k2 < 3; k2++) {
                        float aa = base + k2 * 2.0944f;
                        int px = cx3 + (int)(std::cos(aa) * rad) - mw/2;
                        int py = cy3 + (int)(std::sin(aa) * rad * 0.6f) - mh2/2;
                        if (px < 0 || py < 0 || px+mw >= w || py+mh2 >= h) continue;
                        for (int yy = 0; yy < mh2; yy++) {
                            cv::Vec3b *dr = img.ptr<cv::Vec3b>(py+yy) + px;
                            const cv::Vec3b *sr = mini.ptr<cv::Vec3b>(yy);
                            float ny = (yy - mh2/2.f) / (mh2/2.f);
                            for (int xx = 0; xx < mw; xx++) {
                                float nx = (xx - mw/2.f) / (mw/2.f);
                                float d2 = nx*nx + ny*ny;
                                if (d2 >= 1.0f) continue;
                                float al = d2 < 0.6f ? 1.0f : (1.0f-d2)/0.4f;
                                dr[xx][0]=(uchar)(dr[xx][0]+(sr[xx][0]-dr[xx][0])*al);
                                dr[xx][1]=(uchar)(dr[xx][1]+(sr[xx][1]-dr[xx][1])*al);
                                dr[xx][2]=(uchar)(dr[xx][2]+(sr[xx][2]-dr[xx][2])*al);
                            }
                        }
                    }
                }
            } else if (mode == 15) {
                /* MELTING FACE: your face drips downward, Dali-style;
                 * each column sags by its own amount, breathing slowly. */
                float ph = (0.5f + 0.5f * std::sin(T * 0.04f)) * amount;
                if (ph > 1.4f) ph = 1.4f;
                int reach = fh + fh/2;
                int y_end = std::min(h - 1, y + reach);
                for (int c = 0; c < fw; c++) {
                    int X = x + c;
                    if (X < 0 || X >= w) continue;
                    float n = uhash(c / 5, 0, 0);
                    int sag = (int)(ph * n * fh * 0.55f);
                    if (sag < 2) continue;
                    for (int Y = y_end; Y > y + sag; Y--) {
                        int sy2 = y + (int)((float)(Y - y - sag) * (float)(y_end - y)
                                            / (float)(y_end - y - sag + 1));
                        sy2 = std::max(0, std::min(h - 1, sy2));
                        img.at<cv::Vec3b>(Y, X) = img.at<cv::Vec3b>(sy2, X);
                    }
                }
            } else if (mode == 9) {
                /* MIRROR FACE: left half of your face reflected right */
                for (int yy = y; yy < y+fh && yy < h; yy++) {
                    cv::Vec3b *rw2 = img.ptr<cv::Vec3b>(yy);
                    for (int xx = 0; xx < fw/2; xx++) {
                        int l2 = x+xx, r2 = x+fw-1-xx;
                        if (l2>=0 && r2<w) rw2[r2] = rw2[l2];
                    }
                }
            } else if (mode == 10) {
                /* CYCLOPS: one giant eye between your two real ones */
                int ex2=(int)((f[4]+f[6])*inv/2), ey2=(int)((f[5]+f[7])*inv/2);
                int R4=std::max(8,fw/4);
                cv::ellipse(img,cv::Point(ex2,ey2),cv::Size(R4,R4*2/3),0,0,360,cv::Scalar(245,245,245),cv::FILLED);
                cv::circle(img,cv::Point(ex2,ey2),R4/2,cv::Scalar(90,40,20),cv::FILLED);
                cv::circle(img,cv::Point(ex2,ey2),R4/4,cv::Scalar(10,10,10),cv::FILLED);
                cv::circle(img,cv::Point(ex2-R4/6,ey2-R4/6),std::max(2,R4/8),cv::Scalar(255,255,255),cv::FILLED);
                cv::ellipse(img,cv::Point(ex2,ey2),cv::Size(R4,R4*2/3),0,0,360,cv::Scalar(30,30,30),2);
            } else if (mode == 11) {
                /* HEART EYES */
                for (int e2=0;e2<2;e2++){
                    int hx2=(int)(f[4+e2*2]*inv), hy2=(int)(f[5+e2*2]*inv);
                    int r5=std::max(5,fw/8);
                    cv::Scalar red(60,40,235);
                    cv::circle(img,cv::Point(hx2-r5/2,hy2-r5/3),r5/2+1,red,cv::FILLED);
                    cv::circle(img,cv::Point(hx2+r5/2,hy2-r5/3),r5/2+1,red,cv::FILLED);
                    cv::Point tri[3]={{hx2-r5,hy2-r5/6},{hx2+r5,hy2-r5/6},{hx2,hy2+r5}};
                    cv::fillConvexPoly(img,tri,3,red);
                }
            } else if (mode == 12) {
                /* INTERROGATION: darkness except an oval on your face */
                float fcx=x+fw/2.f, fcy=y+fh/2.f, ra=fw*0.85f, rb=fh*0.95f;
                for (int yy=0;yy<h;yy++){
                    uchar *rw2=data+yy*stride;
                    float ny=(yy-fcy)/rb;
                    for (int xx=0;xx<w;xx++){
                        float nx=(xx-fcx)/ra, d2=nx*nx+ny*ny;
                        if (d2<1.0f) continue;
                        float k2 = d2>2.0f?0.12f:(1.0f-(d2-1.0f)*0.88f);
                        uchar *p2=rw2+xx*3;
                        p2[0]=(uchar)(p2[0]*k2); p2[1]=(uchar)(p2[1]*k2); p2[2]=(uchar)(p2[2]*k2);
                    }
                }
            } else if (mode == 13) {
                /* TARGET LOCK: rotating sci-fi HUD reticle on your face */
                cv::Point c3(x+fw/2,y+fh/2);
                int R5=(int)(std::max(fw,fh)*0.75);
                cv::Scalar g3(80,255,80);
                float a3=std::fmod(T, 90.0f)/90.0f*6.2832f;
                for(int k2=0;k2<4;k2++){
                    float aa=a3+k2*1.5708f;
                    cv::ellipse(img,c3,cv::Size(R5,R5),aa*57.3,0,60,g3,2);
                }
                cv::circle(img,c3,R5*2/3,g3,1);
                cv::line(img,cv::Point(c3.x-R5,c3.y),cv::Point(c3.x-R5/2,c3.y),g3,2);
                cv::line(img,cv::Point(c3.x+R5/2,c3.y),cv::Point(c3.x+R5,c3.y),g3,2);
                cv::line(img,cv::Point(c3.x,c3.y-R5),cv::Point(c3.x,c3.y-R5/2),g3,2);
                cv::line(img,cv::Point(c3.x,c3.y+R5/2),cv::Point(c3.x,c3.y+R5),g3,2);
                cv::putText(img,"LOCK",cv::Point(c3.x+R5/2,c3.y-R5/2),cv::FONT_HERSHEY_SIMPLEX,0.7,g3,2);
            } else {
                /* googly eyes at the two eye landmarks (idx 4,5 & 6,7) */
                for (int e = 0; e < 2; e++) {
                    int lx = (int)(f[4 + e*2] * inv);
                    int ly = (int)(f[5 + e*2] * inv);
                    int rad = std::max(6, fw / 6);
                    cv::circle(img, cv::Point(lx, ly), rad,
                               cv::Scalar(255, 255, 255), -1);
                    cv::circle(img, cv::Point(lx, ly), rad,
                               cv::Scalar(0, 0, 0), std::max(1, rad/6));
                    /* pupil offset a little for a wobble */
                    int px = lx + rad/3, py = ly + rad/4;
                    cv::circle(img, cv::Point(px, py),
                               std::max(2, rad/2), cv::Scalar(0, 0, 0), -1);
                }
            }
        }
    } catch (const std::exception &e) {
        warn_once("face", std::string("face fx failed: ") + e.what());
    }
}

/* ── shared interactive-physics helpers (used by the game modes) ──
 * The matte is the collider; the matte's CHANGE is the force. */
struct BodyProbe {
    bool  hit;        /* body touching the probe circle? */
    float nx, ny;     /* surface normal, pointing away from your body   */
    float pux, puy;   /* your push: direction you're sweeping, x speed  */
};
static BodyProbe probe_body(const cv::Mat &mask, const cv::Mat &prev,
                            float px, float py, float R)
{
    BodyProbe b; b.hit = false; b.nx = 0; b.ny = -1; b.pux = 0; b.puy = 0;
    if (mask.empty()) return b;
    /* contact + normal: ring samples, weighted away from the body mass */
    const int NS = 16;
    float sumx = 0, sumy = 0, best = 0;
    for (int i = 0; i < NS; i++) {
        float a = 6.2832f * i / NS;
        int sx = (int)(px + std::cos(a) * R);
        int sy = (int)(py + std::sin(a) * R);
        if (sx < 0 || sy < 0 || sx >= mask.cols || sy >= mask.rows) continue;
        float m = mask.at<float>(sy, sx);
        if (m > 0.5f) {
            sumx += std::cos(a) * m;
            sumy += std::sin(a) * m;
            best = std::max(best, m);
        }
    }
    if (best <= 0.5f) return b;
    b.hit = true;
    float len = std::sqrt(sumx*sumx + sumy*sumy);
    if (len > 1e-3f) { b.nx = -sumx/len; b.ny = -sumy/len; }
    /* push: where matte APPEARED minus where it VANISHED (immune to the
     * aperture problem that defeats a local gradient solve) */
    if (prev.empty() || prev.size() != mask.size()) return b;
    float pcx=0, pcy=0, pw=0, ncx=0, ncy=0, nw=0;
    int W = (int)(R * 3.2f);
    for (int yy = -W; yy <= W; yy += 2)
        for (int xx = -W; xx <= W; xx += 2) {
            int sx = (int)px + xx, sy = (int)py + yy;
            if (sx < 0 || sy < 0 || sx >= mask.cols || sy >= mask.rows) continue;
            float d = mask.at<float>(sy, sx) - prev.at<float>(sy, sx);
            if (d > 0.20f)       { pcx += sx*d; pcy += sy*d; pw += d; }
            else if (d < -0.20f) { ncx -= sx*d; ncy -= sy*d; nw -= d; }
        }
    if (pw > 1.0f && nw > 1.0f) {
        float dx = pcx/pw - ncx/nw, dy = pcy/pw - ncy/nw;
        float dl = std::sqrt(dx*dx + dy*dy);
        if (dl > 1e-3f) {
            float punch = std::min(11.0f,
                (pw + nw) / std::max(1.0f, R*R*0.10f) * 2.4f);
            b.pux = dx/dl * punch;
            b.puy = dy/dl * punch;
        }
    }
    return b;
}

void cv_fx_bgblur(unsigned char *data, int w, int h, int stride, int mode,
                  float edge, float smooth, float size, float fx)
{
    if (!data || w < 16 || h < 16) return;
    if (edge <= 0.0f) edge = 0.22f;          /* mask generosity: lower = more hair */
    if (edge < 0.05f) edge = 0.05f;
    if (edge > 0.50f) edge = 0.50f;
    if (smooth <= 0.0f) smooth = 0.80f;      /* responsiveness: 1 = instant */
    if (smooth < 0.40f) smooth = 0.40f;
    if (smooth > 1.00f) smooth = 1.00f;
    if (fx <= 0.0f) fx = 1.0f;               /* per-mode effect strength */
    if (fx < 0.2f) fx = 0.2f;
    if (fx > 3.0f) fx = 3.0f;
    int size_w = (int)size;                  /* RVM refinement width */
    if (size_w <= 0) size_w = 512;           /* 0 = default */
    if (size_w < 256) size_w = 256;
    if (size_w > 640) size_w = 640;
    try {
        std::lock_guard<std::mutex> lock(g_seg.mtx);
        cv::Mat img = wrap_bgr(data, w, h, stride);

        /* PP-HumanSeg is only REQUIRED when RVM can't serve as the
         * matting engine. With RVM available (ORT backend, or a DNN
         * build that computes it), personfx works without the
         * pphumanseg model file entirely. */
        bool rvm_possible =
            (g_seg.rvm_state >= 0 || (ort_available() && g_seg.ort_state >= 0));
        if (!rvm_possible && !ensure_seg()) return;   /* nothing can matte */

        /* Inference EVERY frame: PP-HumanSeg is ~6 ms on a laptop CPU,
         * and a stale mask visibly lags moving hands (the old every-3rd
         * throttle caused exactly that). Temporal EMA keeps edges calm. */
        tick(g_seg.counter);
        {
            /* body ROI from the face detector — only needed for the
             * PP-HumanSeg path (RVM mattes the full frame natively).
             * Lock order seg->face is the established unique nesting. */
            cv::Rect body(0, 0, img.cols, img.rows);
            bool rvm_alive =
                (ort_available() && g_seg.ort_state >= 0) || g_seg.rvm_state > 0 ||
                (g_seg.rvm_state == 0 &&
                 path_exists(model_dir() + "/rvm_mobilenetv3_fp32.onnx"));
            if (!rvm_alive) {
                std::lock_guard<std::mutex> fl(g_face.mtx);
                const int TW = 320;
                double fs = (img.cols > TW) ? (double)TW / img.cols : 1.0;
                int dw = std::max(16, (int)(img.cols * fs));
                int dh = std::max(16, (int)(img.rows * fs));
                if (ensure_face(dw, dh)) {
                    if (tick(g_face.counter) % INFER_EVERY == 0) {
                        cv::Mat sm2; cv::resize(img, sm2, cv::Size(dw, dh));
                        cv::Mat fc2; g_face.det->detect(sm2, fc2);
                        fc2.copyTo(g_face.faces);
                    }
                    if (!g_face.faces.empty()) {
                        const float *ff = g_face.faces.ptr<float>(0);
                        double iv = 1.0 / fs;
                        int fx2=(int)(ff[0]*iv), fy2=(int)(ff[1]*iv);
                        int fw2=(int)(ff[2]*iv), fh2=(int)(ff[3]*iv);
                        int bx = fx2 + fw2/2 - fw2*5/2;      /* 5 face-widths */
                        int bw2 = fw2*5;
                        int by3 = fy2 - fh2;                  /* headroom */
                        bx = std::max(0, bx);
                        by3 = std::max(0, by3);
                        bw2 = std::min(img.cols - bx, bw2);
                        int bh3 = img.rows - by3;             /* to the floor */
                        if (bw2 > 32 && bh3 > 32)
                            body = cv::Rect(bx, by3, bw2, bh3);
                    }
                }
            }
            /* ── adaptive pacing: measure inference cost and skip the
             * right number of frames to stay realtime. The recurrent
             * matte + EMA hide the skips; the picture stays smooth
             * because only the MASK updates less often, never the
             * video. 0 skips under ~24 ms, up to 3 when very slow. */
            static int skip_left = 0;
            static double ema_ms = 0.0;
            if (skip_left > 0 && !g_seg.mask.empty() &&
                g_seg.mask.size() == img.size()) {
                skip_left--;
                goto mask_ready;   /* reuse previous mask this frame */
            }
            {
            auto t0 = std::chrono::steady_clock::now();
            cv::Mat m = rvm_infer(img, size_w);
            if (!m.empty()) {
                /* RVM alpha is already clean; light remap honors 'edge' */
                m = (m - edge*0.5) * (1.0/(1.0 - edge*0.5));
                cv::min(m, 1.0, m); cv::max(m, 0.0, m);
            } else if (ensure_seg()) {
                m = seg_infer(img, body, edge);
            } else {
                return;   /* RVM just died and pphumanseg is absent */
            }
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            ema_ms = (ema_ms <= 0.0) ? ms : 0.7*ema_ms + 0.3*ms;
            skip_left = (ema_ms <= 24.0) ? 0
                      : (ema_ms <= 48.0) ? 1
                      : (ema_ms <= 80.0) ? 2 : 3;
            if (!m.empty()) {
                if (!g_seg.mask.empty() && g_seg.mask.size() == m.size())
                    cv::addWeighted(m, smooth, g_seg.mask, 1.0 - smooth, 0.0, g_seg.mask);
                else
                    m.copyTo(g_seg.mask);
            }
            }              /* end of timed inference block */
mask_ready: ;
        }
        if (g_seg.mask.empty() || g_seg.mask.size() != img.size()) return;

        /* Running background estimate: pixels the mask calls background
         * feed a slow EMA — after a second or two vcam knows what the
         * room looks like WITHOUT you. Powers mode 4 (invisibility). */
        if (g_seg.bg_est.empty() || g_seg.bg_est.size() != img.size()) {
            img.copyTo(g_seg.bg_est);
            g_seg.bg_frames = 0;
        } else {
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                cv::Vec3b *bp = g_seg.bg_est.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    if (mm[x] < 0.25f) {
                        bp[x][0] = (uchar)((bp[x][0]*15 + ip[x][0]) / 16);
                        bp[x][1] = (uchar)((bp[x][1]*15 + ip[x][1]) / 16);
                        bp[x][2] = (uchar)((bp[x][2]*15 + ip[x][2]) / 16);
                    }
                }
            }
            if (g_seg.bg_frames < 1000) g_seg.bg_frames++;
        }

        cv::Mat person3;
        cv::cvtColor(g_seg.mask, person3, cv::COLOR_GRAY2BGR);
        cv::Mat inv;
        cv::bitwise_not(person3, inv);

        if (mode == 11) {
            /* RAINBOW AURA: hue-cycling glow rings around your body */
            cv::Mat m8,d1,d2m;
            g_seg.mask.convertTo(m8,CV_8U,255.0);
            unsigned tk=g_seg.counter;
            for(int ring=0;ring<3;ring++){
                cv::dilate(m8,d1,cv::Mat(),cv::Point(-1,-1),2+ring*3);
                cv::dilate(m8,d2m,cv::Mat(),cv::Point(-1,-1),ring*3);
                float hue=(tk*0.06f)+ring*2.1f;
                int cb2=(int)((std::sin(hue+4.19f)*0.5f+0.5f)*255);
                int cg2=(int)((std::sin(hue+2.09f)*0.5f+0.5f)*255);
                int cr2=(int)((std::sin(hue)*0.5f+0.5f)*255);
                for(int y=0;y<img.rows;y++){
                    const uchar *da=d1.ptr<uchar>(y),*db=d2m.ptr<uchar>(y);
                    cv::Vec3b *ip=img.ptr<cv::Vec3b>(y);
                    for(int x=0;x<img.cols;x++){
                        if(da[x]>128 && db[x]<128){
                            ip[x][0]=clamp_int_u8(ip[x][0]+cb2/2);
                            ip[x][1]=clamp_int_u8(ip[x][1]+cg2/2);
                            ip[x][2]=clamp_int_u8(ip[x][2]+cr2/2);
                        }
                    }
                }
            }
            return;
        }
        if (mode == 12) {
            /* PAINTED WORLD: neural style transfers the BACKGROUND
             * while you stay photographic — two nets chained through
             * the matte. Needs --models-style. */
            cv::Mat styled = img.clone();
            cv_fx_style(styled.data, img.cols, img.rows,
                        (int)styled.step, 1 /*candy*/, 1.0f, 160.0f);
            for(int y=0;y<img.rows;y++){
                const float *mm=g_seg.mask.ptr<float>(y);
                const cv::Vec3b *sp2=styled.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip=img.ptr<cv::Vec3b>(y);
                for(int x=0;x<img.cols;x++){
                    float a=1.0f-mm[x]; if(a<0.03f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(sp2[x][0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(sp2[x][1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(sp2[x][2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 13) {
            /* BODY PRIVACY: pixelate only the person */
            cv::Mat tiny;
            cv::resize(img,tiny,cv::Size(img.cols/24,img.rows/24));
            cv::resize(tiny,tiny,img.size(),0,0,cv::INTER_NEAREST);
            for(int y=0;y<img.rows;y++){
                const float *mm=g_seg.mask.ptr<float>(y);
                const cv::Vec3b *tp=tiny.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip=img.ptr<cv::Vec3b>(y);
                for(int x=0;x<img.cols;x++){
                    float a=mm[x]; if(a<0.05f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(tp[x][0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(tp[x][1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(tp[x][2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 119) {
            /* BOUNCY BALLS: 64 rubber balls with real ball-to-ball
             * collisions. They bounce off you AND off each other, pile
             * up on the floor, and you can bat them around. Elastic
             * impulses, no cheating. fx = bounciness. */
            const int NB2 = 64;
            static float bx4[NB2], by4[NB2], bvx2[NB2], bvy2[NB2], br2[NB2];
            static int bc2[NB2], binit3 = 0;
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!binit3) {
                for (int i2 = 0; i2 < NB2; i2++) {
                    bx4[i2] = 20 + uhash(i2,1,31)*(img.cols-40);
                    by4[i2] = -uhash(i2,2,31)*img.rows*1.6f;
                    bvx2[i2] = (uhash(i2,3,31)-0.5f)*3;
                    bvy2[i2] = 0;
                    br2[i2] = 9 + uhash(i2,4,31)*13;
                    bc2[i2] = i2 % 5;
                }
                binit3 = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            float rest = 0.62f + 0.22f*std::min(2.0f, fx);   /* restitution */
            /* integrate + body collision */
            for (int i2 = 0; i2 < NB2; i2++) {
                bvy2[i2] += 0.34f;
                bx4[i2] += bvx2[i2];
                by4[i2] += bvy2[i2];
                int xi = std::max(0,std::min(img.cols-1,(int)bx4[i2]));
                int yi = std::max(0,std::min(img.rows-1,(int)by4[i2]));
                if (g_seg.mask.at<float>(yi,xi) > 0.5f) {   /* hit YOU */
                    int x1=std::min(img.cols-1,xi+4), x0=std::max(0,xi-4);
                    int y1=std::min(img.rows-1,yi+4), y0=std::max(0,yi-4);
                    float gx2 = g_seg.mask.at<float>(yi,x1)-g_seg.mask.at<float>(yi,x0);
                    float gy2 = g_seg.mask.at<float>(y1,xi)-g_seg.mask.at<float>(y0,xi);
                    float l2 = std::sqrt(gx2*gx2+gy2*gy2);
                    float nx2 = l2>1e-3f ? -gx2/l2 : 0;
                    float ny2 = l2>1e-3f ? -gy2/l2 : -1;
                    float d = g_seg.mask.at<float>(yi,xi) - prev.at<float>(yi,xi);
                    float dot = bvx2[i2]*nx2 + bvy2[i2]*ny2;
                    if (dot < 0) {
                        bvx2[i2] -= (1+rest)*dot*nx2;
                        bvy2[i2] -= (1+rest)*dot*ny2;
                    }
                    /* your motion adds kick */
                    bvx2[i2] += nx2 * (1.6f + std::fabs(d)*13.0f);
                    bvy2[i2] += ny2 * (1.6f + std::fabs(d)*13.0f);
                    bx4[i2] += nx2*2.5f; by4[i2] += ny2*2.5f;
                }
                /* walls + floor */
                if (bx4[i2] < br2[i2]) { bx4[i2]=br2[i2]; bvx2[i2]=std::fabs(bvx2[i2])*rest; }
                if (bx4[i2] > img.cols-br2[i2]) { bx4[i2]=img.cols-br2[i2]; bvx2[i2]=-std::fabs(bvx2[i2])*rest; }
                if (by4[i2] > img.rows-br2[i2]) {
                    by4[i2]=img.rows-br2[i2];
                    bvy2[i2]=-std::fabs(bvy2[i2])*rest;
                    bvx2[i2]*=0.99f;
                }
                if (by4[i2] < -img.rows*2) by4[i2] = -20;
            }
            /* ball-to-ball elastic collisions (N^2 at N=64: cheap) */
            for (int i2 = 0; i2 < NB2; i2++)
                for (int j2 = i2+1; j2 < NB2; j2++) {
                    float dx3 = bx4[j2]-bx4[i2], dy3 = by4[j2]-by4[i2];
                    float d2 = dx3*dx3+dy3*dy3;
                    float rr = br2[i2]+br2[j2];
                    if (d2 >= rr*rr || d2 < 1e-4f) continue;
                    float d = std::sqrt(d2);
                    float nx2 = dx3/d, ny2 = dy3/d;
                    float overlap = (rr - d) * 0.5f;
                    bx4[i2] -= nx2*overlap; by4[i2] -= ny2*overlap;
                    bx4[j2] += nx2*overlap; by4[j2] += ny2*overlap;
                    float rvx = bvx2[j2]-bvx2[i2], rvy = bvy2[j2]-bvy2[i2];
                    float vn = rvx*nx2 + rvy*ny2;
                    if (vn > 0) continue;
                    float imp = -(1+rest) * vn * 0.5f;
                    bvx2[i2] -= imp*nx2; bvy2[i2] -= imp*ny2;
                    bvx2[j2] += imp*nx2; bvy2[j2] += imp*ny2;
                }
            static const int BCOL[5][3] = {{80,90,240},{70,200,255},{110,220,120},
                                           {230,170,70},{210,110,230}};
            for (int i2 = 0; i2 < NB2; i2++) {
                const int *c2 = BCOL[bc2[i2]];
                cv::circle(img, {(int)bx4[i2],(int)by4[i2]}, (int)br2[i2]+2,
                           cv::Scalar(20,20,25), cv::FILLED, cv::LINE_AA);
                cv::circle(img, {(int)bx4[i2],(int)by4[i2]}, (int)br2[i2],
                           cv::Scalar(c2[0],c2[1],c2[2]), cv::FILLED, cv::LINE_AA);
                cv::circle(img, {(int)(bx4[i2]-br2[i2]*0.32f),
                                 (int)(by4[i2]-br2[i2]*0.32f)},
                           std::max(2,(int)(br2[i2]*0.3f)),
                           cv::Scalar(255,255,255), cv::FILLED, cv::LINE_AA);
            }
            (void)tk;
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 120) {
            /* LIQUID POOL: real liquid particles — they fall, POOL at the
             * bottom, level out, slosh when you disturb them, and splash
             * where you strike. Grid-density repulsion (position-based
             * fluid, lite). Scoop it with your hands. fx = liquid amount. */
            const int NL = 900;
            static float lx[NL], ly[NL], lvx[NL], lvy[NL];
            static int linit = 0;
            static cv::Mat prev;
            const int CG = 12;                       /* density cell size */
            static int cnt[64*48];
            unsigned tk = g_seg.counter;
            if (!linit) {
                for (int i2 = 0; i2 < NL; i2++) {
                    lx[i2] = 30 + uhash(i2,1,41)*(img.cols-60);
                    ly[i2] = -uhash(i2,2,41)*img.rows;
                    lvx[i2] = 0; lvy[i2] = 0;
                }
                linit = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            int GW2 = std::min(64, img.cols/CG), GH2 = std::min(48, img.rows/CG);
            std::memset(cnt, 0, sizeof cnt);
            int active = (int)(NL * std::min(1.0f, 0.3f + fx*0.35f));
            for (int i2 = 0; i2 < active; i2++) {
                int cx2 = std::min(GW2-1, std::max(0, (int)(lx[i2]/CG)));
                int cy2 = std::min(GH2-1, std::max(0, (int)(ly[i2]/CG)));
                cnt[cy2*GW2+cx2]++;
            }
            for (int i2 = 0; i2 < active; i2++) {
                lvy[i2] += 0.30f;                             /* gravity */
                /* pressure: flee crowded cells */
                int cx2 = std::min(GW2-2, std::max(1, (int)(lx[i2]/CG)));
                int cy2 = std::min(GH2-2, std::max(1, (int)(ly[i2]/CG)));
                float pl = (float)cnt[cy2*GW2+cx2-1], pr = (float)cnt[cy2*GW2+cx2+1];
                float pu = (float)cnt[(cy2-1)*GW2+cx2], pd = (float)cnt[(cy2+1)*GW2+cx2];
                lvx[i2] += (pl - pr) * 0.055f;
                lvy[i2] += (pu - pd) * 0.055f - 0.02f*cnt[cy2*GW2+cx2];
                /* body: push out + inherit your motion (splash) */
                int xi = std::max(0,std::min(img.cols-1,(int)lx[i2]));
                int yi = std::max(0,std::min(img.rows-1,(int)ly[i2]));
                float m = g_seg.mask.at<float>(yi,xi);
                float d = m - prev.at<float>(yi,xi);
                if (m > 0.4f || std::fabs(d) > 0.2f) {
                    int x1=std::min(img.cols-1,xi+4), x0=std::max(0,xi-4);
                    int y1=std::min(img.rows-1,yi+4), y0=std::max(0,yi-4);
                    float gx2 = g_seg.mask.at<float>(yi,x1)-g_seg.mask.at<float>(yi,x0);
                    float gy2 = g_seg.mask.at<float>(y1,xi)-g_seg.mask.at<float>(y0,xi);
                    float l2 = std::sqrt(gx2*gx2+gy2*gy2)+1e-3f;
                    float force = (m > 0.4f ? 2.2f : 0.0f)
                                + std::fabs(d) * 15.0f * fx;
                    lvx[i2] -= gx2/l2 * force;
                    lvy[i2] -= gy2/l2 * force;
                }
                lvx[i2] *= 0.94f; lvy[i2] *= 0.96f;           /* viscosity */
                lx[i2] += lvx[i2]; ly[i2] += lvy[i2];
                if (lx[i2] < 3) { lx[i2]=3; lvx[i2]=std::fabs(lvx[i2])*0.4f; }
                if (lx[i2] > img.cols-4) { lx[i2]=img.cols-4; lvx[i2]=-std::fabs(lvx[i2])*0.4f; }
                if (ly[i2] > img.rows-4) { ly[i2]=img.rows-4; lvy[i2]=-std::fabs(lvy[i2])*0.25f; }
                if (ly[i2] < -img.rows) ly[i2] = -10;
            }
            /* render: additive blue blobs -> reads as a fluid body */
            for (int i2 = 0; i2 < active; i2++) {
                int x = (int)lx[i2], y = (int)ly[i2];
                if (x < 2 || y < 2 || x >= img.cols-2 || y >= img.rows-2) continue;
                float sp = std::sqrt(lvx[i2]*lvx[i2]+lvy[i2]*lvy[i2]);
                int hi = std::min(255, 110 + (int)(sp*26));
                for (int dy = -3; dy <= 3; dy++)
                    for (int dx = -3; dx <= 3; dx++) {
                        int r2 = dx*dx+dy*dy;
                        if (r2 > 9) continue;
                        cv::Vec3b &p2 = img.at<cv::Vec3b>(y+dy, x+dx);
                        float k = (1.0f - r2/9.0f) * 0.85f;
                        p2[0]=clamp_int_u8((int)(p2[0]+k*230));
                        p2[1]=clamp_int_u8((int)(p2[1]+k*150));
                        p2[2]=clamp_int_u8((int)(p2[2]+k*(60+hi/4)));
                    }
            }
            (void)tk;
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 121) {
            /* SLIME: sticky goo that CLINGS to you. It falls, sticks
             * where it lands on your body, stretches and sags, drips off
             * in strands — and flings away when you move fast enough.
             * fx = stickiness. */
            const int NS2 = 620;
            static float sx4[NS2], sy4[NS2], svx[NS2], svy[NS2];
            static float ax[NS2], ay[NS2];        /* anchor on your body */
            static int stuck[NS2], sinit2 = 0;
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!sinit2) {
                for (int i2 = 0; i2 < NS2; i2++) {
                    sx4[i2] = uhash(i2,1,43)*img.cols;
                    sy4[i2] = -uhash(i2,2,43)*img.rows*1.4f;
                    svx[i2] = 0; svy[i2] = 1; stuck[i2] = 0;
                }
                sinit2 = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            for (int i2 = 0; i2 < NS2; i2++) {
                int xi = std::max(0,std::min(img.cols-1,(int)sx4[i2]));
                int yi = std::max(0,std::min(img.rows-1,(int)sy4[i2]));
                float m = g_seg.mask.at<float>(yi,xi);
                float d = m - prev.at<float>(yi,xi);
                if (stuck[i2]) {
                    /* cling to the anchor, sag downward, fling if you
                     * move fast */
                    float pull = 0.30f * fx;
                    svx[i2] += (ax[i2]-sx4[i2]) * pull;
                    svy[i2] += (ay[i2]-sy4[i2]) * pull + 0.16f;   /* sag */
                    svx[i2] *= 0.7f; svy[i2] *= 0.7f;
                    sx4[i2] += svx[i2]; sy4[i2] += svy[i2];
                    if (std::fabs(d) > 0.3f/std::max(0.5f,fx)) {  /* flung off */
                        stuck[i2] = 0;
                        int x1=std::min(img.cols-1,xi+4), x0=std::max(0,xi-4);
                        float gx2 = g_seg.mask.at<float>(yi,x1)
                                  - g_seg.mask.at<float>(yi,x0);
                        svx[i2] = -gx2*9.0f + (uhash(i2,tk%37,3)-0.5f)*5;
                        svy[i2] = -2.5f;
                    }
                    /* strand back to the anchor */
                    cv::line(img, {(int)sx4[i2],(int)sy4[i2]},
                             {(int)ax[i2],(int)ay[i2]},
                             cv::Scalar(90,220,140), 2, cv::LINE_AA);
                } else {
                    svy[i2] += 0.30f;
                    svx[i2] *= 0.99f;
                    sx4[i2] += svx[i2]; sy4[i2] += svy[i2];
                    xi = std::max(0,std::min(img.cols-1,(int)sx4[i2]));
                    yi = std::max(0,std::min(img.rows-1,(int)sy4[i2]));
                    if (g_seg.mask.at<float>(yi,xi) > 0.5f) {     /* SPLAT */
                        stuck[i2] = 1;
                        ax[i2] = sx4[i2]; ay[i2] = sy4[i2];
                        svx[i2] = svy[i2] = 0;
                    }
                    if (sy4[i2] > img.rows + 8) {                 /* recycle */
                        sx4[i2] = uhash(i2,tk%83,9)*img.cols;
                        sy4[i2] = -8 - uhash(i2,tk%79,11)*90;
                        svx[i2] = 0; svy[i2] = 1;
                    }
                }
                int rr = stuck[i2] ? 5 : 4;
                cv::circle(img, {(int)sx4[i2],(int)sy4[i2]}, rr+2,
                           cv::Scalar(40,120,60), cv::FILLED, cv::LINE_AA);
                cv::circle(img, {(int)sx4[i2],(int)sy4[i2]}, rr,
                           cv::Scalar(110,240,150), cv::FILLED, cv::LINE_AA);
                cv::circle(img, {(int)sx4[i2]-1,(int)sy4[i2]-1}, 1,
                           cv::Scalar(220,255,220), cv::FILLED);
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 122) {
            /* LEAF STORM: autumn leaves with air drag — they get caught
             * in the WAKE of your movement, swirl, tumble with spin, and
             * settle on the floor until you kick them up again.
             * fx = wind strength. */
            const int NL2 = 240;
            static float lx2[NL2], ly2[NL2], lvx2[NL2], lvy2[NL2];
            static float rot[NL2], rv[NL2];
            static int lset[NL2], linit2 = 0;
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!linit2) {
                for (int i2 = 0; i2 < NL2; i2++) {
                    lx2[i2] = uhash(i2,1,53)*img.cols;
                    ly2[i2] = -uhash(i2,2,53)*img.rows*1.5f;
                    lvx2[i2] = 0; lvy2[i2] = 0.7f + uhash(i2,3,53);
                    rot[i2] = uhash(i2,4,53)*6.28f;
                    rv[i2] = (uhash(i2,5,53)-0.5f)*0.25f;
                    lset[i2] = 0;
                }
                linit2 = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            static const int LCOL[4][3] = {{40,110,205},{30,150,225},
                                           {50,80,170},{60,170,235}};
            for (int i2 = 0; i2 < NL2; i2++) {
                int xi = std::max(0,std::min(img.cols-1,(int)lx2[i2]));
                int yi = std::max(0,std::min(img.rows-1,(int)ly2[i2]));
                float m = g_seg.mask.at<float>(yi,xi);
                float d = m - prev.at<float>(yi,xi);
                /* your wake: the mask's change drives a wind gust */
                if (std::fabs(d) > 0.12f) {
                    int x1=std::min(img.cols-1,xi+5), x0=std::max(0,xi-5);
                    int y1=std::min(img.rows-1,yi+5), y0=std::max(0,yi-5);
                    float gx2 = g_seg.mask.at<float>(yi,x1)-g_seg.mask.at<float>(yi,x0);
                    float gy2 = g_seg.mask.at<float>(y1,xi)-g_seg.mask.at<float>(y0,xi);
                    float l2 = std::sqrt(gx2*gx2+gy2*gy2)+1e-3f;
                    float gust = std::fabs(d) * 14.0f * fx;
                    lvx2[i2] -= gx2/l2 * gust;
                    lvy2[i2] -= gy2/l2 * gust - 0.5f*gust;    /* lift */
                    rv[i2] += (uhash(i2,tk%41,7)-0.5f)*0.5f;
                    lset[i2] = 0;
                }
                if (m > 0.55f) {                    /* can't pass through you */
                    int x1=std::min(img.cols-1,xi+4), x0=std::max(0,xi-4);
                    float gx2 = g_seg.mask.at<float>(yi,x1)-g_seg.mask.at<float>(yi,x0);
                    lvx2[i2] -= gx2 * 4.0f;
                    lvy2[i2] -= 0.6f;
                }
                if (!lset[i2]) {
                    lvy2[i2] += 0.11f;                         /* gravity */
                    /* leaf flutter: sideways oscillation + drag */
                    lvx2[i2] += std::sin(ly2[i2]*0.05f + i2) * 0.16f;
                    lvx2[i2] *= 0.95f; lvy2[i2] *= 0.96f;
                    if (lvy2[i2] > 3.2f) lvy2[i2] = 3.2f;
                    lx2[i2] += lvx2[i2]; ly2[i2] += lvy2[i2];
                    rot[i2] += rv[i2]; rv[i2] *= 0.97f;
                    if (ly2[i2] > img.rows - 7) {              /* settle */
                        ly2[i2] = img.rows - 7;
                        lvx2[i2] = lvy2[i2] = 0;
                        lset[i2] = 1;
                    }
                    if (lx2[i2] < -20) lx2[i2] = img.cols + 10;
                    if (lx2[i2] > img.cols + 20) lx2[i2] = -10;
                }
                /* draw a little leaf: rotated ellipse + stem */
                const int *c2 = LCOL[i2 & 3];
                cv::ellipse(img, {(int)lx2[i2],(int)ly2[i2]}, {7,4},
                            rot[i2]*57.3f, 0, 360,
                            cv::Scalar(c2[0],c2[1],c2[2]), cv::FILLED, cv::LINE_AA);
                cv::ellipse(img, {(int)lx2[i2],(int)ly2[i2]}, {7,4},
                            rot[i2]*57.3f, 0, 360,
                            cv::Scalar(c2[0]/2,c2[1]/2,c2[2]/2), 1, cv::LINE_AA);
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 123) {
            /* IRON FILINGS: you are the magnet. Thousands of filings are
             * DRAWN to your outline, align along your field, and cling in
             * a shimmering shell — drag your body and they follow, flow
             * around you, and re-settle. fx = magnet strength. */
            const int NF = 1500;
            static float fxs[NF], fys[NF], fvx[NF], fvy[NF];
            static int finit2 = 0;
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!finit2) {
                for (int i2 = 0; i2 < NF; i2++) {
                    fxs[i2] = uhash(i2,1,61)*img.cols;
                    fys[i2] = uhash(i2,2,61)*img.rows;
                    fvx[i2] = fvy[i2] = 0;
                }
                finit2 = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            for (int i2 = 0; i2 < NF; i2++) {
                int xi = std::max(2,std::min(img.cols-3,(int)fxs[i2]));
                int yi = std::max(2,std::min(img.rows-3,(int)fys[i2]));
                float m = g_seg.mask.at<float>(yi,xi);
                /* the mask gradient IS the field: filings ride it toward
                 * your edge, then sit in the rim where the field is
                 * strongest */
                float gx2 = g_seg.mask.at<float>(yi, std::min(img.cols-1,xi+3))
                          - g_seg.mask.at<float>(yi, std::max(0,xi-3));
                float gy2 = g_seg.mask.at<float>(std::min(img.rows-1,yi+3), xi)
                          - g_seg.mask.at<float>(std::max(0,yi-3), xi);
                float gl = std::sqrt(gx2*gx2+gy2*gy2);
                if (gl > 0.02f) {                     /* near your edge */
                    /* attraction along the field + tangential alignment */
                    float attract = (m < 0.5f) ? 1.0f : -1.4f;   /* stay outside */
                    fvx[i2] += gx2/gl * attract * 0.55f * fx;
                    fvy[i2] += gy2/gl * attract * 0.55f * fx;
                    fvx[i2] += -gy2/gl * 0.32f * fx;             /* swirl */
                    fvy[i2] +=  gx2/gl * 0.32f * fx;
                } else {
                    /* far away: drift slowly toward the body centre */
                    float d = m - prev.at<float>(yi,xi);
                    if (std::fabs(d) > 0.2f) {                   /* your wake */
                        fvx[i2] += (uhash(i2,tk%43,3)-0.5f)*4.0f;
                        fvy[i2] += (uhash(i2,tk%47,5)-0.5f)*4.0f;
                    }
                    fvy[i2] += 0.02f;
                }
                fvx[i2] *= 0.86f; fvy[i2] *= 0.86f;
                fxs[i2] += fvx[i2]; fys[i2] += fvy[i2];
                if (fxs[i2] < 1) { fxs[i2]=1; fvx[i2]=0; }
                if (fxs[i2] > img.cols-2) { fxs[i2]=img.cols-2; fvx[i2]=0; }
                if (fys[i2] < 1) { fys[i2]=1; fvy[i2]=0; }
                if (fys[i2] > img.rows-2) { fys[i2]=img.rows-2; fvy[i2]=0; }
                /* draw each filing as a short needle along the field */
                float sp = std::sqrt(fvx[i2]*fvx[i2]+fvy[i2]*fvy[i2]);
                int br = std::min(255, 120 + (int)(sp*55) + (int)(gl*300));
                float dxn = gl > 0.02f ? -gy2/gl : 1, dyn = gl > 0.02f ? gx2/gl : 0;
                cv::line(img, {(int)(fxs[i2]-dxn*2.2f),(int)(fys[i2]-dyn*2.2f)},
                              {(int)(fxs[i2]+dxn*2.2f),(int)(fys[i2]+dyn*2.2f)},
                         cv::Scalar(br, br, (int)(br*0.85f)), 1, cv::LINE_AA);
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 114) {
            /* SMOKE FLOW: a real fluid — a velocity grid your body stirs,
             * with dye advected through it (semi-Lagrangian). Wave a hand
             * and you push actual currents; the smoke curls and lingers.
             * fx = how much force your motion injects. */
            const int GW = 96, GH = 72;
            static float vxg[GW*GH], vyg[GW*GH], dye[GW*GH];
            static float vx2[GW*GH], vy2[GW*GH], dy2[GW*GH];
            static int finit = 0;
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!finit) {
                std::memset(vxg,0,sizeof vxg); std::memset(vyg,0,sizeof vyg);
                std::memset(dye,0,sizeof dye); finit = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            /* 1. inject: body motion -> velocity, body edge -> dye source */
            for (int gy3 = 1; gy3 < GH-1; gy3++)
                for (int gx3 = 1; gx3 < GW-1; gx3++) {
                    int px3 = gx3 * img.cols / GW, py3 = gy3 * img.rows / GH;
                    float d = g_seg.mask.at<float>(py3,px3)
                            - prev.at<float>(py3,px3);
                    if (std::fabs(d) > 0.15f) {
                        int x1 = std::min(img.cols-1, px3+4);
                        int x0 = std::max(0, px3-4);
                        int y1 = std::min(img.rows-1, py3+4);
                        int y0 = std::max(0, py3-4);
                        float gx2 = g_seg.mask.at<float>(py3,x1)
                                  - g_seg.mask.at<float>(py3,x0);
                        float gy2 = g_seg.mask.at<float>(y1,px3)
                                  - g_seg.mask.at<float>(y0,px3);
                        float l2 = std::sqrt(gx2*gx2+gy2*gy2)+1e-3f;
                        vxg[gy3*GW+gx3] -= gx2/l2 * d * 9.0f * fx;
                        vyg[gy3*GW+gx3] -= gy2/l2 * d * 9.0f * fx;
                    }
                    float m = g_seg.mask.at<float>(py3,px3);
                    if (m > 0.35f && m < 0.85f)         /* your rim smokes */
                        dye[gy3*GW+gx3] = std::min(1.0f,
                            dye[gy3*GW+gx3] + 0.35f);
                }
            /* a little buoyancy + swirl so it always lives */
            for (int i2 = 0; i2 < GW*GH; i2++) {
                vyg[i2] -= dye[i2] * 0.012f;
                vxg[i2] += std::sin((i2/GW)*0.3f + tk*0.02f) * 0.004f;
            }
            /* 2. advect velocity and dye (semi-Lagrangian backtrace) */
            for (int gy3 = 1; gy3 < GH-1; gy3++)
                for (int gx3 = 1; gx3 < GW-1; gx3++) {
                    int i2 = gy3*GW+gx3;
                    float sx2 = gx3 - vxg[i2] * 0.55f;
                    float sy2 = gy3 - vyg[i2] * 0.55f;
                    sx2 = std::max(0.0f, std::min((float)GW-1.01f, sx2));
                    sy2 = std::max(0.0f, std::min((float)GH-1.01f, sy2));
                    int ix = (int)sx2, iy = (int)sy2;
                    float fxr = sx2-ix, fyr = sy2-iy;
                    #define SMP(A) ((A)[iy*GW+ix]*(1-fxr)*(1-fyr) + \
                                    (A)[iy*GW+ix+1]*fxr*(1-fyr) + \
                                    (A)[(iy+1)*GW+ix]*(1-fxr)*fyr + \
                                    (A)[(iy+1)*GW+ix+1]*fxr*fyr)
                    vx2[i2] = SMP(vxg) * 0.985f;
                    vy2[i2] = SMP(vyg) * 0.985f;
                    dy2[i2] = SMP(dye) * 0.988f;
                    #undef SMP
                }
            std::memcpy(vxg, vx2, sizeof vxg);
            std::memcpy(vyg, vy2, sizeof vyg);
            std::memcpy(dye, dy2, sizeof dye);
            /* 3. render the dye, warm and glowing, never over you */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                float gy4 = (float)y * GH / img.rows;
                int iy = std::min(GH-2, (int)gy4);
                float fy2 = gy4 - iy;
                for (int x = 0; x < img.cols; x++) {
                    float gx4 = (float)x * GW / img.cols;
                    int ix = std::min(GW-2, (int)gx4);
                    float fx2 = gx4 - ix;
                    float d = dye[iy*GW+ix]*(1-fx2)*(1-fy2)
                            + dye[iy*GW+ix+1]*fx2*(1-fy2)
                            + dye[(iy+1)*GW+ix]*(1-fx2)*fy2
                            + dye[(iy+1)*GW+ix+1]*fx2*fy2;
                    d *= (1.0f - mm[x]);
                    if (d < 0.02f) continue;
                    if (d > 1) d = 1;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(d*90));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(d*165));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(d*235));
                }
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 115) {
            /* RIPPLE POOL: the room is a water surface. Every move you
             * make drops waves into it — a real wave equation — and the
             * ripples REFRACT the room image as they spread and bounce
             * off the walls. fx = wave energy. */
            const int RW = 200, RH = 150;
            static float w1[RW*RH], w0[RW*RH];
            static int rinit = 0;
            static cv::Mat prev;
            if (!rinit) {
                std::memset(w1,0,sizeof w1); std::memset(w0,0,sizeof w0);
                rinit = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            /* poke the surface where your body moved */
            for (int gy3 = 2; gy3 < RH-2; gy3++)
                for (int gx3 = 2; gx3 < RW-2; gx3++) {
                    int px3 = gx3*img.cols/RW, py3 = gy3*img.rows/RH;
                    float d = g_seg.mask.at<float>(py3,px3)
                            - prev.at<float>(py3,px3);
                    if (std::fabs(d) > 0.25f)
                        w1[gy3*RW+gx3] -= d * 26.0f * fx;
                }
            /* wave equation: w_next = 2w - w_prev + c^2 * laplacian(w) */
            for (int gy3 = 1; gy3 < RH-1; gy3++)
                for (int gx3 = 1; gx3 < RW-1; gx3++) {
                    int i2 = gy3*RW+gx3;
                    float lap = w1[i2-1] + w1[i2+1] + w1[i2-RW] + w1[i2+RW]
                              - 4*w1[i2];
                    float nv = 2*w1[i2] - w0[i2] + lap * 0.22f;
                    w0[i2] = nv * 0.992f;               /* damping */
                }
            std::swap(*(float(*)[RW*RH])w0, *(float(*)[RW*RH])w1);
            /* refract the room through the wave gradient */
            cv::Mat keep = img.clone();
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                int gy3 = std::min(RH-2, std::max(1, y*RH/img.rows));
                for (int x = 0; x < img.cols; x++) {
                    if (mm[x] > 0.5f) continue;         /* you stay dry */
                    int gx3 = std::min(RW-2, std::max(1, x*RW/img.cols));
                    int i2 = gy3*RW+gx3;
                    float dx3 = w1[i2+1] - w1[i2-1];
                    float dy3 = w1[i2+RW] - w1[i2-RW];
                    int sx2 = x + (int)(dx3 * 26.0f);
                    int sy2 = y + (int)(dy3 * 26.0f);
                    sx2 = std::max(0, std::min(img.cols-1, sx2));
                    sy2 = std::max(0, std::min(img.rows-1, sy2));
                    cv::Vec3b c = keep.at<cv::Vec3b>(sy2, sx2);
                    float spec = std::max(0.0f, -(dx3+dy3) * 90.0f);
                    ip[x][0]=clamp_int_u8((int)(c[0] + spec*1.3f));
                    ip[x][1]=clamp_int_u8((int)(c[1] + spec*1.15f));
                    ip[x][2]=clamp_int_u8((int)(c[2] + spec*0.9f));
                }
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 116) {
            /* CLOTH NET: a spring-lattice net hangs in front of you. Push
             * it with your hands — it stretches, sags, wobbles and
             * settles. Real Verlet cloth. fx = how bouncy it is. */
            const int CW = 26, CH = 18;
            static float cxs[CW*CH], cys[CW*CH], oxs[CW*CH], oys[CW*CH];
            static int cinit = 0, cw2 = 0, ch2 = 0;
            static cv::Mat prev;
            if (!cinit || cw2 != img.cols || ch2 != img.rows) {
                for (int gy3 = 0; gy3 < CH; gy3++)
                    for (int gx3 = 0; gx3 < CW; gx3++) {
                        int i2 = gy3*CW+gx3;
                        cxs[i2] = (gx3+0.5f)*img.cols/CW;
                        cys[i2] = (gy3+0.5f)*img.rows/(CH+2);
                        oxs[i2] = cxs[i2]; oys[i2] = cys[i2];
                    }
                cinit = 1; cw2 = img.cols; ch2 = img.rows;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            float rest_x = (float)img.cols/CW, rest_y = (float)img.rows/(CH+2);
            /* Verlet integrate + body push */
            for (int i2 = 0; i2 < CW*CH; i2++) {
                if (i2 < CW) continue;                   /* top row pinned */
                float vx3 = (cxs[i2]-oxs[i2])*0.97f, vy3 = (cys[i2]-oys[i2])*0.97f;
                oxs[i2] = cxs[i2]; oys[i2] = cys[i2];
                int xi = std::max(0,std::min(img.cols-1,(int)cxs[i2]));
                int yi = std::max(0,std::min(img.rows-1,(int)cys[i2]));
                float d = g_seg.mask.at<float>(yi,xi) - prev.at<float>(yi,xi);
                float m = g_seg.mask.at<float>(yi,xi);
                if (std::fabs(d) > 0.2f || m > 0.6f) {
                    int x1=std::min(img.cols-1,xi+4), x0=std::max(0,xi-4);
                    int y1=std::min(img.rows-1,yi+4), y0=std::max(0,yi-4);
                    float gx2 = g_seg.mask.at<float>(yi,x1)-g_seg.mask.at<float>(yi,x0);
                    float gy2 = g_seg.mask.at<float>(y1,xi)-g_seg.mask.at<float>(y0,xi);
                    float l2 = std::sqrt(gx2*gx2+gy2*gy2)+1e-3f;
                    float push = (std::fabs(d)>0.2f ? 5.0f*fx : 2.0f);
                    vx3 -= gx2/l2 * push;
                    vy3 -= gy2/l2 * push;
                }
                cxs[i2] += vx3;
                cys[i2] += vy3 + 0.35f;                  /* gravity */
            }
            /* satisfy spring constraints */
            for (int it = 0; it < 3; it++)
                for (int gy3 = 0; gy3 < CH; gy3++)
                    for (int gx3 = 0; gx3 < CW; gx3++) {
                        int i2 = gy3*CW+gx3;
                        if (gx3+1 < CW) {
                            int j2 = i2+1;
                            float dx3 = cxs[j2]-cxs[i2], dy3 = cys[j2]-cys[i2];
                            float dl = std::sqrt(dx3*dx3+dy3*dy3)+1e-4f;
                            float corr = (dl - rest_x)/dl*0.5f;
                            float ax = dx3*corr, ay = dy3*corr;
                            if (i2 >= CW) { cxs[i2]+=ax; cys[i2]+=ay; }
                            if (j2 >= CW) { cxs[j2]-=ax; cys[j2]-=ay; }
                        }
                        if (gy3+1 < CH) {
                            int j2 = i2+CW;
                            float dx3 = cxs[j2]-cxs[i2], dy3 = cys[j2]-cys[i2];
                            float dl = std::sqrt(dx3*dx3+dy3*dy3)+1e-4f;
                            float corr = (dl - rest_y)/dl*0.5f;
                            float ax = dx3*corr, ay = dy3*corr;
                            if (i2 >= CW) { cxs[i2]+=ax; cys[i2]+=ay; }
                            cxs[j2]-=ax; cys[j2]-=ay;
                        }
                    }
            /* draw the net */
            for (int gy3 = 0; gy3 < CH; gy3++)
                for (int gx3 = 0; gx3 < CW; gx3++) {
                    int i2 = gy3*CW+gx3;
                    cv::Point p2((int)cxs[i2],(int)cys[i2]);
                    if (gx3+1 < CW) {
                        int j2=i2+1;
                        float st = std::fabs(std::sqrt(
                            (cxs[j2]-cxs[i2])*(cxs[j2]-cxs[i2])+
                            (cys[j2]-cys[i2])*(cys[j2]-cys[i2])) - rest_x)/rest_x;
                        int b2 = std::min(255, 90+(int)(st*420));
                        cv::line(img, p2, {(int)cxs[j2],(int)cys[j2]},
                                 cv::Scalar(200-b2/2, 140+b2/3, b2), 1, cv::LINE_AA);
                    }
                    if (gy3+1 < CH) {
                        int j2=i2+CW;
                        cv::line(img, p2, {(int)cxs[j2],(int)cys[j2]},
                                 cv::Scalar(190, 150, 110), 1, cv::LINE_AA);
                    }
                    cv::circle(img, p2, 2, cv::Scalar(230,220,190),
                               cv::FILLED, cv::LINE_AA);
                }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 117) {
            /* SWARM: a flock of glowing insects with real boid rules —
             * they cohere, align, avoid each other, and FLEE your hands.
             * Sweep at them and the swarm scatters and re-forms.
             * fx = how scared they are. */
            const int NB = 190;
            static float bx3[NB], by3[NB], bvx[NB], bvy[NB];
            static int binit2 = 0;
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!binit2) {
                for (int i2 = 0; i2 < NB; i2++) {
                    bx3[i2] = uhash(i2,1,11)*img.cols;
                    by3[i2] = uhash(i2,2,11)*img.rows;
                    bvx[i2] = (uhash(i2,3,11)-0.5f)*2;
                    bvy[i2] = (uhash(i2,4,11)-0.5f)*2;
                }
                binit2 = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            /* flock centre + mean heading (global approximation: cheap) */
            float mx2=0,my2=0,mvx=0,mvy=0;
            for (int i2 = 0; i2 < NB; i2++) {
                mx2+=bx3[i2]; my2+=by3[i2]; mvx+=bvx[i2]; mvy+=bvy[i2];
            }
            mx2/=NB; my2/=NB; mvx/=NB; mvy/=NB;
            for (int i2 = 0; i2 < NB; i2++) {
                /* cohesion + alignment */
                bvx[i2] += (mx2-bx3[i2])*0.0009f + (mvx-bvx[i2])*0.03f;
                bvy[i2] += (my2-by3[i2])*0.0009f + (mvy-bvy[i2])*0.03f;
                /* separation from a few neighbours */
                for (int k2 = 1; k2 <= 3; k2++) {
                    int j2 = (i2+k2*37) % NB;
                    float dx3 = bx3[i2]-bx3[j2], dy3 = by3[i2]-by3[j2];
                    float d2 = dx3*dx3+dy3*dy3;
                    if (d2 < 420 && d2 > 1e-3f) {
                        bvx[i2] += dx3/d2 * 5.5f;
                        bvy[i2] += dy3/d2 * 5.5f;
                    }
                }
                /* FLEE you: strongest where your body just moved */
                int xi = std::max(0,std::min(img.cols-1,(int)bx3[i2]));
                int yi = std::max(0,std::min(img.rows-1,(int)by3[i2]));
                float m = g_seg.mask.at<float>(yi,xi);
                float d = m - prev.at<float>(yi,xi);
                if (m > 0.25f || std::fabs(d) > 0.15f) {
                    int x1=std::min(img.cols-1,xi+5), x0=std::max(0,xi-5);
                    int y1=std::min(img.rows-1,yi+5), y0=std::max(0,yi-5);
                    float gx2 = g_seg.mask.at<float>(yi,x1)-g_seg.mask.at<float>(yi,x0);
                    float gy2 = g_seg.mask.at<float>(y1,xi)-g_seg.mask.at<float>(y0,xi);
                    float l2 = std::sqrt(gx2*gx2+gy2*gy2)+1e-3f;
                    float scare = (2.2f + std::fabs(d)*16.0f) * fx;
                    bvx[i2] -= gx2/l2 * scare;
                    bvy[i2] -= gy2/l2 * scare;
                }
                /* wander + speed limit */
                bvx[i2] += (uhash(i2,tk%31,13)-0.5f)*0.5f;
                bvy[i2] += (uhash(i2,tk%29,17)-0.5f)*0.5f;
                float sp = std::sqrt(bvx[i2]*bvx[i2]+bvy[i2]*bvy[i2]);
                float mxs = 6.5f;
                if (sp > mxs) { bvx[i2]*=mxs/sp; bvy[i2]*=mxs/sp; }
                if (sp < 1.2f && sp > 1e-3f) { bvx[i2]*=1.2f/sp; bvy[i2]*=1.2f/sp; }
                bx3[i2] += bvx[i2]; by3[i2] += bvy[i2];
                if (bx3[i2] < 0) { bx3[i2]=0; bvx[i2]=std::fabs(bvx[i2]); }
                if (bx3[i2] > img.cols-1) { bx3[i2]=img.cols-1; bvx[i2]=-std::fabs(bvx[i2]); }
                if (by3[i2] < 0) { by3[i2]=0; bvy[i2]=std::fabs(bvy[i2]); }
                if (by3[i2] > img.rows-1) { by3[i2]=img.rows-1; bvy[i2]=-std::fabs(bvy[i2]); }
                float tw = 0.6f + 0.4f*std::sin(tk*0.3f + i2);
                cv::circle(img, {(int)bx3[i2],(int)by3[i2]}, 2,
                           cv::Scalar(60*tw, 230*tw, 255*tw), cv::FILLED, cv::LINE_AA);
                cv::line(img, {(int)bx3[i2],(int)by3[i2]},
                         {(int)(bx3[i2]-bvx[i2]*2.2f),(int)(by3[i2]-bvy[i2]*2.2f)},
                         cv::Scalar(30*tw,150*tw,190*tw), 1, cv::LINE_AA);
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 118) {
            /* SAND FALL: grains rain down and PILE UP on your head and
             * shoulders — they stack where you're solid, tumble off your
             * slopes, and get swept away when you move. Shake them off.
             * fx = grain count. */
            const int NG = 1400;
            static float gx4[NG], gy4[NG], gvy[NG], gvx[NG];
            static int ginit = 0, settled[NG];
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!ginit) {
                for (int i2 = 0; i2 < NG; i2++) {
                    gx4[i2] = uhash(i2,1,19)*img.cols;
                    gy4[i2] = -uhash(i2,2,19)*img.rows*2;
                    gvy[i2] = 1.2f + uhash(i2,3,19)*1.6f;
                    gvx[i2] = 0; settled[i2] = 0;
                }
                ginit = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            int active = (int)(NG * std::min(1.0f, fx/2.0f + 0.25f));
            for (int i2 = 0; i2 < active; i2++) {
                int xi = std::max(0,std::min(img.cols-1,(int)gx4[i2]));
                int yi = std::max(0,std::min(img.rows-1,(int)gy4[i2]));
                if (settled[i2]) {
                    /* swept off by your motion? */
                    float d = g_seg.mask.at<float>(yi,xi) - prev.at<float>(yi,xi);
                    if (std::fabs(d) > 0.18f) {
                        settled[i2] = 0;
                        gvx[i2] = (uhash(i2,tk%23,7)-0.5f)*7.0f;
                        gvy[i2] = -1.5f;
                    }
                } else {
                    gvy[i2] += 0.16f;
                    gvx[i2] *= 0.96f;
                    gx4[i2] += gvx[i2];
                    gy4[i2] += gvy[i2];
                    xi = std::max(0,std::min(img.cols-1,(int)gx4[i2]));
                    yi = std::max(0,std::min(img.rows-1,(int)gy4[i2]));
                    /* landed on you? */
                    if (gy4[i2] > 0 && g_seg.mask.at<float>(yi,xi) > 0.55f) {
                        /* tumble down the slope instead of sticking to walls */
                        int x1=std::min(img.cols-1,xi+3), x0=std::max(0,xi-3);
                        float gxs = g_seg.mask.at<float>(yi,x1)
                                  - g_seg.mask.at<float>(yi,x0);
                        if (std::fabs(gxs) > 0.30f) {      /* steep: slide */
                            gx4[i2] -= gxs * 5.0f;
                            gy4[i2] -= 1.0f;
                            gvy[i2] *= 0.5f;
                        } else {
                            settled[i2] = 1;
                            gy4[i2] -= 1.0f;
                            gvy[i2] = 0; gvx[i2] = 0;
                        }
                    }
                    if (gy4[i2] > img.rows + 6) {          /* recycle */
                        gx4[i2] = uhash(i2,tk%97,23)*img.cols;
                        gy4[i2] = -6 - uhash(i2,tk%89,29)*60;
                        gvy[i2] = 1.2f + uhash(i2,3,19)*1.6f;
                        gvx[i2] = 0;
                    }
                }
                float b2 = settled[i2] ? 1.0f : 0.85f;
                cv::circle(img, {(int)gx4[i2],(int)gy4[i2]}, settled[i2]?2:1,
                           cv::Scalar(120*b2, 200*b2, 235*b2), cv::FILLED);
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 109) {
            /* BUBBLE POP: bubbles rise from below — touch one and it
             * pops in a burst. Combo counter rewards fast chains.
             * fx = bubble rate. */
            static float bx[26], by2[26], bs[26], bvy[26];
            static int binit = 0, popped = 0, combo = 0, combo_t = 0;
            static cv::Mat prev;
            unsigned tk = g_seg.counter;
            if (!binit) {
                for (int i = 0; i < 26; i++) {
                    bx[i] = uhash(i,1,3)*img.cols;
                    by2[i] = img.rows + uhash(i,2,3)*img.rows*1.5f;
                    bs[i] = 12 + uhash(i,3,3)*22;
                    bvy[i] = -(0.9f + uhash(i,4,3)*1.6f);
                }
                binit = 1;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            for (int i = 0; i < 26; i++) {
                by2[i] += bvy[i] * fx;
                bx[i] += std::sin((by2[i]+i*40)*0.02f) * 0.9f;
                if (by2[i] < -bs[i]) {                    /* respawn below */
                    bx[i] = uhash(i, tk%97, 5)*img.cols;
                    by2[i] = img.rows + bs[i] + uhash(i,tk%53,6)*120;
                }
                if (bx[i] < 0 || bx[i] >= img.cols) continue;
                if (by2[i] < 0 || by2[i] >= img.rows) continue;
                float m = g_seg.mask.at<float>((int)by2[i], (int)bx[i]);
                if (m > 0.5f) {                            /* POP */
                    for (int k2 = 0; k2 < 10; k2++) {      /* burst */
                        float a2 = 6.2832f*k2/10;
                        cv::line(img, {(int)bx[i],(int)by2[i]},
                            {(int)(bx[i]+std::cos(a2)*bs[i]*1.7f),
                             (int)(by2[i]+std::sin(a2)*bs[i]*1.7f)},
                            cv::Scalar(255,240,180), 2, cv::LINE_AA);
                    }
                    popped++;
                    combo = (combo_t > 0) ? combo+1 : 1;
                    combo_t = 40;
                    bx[i] = uhash(i, tk%89, 7)*img.cols;
                    by2[i] = img.rows + bs[i] + 40;
                    continue;
                }
                /* draw bubble: rim + highlight, see-through */
                cv::circle(img, {(int)bx[i],(int)by2[i]}, (int)bs[i],
                           cv::Scalar(255,225,170), 2, cv::LINE_AA);
                cv::circle(img, {(int)(bx[i]-bs[i]*0.3f),(int)(by2[i]-bs[i]*0.3f)},
                           std::max(2,(int)(bs[i]*0.22f)),
                           cv::Scalar(255,255,240), cv::FILLED, cv::LINE_AA);
            }
            if (combo_t > 0) combo_t--; else combo = 0;
            g_seg.mask.copyTo(prev);
            char hud[48];
            snprintf(hud, sizeof hud, "POPPED %d", popped);
            cv::putText(img, hud, {20,46}, cv::FONT_HERSHEY_DUPLEX, 1.0,
                        cv::Scalar(20,20,20), 4, cv::LINE_AA);
            cv::putText(img, hud, {20,46}, cv::FONT_HERSHEY_DUPLEX, 1.0,
                        cv::Scalar(255,230,170), 2, cv::LINE_AA);
            if (combo > 2) {
                snprintf(hud, sizeof hud, "x%d COMBO", combo);
                cv::putText(img, hud, {20,80}, cv::FONT_HERSHEY_DUPLEX, 0.8,
                            cv::Scalar(60,220,255), 2, cv::LINE_AA);
            }
            return;
        }
        if (mode == 110) {
            /* PUNCHING BAG: a heavy bag hangs from the ceiling. Punch it
             * — your sweep direction and speed become angular impulse;
             * it swings, damps, and a POWER meter reads your best hit.
             * fx = bag weight (low = light, swings wildly). */
            static float ang = 0, av = 0;
            static float power = 0, bestp = 0;
            static int flash = 0;
            static cv::Mat prev;
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            float L = img.rows * 0.42f;                  /* rope length */
            float px2 = img.cols*0.5f + std::sin(ang)*L;
            float py2 = 10 + std::cos(ang)*L;
            float R2 = std::max(26.0f, img.cols*0.075f);
            BodyProbe b = probe_body(g_seg.mask, prev, px2, py2, R2);
            if (b.hit) {
                float pw2 = std::sqrt(b.pux*b.pux + b.puy*b.puy);
                /* tangential component of your push drives the swing */
                float tx = std::cos(ang), ty = -std::sin(ang);
                float torque = (b.pux*tx + b.puy*ty) * 0.0016f / fx;
                if (std::fabs(torque) < 1e-4f)            /* dead-on shove */
                    torque = (b.nx*tx + b.ny*ty) * 0.004f / fx;
                av += torque;
                if (pw2 > power*0.9f) { power = pw2; flash = 10; }
                if (power > bestp) bestp = power;
            }
            av -= std::sin(ang) * 0.0016f;               /* gravity */
            av *= 0.985f;                                /* damping */
            if (av > 0.08f) av = 0.08f;
            if (av < -0.08f) av = -0.08f;
            ang += av;
            power *= 0.97f;
            if (flash > 0) flash--;
            /* draw: chain, bag body, straps */
            cv::line(img, {img.cols/2, 4}, {(int)px2,(int)py2},
                     cv::Scalar(150,150,155), 3, cv::LINE_AA);
            cv::ellipse(img, {(int)px2,(int)(py2+R2*0.55f)},
                        {(int)(R2*0.92f),(int)(R2*1.55f)},
                        ang*57.3f, 0, 360,
                        flash>0 ? cv::Scalar(60,90,220) : cv::Scalar(35,45,120),
                        cv::FILLED, cv::LINE_AA);
            cv::ellipse(img, {(int)px2,(int)(py2+R2*0.55f)},
                        {(int)(R2*0.92f),(int)(R2*1.55f)},
                        ang*57.3f, 0, 360, cv::Scalar(20,25,60), 3, cv::LINE_AA);
            cv::ellipse(img, {(int)(px2-R2*0.3f),(int)(py2+R2*0.2f)},
                        {(int)(R2*0.22f),(int)(R2*0.8f)}, ang*57.3f, 0, 360,
                        cv::Scalar(120,140,220), cv::FILLED, cv::LINE_AA);
            /* power meter */
            int bw = (int)(std::min(1.0f, power/11.0f) * (img.cols*0.3f));
            cv::rectangle(img, {20, img.rows-40},
                          {20+(int)(img.cols*0.3f), img.rows-22},
                          cv::Scalar(60,60,65), 2);
            cv::rectangle(img, {20, img.rows-40}, {20+bw, img.rows-22},
                          cv::Scalar(60,180,255), cv::FILLED);
            char hud[48];
            snprintf(hud, sizeof hud, "POWER  best %.0f", bestp*9);
            cv::putText(img, hud, {20, img.rows-48}, cv::FONT_HERSHEY_PLAIN,
                        1.2, cv::Scalar(220,220,230), 1, cv::LINE_AA);
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 111) {
            /* GOALKEEPER: shots fly at you out of the background, growing
             * as they come. Block them with ANY part of your body — a
             * save flashes green, a goal flashes red. fx = shot rate. */
            static float sx3[6], sy3[6], sz[6], svx[6], svy[6];
            static int alive[6], sinit = 0, saves = 0, goals = 0, flash = 0;
            unsigned tk = g_seg.counter;
            if (!sinit) { for (int i=0;i<6;i++) alive[i]=0; sinit=1; }
            /* spawn */
            if ((tk % (unsigned)std::max(14.0f, 46.0f/fx)) == 0) {
                for (int i = 0; i < 6; i++)
                    if (!alive[i]) {
                        sx3[i] = img.cols*(0.15f+0.7f*uhash(i,tk%71,2));
                        sy3[i] = img.rows*(0.15f+0.5f*uhash(i,tk%67,3));
                        sz[i] = 0.06f;
                        svx[i] = (uhash(i,tk%59,4)-0.5f)*1.2f;
                        svy[i] = (uhash(i,tk%61,5)-0.5f)*0.9f;
                        alive[i] = 1;
                        break;
                    }
            }
            for (int i = 0; i < 6; i++) {
                if (!alive[i]) continue;
                sz[i] *= 1.055f;                    /* coming at you */
                sx3[i] += svx[i]; sy3[i] += svy[i];
                float R2 = 8 + sz[i]*110;
                if (sx3[i] < 0 || sx3[i] >= img.cols ||
                    sy3[i] < 0 || sy3[i] >= img.rows) { alive[i]=0; continue; }
                float m = g_seg.mask.at<float>((int)sy3[i], (int)sx3[i]);
                if (m > 0.5f && sz[i] > 0.20f) {    /* SAVED */
                    saves++; alive[i] = 0; flash = 12;
                    for (int k2=0;k2<12;k2++) {
                        float a2=6.2832f*k2/12;
                        cv::line(img, {(int)sx3[i],(int)sy3[i]},
                          {(int)(sx3[i]+std::cos(a2)*R2*1.8f),
                           (int)(sy3[i]+std::sin(a2)*R2*1.8f)},
                          cv::Scalar(120,255,120), 2, cv::LINE_AA);
                    }
                    continue;
                }
                if (sz[i] > 1.15f) { goals++; alive[i]=0; flash = -12; continue; }
                /* draw the shot: dark core, motion ring */
                cv::circle(img, {(int)sx3[i],(int)sy3[i]}, (int)(R2*1.35f),
                           cv::Scalar(40,40,50), 2, cv::LINE_AA);
                cv::circle(img, {(int)sx3[i],(int)sy3[i]}, (int)R2,
                           cv::Scalar(70,90,240), cv::FILLED, cv::LINE_AA);
                cv::circle(img, {(int)(sx3[i]-R2*0.3f),(int)(sy3[i]-R2*0.3f)},
                           std::max(2,(int)(R2*0.25f)),
                           cv::Scalar(200,220,255), cv::FILLED, cv::LINE_AA);
            }
            if (flash > 0) { flash--;
                cv::rectangle(img, {0,0}, {img.cols-1,img.rows-1},
                              cv::Scalar(80,220,80), 8);
            } else if (flash < 0) { flash++;
                cv::rectangle(img, {0,0}, {img.cols-1,img.rows-1},
                              cv::Scalar(70,70,235), 8);
            }
            char hud[48];
            snprintf(hud, sizeof hud, "SAVES %d   GOALS %d", saves, goals);
            cv::putText(img, hud, {20,46}, cv::FONT_HERSHEY_DUPLEX, 0.95,
                        cv::Scalar(20,20,20), 4, cv::LINE_AA);
            cv::putText(img, hud, {20,46}, cv::FONT_HERSHEY_DUPLEX, 0.95,
                        cv::Scalar(240,240,245), 2, cv::LINE_AA);
            return;
        }
        if (mode == 112) {
            /* MOTION PAINT: your movement sprays paint onto the room —
             * hue follows the direction you swipe, thickness follows
             * speed. The canvas persists and slowly dries.
             * fx = paint fade (low = permanent). */
            static cv::Mat canvas, prev;
            if (canvas.empty() || canvas.size() != img.size())
                canvas = cv::Mat::zeros(img.size(), CV_8UC3);
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            /* paint where the matte moved */
            for (int y = 2; y < img.rows-2; y += 2)
                for (int x = 2; x < img.cols-2; x += 2) {
                    float d = g_seg.mask.at<float>(y,x) - prev.at<float>(y,x);
                    if (d < 0.25f) continue;              /* newly covered */
                    float gx2 = g_seg.mask.at<float>(y,x+2)
                              - g_seg.mask.at<float>(y,x-2);
                    float gy2 = g_seg.mask.at<float>(y+2,x)
                              - g_seg.mask.at<float>(y-2,x);
                    float hue = std::atan2(gy2, gx2) * 2.0f;
                    cv::Scalar col((std::sin(hue+4.19f)*.5f+.5f)*255,
                                   (std::sin(hue+2.09f)*.5f+.5f)*255,
                                   (std::sin(hue)*.5f+.5f)*255);
                    cv::circle(canvas, {x,y}, (int)(2 + d*5), col,
                               cv::FILLED, cv::LINE_AA);
                }
            canvas *= (1.0 - 0.006 * fx);                 /* drying */
            /* composite paint over the room, but never over you */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *cp = canvas.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    int pl = cp[x][0]+cp[x][1]+cp[x][2];
                    if (pl < 24) continue;
                    float a = (1.0f - mm[x]) * std::min(1.0f, pl/300.0f);
                    ip[x][0]=(uchar)(ip[x][0]+(cp[x][0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(cp[x][1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(cp[x][2]-ip[x][2])*a);
                }
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 113) {
            /* FORCE FIELD: a grid of glowing particles floats in the
             * room. Sweep your hands and you SHOVE them — they scatter,
             * drift, and slowly spring back home. Stir the air.
             * fx = field strength. */
            static float pxs[300], pys[300], pvx[300], pvy[300];
            static float hx[300], hy[300];
            static int pinit = 0, pw2 = 0, ph2 = 0;
            static cv::Mat prev;
            if (!pinit || pw2 != img.cols || ph2 != img.rows) {
                int i = 0;
                for (int gy3 = 0; gy3 < 15 && i < 300; gy3++)
                    for (int gx3 = 0; gx3 < 20 && i < 300; gx3++, i++) {
                        hx[i] = (gx3+0.5f) * img.cols / 20.0f;
                        hy[i] = (gy3+0.5f) * img.rows / 15.0f;
                        pxs[i] = hx[i]; pys[i] = hy[i];
                        pvx[i] = pvy[i] = 0;
                    }
                pinit = 1; pw2 = img.cols; ph2 = img.rows;
            }
            if (prev.empty() || prev.size() != g_seg.mask.size())
                g_seg.mask.copyTo(prev);
            for (int i = 0; i < 300; i++) {
                int xi = std::max(0, std::min(img.cols-1, (int)pxs[i]));
                int yi = std::max(0, std::min(img.rows-1, (int)pys[i]));
                float d = g_seg.mask.at<float>(yi,xi) - prev.at<float>(yi,xi);
                if (std::fabs(d) > 0.2f) {          /* your body moved here */
                    float gx2 = g_seg.mask.at<float>(yi, std::min(img.cols-1,xi+3))
                              - g_seg.mask.at<float>(yi, std::max(0,xi-3));
                    float gy2 = g_seg.mask.at<float>(std::min(img.rows-1,yi+3), xi)
                              - g_seg.mask.at<float>(std::max(0,yi-3), xi);
                    float l2 = std::sqrt(gx2*gx2+gy2*gy2) + 1e-3f;
                    pvx[i] -= gx2/l2 * 5.5f * fx * (d > 0 ? 1.0f : 0.4f);
                    pvy[i] -= gy2/l2 * 5.5f * fx * (d > 0 ? 1.0f : 0.4f);
                }
                if (g_seg.mask.at<float>(yi,xi) > 0.6f) {   /* pushed out */
                    float gx2 = g_seg.mask.at<float>(yi, std::min(img.cols-1,xi+3))
                              - g_seg.mask.at<float>(yi, std::max(0,xi-3));
                    float gy2 = g_seg.mask.at<float>(std::min(img.rows-1,yi+3), xi)
                              - g_seg.mask.at<float>(std::max(0,yi-3), xi);
                    pvx[i] -= gx2 * 3.0f; pvy[i] -= gy2 * 3.0f;
                }
                pvx[i] += (hx[i]-pxs[i]) * 0.012f;   /* spring home */
                pvy[i] += (hy[i]-pys[i]) * 0.012f;
                pvx[i] *= 0.93f; pvy[i] *= 0.93f;
                pxs[i] += pvx[i]; pys[i] += pvy[i];
                if (pxs[i] < 0) pxs[i] = 0;
                if (pys[i] < 0) pys[i] = 0;
                if (pxs[i] > img.cols-1) pxs[i] = img.cols-1;
                if (pys[i] > img.rows-1) pys[i] = img.rows-1;
                float sp = std::sqrt(pvx[i]*pvx[i]+pvy[i]*pvy[i]);
                int br = std::min(255, 70 + (int)(sp*38));
                cv::circle(img, {(int)pxs[i],(int)pys[i]},
                           sp > 1.6f ? 3 : 2,
                           cv::Scalar(br, (int)(br*0.75f), 60),
                           cv::FILLED, cv::LINE_AA);
                if (sp > 2.2f)                        /* motion streak */
                    cv::line(img, {(int)pxs[i],(int)pys[i]},
                             {(int)(pxs[i]-pvx[i]*1.8f),(int)(pys[i]-pvy[i]*1.8f)},
                             cv::Scalar(br,(int)(br*0.6f),40), 1, cv::LINE_AA);
            }
            g_seg.mask.copyTo(prev);
            return;
        }
        if (mode == 108) {
            /* KEEPY-UPPY: ONE ball, real physics, and a score. Head it,
             * hand it, knee it — every touch counts, dropping it resets
             * the streak. The matte is the collider; its gradient gives
             * the surface normal, and your body's MOTION adds impulse,
             * so a fast hand smacks the ball harder than a slow one.
             * fx = gravity (0.2 floaty ... 3 brutal). */
            static float px = -1, py = 0, vx = 0, vy = 0;
            static int score = 0, best = 0, cooldown = 0, dropped = 0;
            static cv::Mat prev_mask;
            unsigned tk = g_seg.counter;
            float R = std::max(14.0f, img.cols * 0.045f);
            float G = 0.42f * fx;

            if (px < 0) {                       /* serve */
                px = img.cols * 0.5f;
                py = R + 4;
                vx = (uhash(tk,1,3) - 0.5f) * 3.0f;
                vy = 0;
            }
            /* motion field: how fast is your body moving here? */
            cv::Mat cur;
            g_seg.mask.copyTo(cur);
            if (prev_mask.empty() || prev_mask.size() != cur.size())
                cur.copyTo(prev_mask);

            vy += G;
            px += vx;
            py += vy;

            /* walls + ceiling bounce, floor = drop */
            if (px < R)            { px = R;            vx = std::fabs(vx)*0.86f; }
            if (px > img.cols - R) { px = img.cols - R; vx = -std::fabs(vx)*0.86f; }
            if (py < R)            { py = R;            vy = std::fabs(vy)*0.7f; }
            if (py > img.rows + R*2) {          /* dropped it */
                if (score > best) best = score;
                score = 0;
                dropped = 25;
                px = -1;                        /* re-serve next frame */
            }

            if (px > 0 && cooldown == 0) {
                /* ---- contact + normal from a RING around the ball ----
                 * Sampling the mask at the ball's centre is useless: on
                 * contact the centre is often INSIDE the body, where the
                 * gradient vanishes. Instead sample the circle around
                 * the ball; the weighted direction of the solid samples
                 * points INTO your body, so its negation is the surface
                 * normal — and it angles correctly off a shoulder, a
                 * forearm, the side of your head. */
                const int NS = 16;
                float sumx = 0, sumy = 0, hit = 0;
                for (int i2 = 0; i2 < NS; i2++) {
                    float a2 = 6.2832f * i2 / NS;
                    float dx2 = std::cos(a2), dy2 = std::sin(a2);
                    int sx2 = (int)(px + dx2 * R * 0.95f);
                    int sy2 = (int)(py + dy2 * R * 0.95f);
                    if (sx2 < 0 || sy2 < 0 || sx2 >= img.cols || sy2 >= img.rows)
                        continue;
                    float m2 = g_seg.mask.at<float>(sy2, sx2);
                    if (m2 > 0.5f) {
                        sumx += dx2 * m2;
                        sumy += dy2 * m2;
                        hit = std::max(hit, m2);
                    }
                }
                if (hit > 0.5f) {
                    float len = std::sqrt(sumx*sumx + sumy*sumy);
                    float nx2, ny2;
                    if (len > 1e-3f) {           /* away from the body mass */
                        nx2 = -sumx/len; ny2 = -sumy/len;
                    } else {                     /* fully engulfed: eject up */
                        nx2 = 0; ny2 = -1;
                    }
                    /* ---- YOUR push direction, from where the matte
                     * APPEARED vs where it VANISHED around the ball.
                     * A hand sweeping right reveals mask on its leading
                     * side and clears it behind: the vector between the
                     * two centroids is the push. Immune to the aperture
                     * problem that defeats a local gradient solve on a
                     * straight edge. */
                    float pcx=0, pcy=0, pw=0, ncx=0, ncy=0, nw=0;
                    int W = (int)(R * 3.2f);
                    for (int yy = -W; yy <= W; yy += 2)
                        for (int xx = -W; xx <= W; xx += 2) {
                            int sx2 = (int)px + xx, sy2 = (int)py + yy;
                            if (sx2 < 0 || sy2 < 0 ||
                                sx2 >= img.cols || sy2 >= img.rows) continue;
                            float d = g_seg.mask.at<float>(sy2, sx2)
                                    - prev_mask.at<float>(sy2, sx2);
                            if (d > 0.20f)      { pcx += sx2*d; pcy += sy2*d; pw += d; }
                            else if (d < -0.20f){ ncx -= sx2*d; ncy -= sy2*d; nw -= d; }
                        }
                    float bu = 0, bv = 0;
                    if (pw > 1.0f && nw > 1.0f) {       /* body moved here */
                        float dx3 = pcx/pw - ncx/nw;    /* vanished -> appeared */
                        float dy3 = pcy/pw - ncy/nw;
                        float dl2 = std::sqrt(dx3*dx3 + dy3*dy3);
                        if (dl2 > 1e-3f) {
                            /* direction from the centroids, strength from
                             * how much matte moved (a fast hand shifts a
                             * lot of mask) */
                            float punch = std::min(11.0f,
                                (pw + nw) / std::max(1.0f, R*R*0.10f) * 2.4f);
                            bu = dx3/dl2 * punch;
                            bv = dy3/dl2 * punch;
                        }
                    }
                    /* ---- bounce: reflect about the ring normal, keep
                     * the tangential component (that's what lets you
                     * glance it sideways), then add your body's own
                     * motion as impulse. */
                    float dot = vx*nx2 + vy*ny2;
                    if (dot < 0) {                   /* moving into you */
                        float rvx = vx - 2*dot*nx2;  /* mirrored velocity */
                        float rvy = vy - 2*dot*ny2;
                        float tvx = rvx - (rvx*nx2 + rvy*ny2)*nx2;  /* tangent */
                        float tvy = rvy - (rvx*nx2 + rvy*ny2)*ny2;
                        float nvn = (rvx*nx2 + rvy*ny2) * 0.92f;    /* normal  */
                        vx = tvx*0.94f + nx2*nvn;
                        vy = tvy*0.94f + ny2*nvn;
                    }
                    /* pop off the surface + transfer your motion */
                    vx += nx2 * 2.6f + bu * 1.15f;
                    vy += ny2 * 2.6f + bv * 1.15f;
                    /* a slow/still body still gives a gentle lift so
                     * the ball doesn't die on your head */
                    if (vy > -1.2f && ny2 < -0.2f) vy = -1.2f;
                    /* a contact must never spike the ball hard downward —
                     * that just ends the rally with no chance to react */
                    if (vy > 5.0f) vy = 5.0f;
                    float sp = std::sqrt(vx*vx + vy*vy);
                    if (sp > 28.0f) { vx *= 28.0f/sp; vy *= 28.0f/sp; }
                    px += nx2 * (R*0.35f);           /* unstick */
                    py += ny2 * (R*0.35f);
                    score++;
                    cooldown = 5;                    /* no double-count */
                }
            }
            if (cooldown > 0) cooldown--;
            if (dropped > 0) dropped--;
            cur.copyTo(prev_mask);

            /* ---- draw: shadow, ball with shading, HUD ---- */
            if (px > 0) {
                int gy3 = img.rows - 8;
                float shrink = std::max(0.25f, 1.0f - (gy3-py)/(float)img.rows);
                cv::ellipse(img, cv::Point((int)px, gy3),
                            cv::Size((int)(R*shrink), (int)(R*0.28f*shrink)),
                            0, 0, 360, cv::Scalar(30,30,30), cv::FILLED,
                            cv::LINE_AA);
                cv::circle(img, cv::Point((int)px,(int)py), (int)R+3,
                           cv::Scalar(20,25,30), cv::FILLED, cv::LINE_AA);
                cv::circle(img, cv::Point((int)px,(int)py), (int)R,
                           cooldown > 3 ? cv::Scalar(90,240,255)
                                        : cv::Scalar(70,180,255),
                           cv::FILLED, cv::LINE_AA);
                cv::circle(img, cv::Point((int)(px-R*0.3f),(int)(py-R*0.32f)),
                           (int)(R*0.32f), cv::Scalar(230,250,255),
                           cv::FILLED, cv::LINE_AA);
                cv::circle(img, cv::Point((int)px,(int)py), (int)R,
                           cv::Scalar(255,255,255), 2, cv::LINE_AA);
            }
            char hud[64];
            snprintf(hud, sizeof hud, "%d", score);
            cv::putText(img, hud, {22, 54}, cv::FONT_HERSHEY_DUPLEX, 1.7,
                        cv::Scalar(20,20,20), 5, cv::LINE_AA);
            cv::putText(img, hud, {22, 54}, cv::FONT_HERSHEY_DUPLEX, 1.7,
                        score > 0 ? cv::Scalar(90,240,255)
                                  : cv::Scalar(200,200,200), 2, cv::LINE_AA);
            snprintf(hud, sizeof hud, "BEST %d", best);
            cv::putText(img, hud, {22, 82}, cv::FONT_HERSHEY_PLAIN, 1.2,
                        cv::Scalar(180,180,190), 1, cv::LINE_AA);
            if (dropped > 0)
                cv::putText(img, "DROPPED!", {img.cols/2 - 90, img.rows/2},
                            cv::FONT_HERSHEY_DUPLEX, 1.4,
                            cv::Scalar(80,80,255), 2, cv::LINE_AA);
            return;
        }
        if (mode == 98) {
            /* SEARCHLIGHT: a prison-yard searchlight wanders the dark
             * room, finds you, and locks on. fx = beam size. */
            unsigned tk = g_seg.counter;
            static float sx3 = -1, sy3 = -1;
            int cx2 = 0, cy2 = 0, n2 = 0;
            for (int y = 0; y < img.rows; y += 8) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 8)
                    if (mm[x] > 0.5f) { cx2 += x; cy2 += y; n2++; }
            }
            float tx2, ty2;
            if (n2 > 8) {                       /* found you: ease toward */
                tx2 = (float)cx2/n2; ty2 = (float)cy2/n2 - img.rows*0.08f;
            } else {                            /* wander */
                tx2 = img.cols*(0.5f+0.4f*std::sin(tk*0.021f));
                ty2 = img.rows*(0.45f+0.3f*std::sin(tk*0.013f+2));
            }
            if (sx3 < 0) { sx3 = tx2; sy3 = ty2; }
            sx3 += (tx2-sx3)*0.06f; sy3 += (ty2-sy3)*0.06f;
            float R = std::min(img.cols,img.rows)*0.30f*fx;
            for (int y = 0; y < img.rows; y++) {
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float d = std::sqrt((x-sx3)*(x-sx3)+(y-sy3)*(y-sy3)*1.4f);
                    float k = d < R ? 1.0f : std::max(0.10f, 1.0f-(d-R)/(R*0.5f));
                    if (k > 1) k = 1;
                    float lift = d < R ? 1.12f : 1.0f;
                    ip[x][0]=clamp_int_u8((int)(ip[x][0]*k*lift));
                    ip[x][1]=clamp_int_u8((int)(ip[x][1]*k*lift));
                    ip[x][2]=clamp_int_u8((int)(ip[x][2]*k*lift*1.02f));
                }
            }
            return;
        }
        if (mode == 99) {
            /* PIXEL SORT: glitch-art — bright runs inside your body get
             * sorted along each row. fx = sort threshold. */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                int x = 0;
                while (x < img.cols) {
                    if (mm[x] < 0.5f) { x++; continue; }
                    int x0 = x;
                    while (x < img.cols && mm[x] >= 0.5f) x++;
                    int len = x - x0;
                    if (len < 8) continue;
                    /* insertion sort by luma (runs are short) */
                    for (int i2 = x0+1; i2 < x; i2++) {
                        cv::Vec3b v = ip[i2];
                        int lv = v[0]+2*v[1]+v[2];
                        int j2 = i2-1;
                        while (j2 >= x0 &&
                               (ip[j2][0]+2*ip[j2][1]+ip[j2][2]) > lv) {
                            ip[j2+1] = ip[j2]; j2--;
                        }
                        ip[j2+1] = v;
                    }
                    (void)fx;
                }
            }
            return;
        }
        if (mode == 100) {
            /* LASER GRID: red security lasers cross the room — where
             * they hit your body they glow hot and break. fx = density. */
            unsigned tk = g_seg.counter;
            int gap = std::max(28, (int)(60/fx));
            int drift = (int)(tk*0.4f) % gap;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                bool row = ((y+drift) % gap) < 2;
                for (int x = 0; x < img.cols; x++) {
                    bool col = ((x+drift) % gap) < 2;
                    if (!row && !col) continue;
                    float a = mm[x];
                    if (a > 0.5f) {             /* beam hits you: hot spot */
                        ip[x][2]=255; ip[x][1]=clamp_int_u8(ip[x][1]+150);
                        ip[x][0]=clamp_int_u8(ip[x][0]+90);
                    } else {                    /* free beam */
                        ip[x][2]=clamp_int_u8(ip[x][2]+170);
                        ip[x][1]=(uchar)(ip[x][1]*0.5f);
                        ip[x][0]=(uchar)(ip[x][0]*0.5f);
                    }
                }
            }
            return;
        }
        if (mode == 101) {
            /* OIL SLICK: iridescent film over you — hue follows your
             * surface gradient like light on spilled petrol. fx = swirl. */
            unsigned tk = g_seg.counter;
            cv::Mat gray, gx2, gy2;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::Sobel(gray, gx2, CV_32F, 1, 0, 5);
            cv::Sobel(gray, gy2, CV_32F, 0, 1, 5);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *ex = gx2.ptr<float>(y);
                const float *ey = gy2.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float ang = std::atan2(ey[x], ex[x]) * fx + tk*0.03f;
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    float b2 = (std::sin(ang)*.5f+.5f)*200*lum+30;
                    float g2 = (std::sin(ang+2.09f)*.5f+.5f)*200*lum+30;
                    float r2 = (std::sin(ang+4.19f)*.5f+.5f)*200*lum+30;
                    ip[x][0]=(uchar)(ip[x][0]+(b2-ip[x][0])*a*0.85f);
                    ip[x][1]=(uchar)(ip[x][1]+(g2-ip[x][1])*a*0.85f);
                    ip[x][2]=(uchar)(ip[x][2]+(r2-ip[x][2])*a*0.85f);
                }
            }
            return;
        }
        if (mode == 102) {
            /* NEON SIGN: lights off — your outline becomes a buzzing
             * neon tube that flickers like a cheap bar sign. fx = glow. */
            unsigned tk = g_seg.counter;
            float buzz = uhash(0,0,tk) < 0.06f ? 0.25f
                       : 0.85f + 0.15f*uhash(1,0,tk);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    float edge2 = (a > 0.06f && a < 0.94f)
                        ? 1.0f - std::fabs(a-0.5f)*2.0f : 0.0f;
                    float k = edge2*edge2 * buzz * fx;
                    ip[x][0]=clamp_int_u8((int)(ip[x][0]*0.16f + k*255));
                    ip[x][1]=clamp_int_u8((int)(ip[x][1]*0.14f + k*90));
                    ip[x][2]=clamp_int_u8((int)(ip[x][2]*0.18f + k*230));
                }
            }
            return;
        }
        if (mode == 103) {
            /* RAIN SHADOW: it rains in your room — everywhere except
             * on you. You repel weather now. fx = rain amount. */
            unsigned tk = g_seg.counter;
            int drops = (int)(220 * fx);
            for (int d = 0; d < drops; d++) {
                float n1 = uhash(d, 0, 0), n2 = uhash(d, 1, 0);
                int x = (int)(n1 * img.cols);
                int len = 10 + (int)(n2*14);
                int y = (int)(uhash(d,2,0)*img.rows + tk*(9+n2*7)) % (img.rows-len);
                if (g_seg.mask.at<float>(std::min(img.rows-1,y+len/2), x) > 0.35f)
                    continue;                   /* you are dry */
                for (int k2 = 0; k2 < len; k2++) {
                    cv::Vec3b *p2 = img.ptr<cv::Vec3b>(y+k2)+x;
                    p2[0][0]=clamp_int_u8(p2[0][0]+70);
                    p2[0][1]=clamp_int_u8(p2[0][1]+60);
                    p2[0][2]=clamp_int_u8(p2[0][2]+50);
                }
            }
            /* gloom outside, normal light on you */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float k = (1.0f-mm[x])*0.25f;
                    ip[x][0]=(uchar)(ip[x][0]*(1-k*0.6f));
                    ip[x][1]=(uchar)(ip[x][1]*(1-k));
                    ip[x][2]=(uchar)(ip[x][2]*(1-k));
                }
            }
            return;
        }
        if (mode == 104) {
            /* MIRROR SHATTER: your body seen through a broken mirror —
             * jittered facets each reflecting a shifted piece of you.
             * fx = shard size. */
            unsigned tk = g_seg.counter; (void)tk;
            int C = std::max(24, (int)(46*fx));
            cv::Mat keep = img.clone();
            for (int y0 = 0; y0 < img.rows; y0 += C)
                for (int x0 = 0; x0 < img.cols; x0 += C) {
                    int cw = std::min(C, img.cols-x0), ch = std::min(C, img.rows-y0);
                    if (g_seg.mask.at<float>(y0+ch/2, x0+cw/2) < 0.3f) continue;
                    int ox = (int)((uhash(x0,y0,7)-0.5f)*C*0.6f);
                    int oy = (int)((uhash(y0,x0,9)-0.5f)*C*0.6f);
                    for (int y = 0; y < ch; y++)
                        for (int x = 0; x < cw; x++) {
                            int sx2 = std::max(0,std::min(img.cols-1,x0+x+ox));
                            int sy2 = std::max(0,std::min(img.rows-1,y0+y+oy));
                            img.at<cv::Vec3b>(y0+y,x0+x)=keep.at<cv::Vec3b>(sy2,sx2);
                        }
                    cv::line(img,{x0,y0},{x0+cw-1,y0},cv::Scalar(210,210,215),1);
                    cv::line(img,{x0,y0},{x0,y0+ch-1},cv::Scalar(210,210,215),1);
                }
            return;
        }
        if (mode == 105) {
            /* MAGNET FIELD: field lines arc away from your outline —
             * you are the magnet. fx = field reach. */
            unsigned tk = g_seg.counter;
            cv::Mat m8, d1;
            g_seg.mask.convertTo(m8, CV_8U, 255.0);
            for (int ring = 1; ring <= 4; ring++) {
                cv::dilate(m8, d1, cv::Mat(), cv::Point(-1,-1),
                           (int)(ring*7*fx));
                float ph = std::sin(tk*0.06f + ring)*0.5f+0.5f;
                int br = (int)((90 - ring*16) * (0.6f+0.4f*ph));
                for (int y = 1; y < img.rows-1; y += 2) {
                    const uchar *da = d1.ptr<uchar>(y);
                    cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                    for (int x = 1; x < img.cols-1; x++) {
                        if (da[x] > 128 && (d1.at<uchar>(y,x-1) < 128 ||
                                            d1.at<uchar>(y-1,x) < 128)) {
                            ip[x][0]=clamp_int_u8(ip[x][0]+br*2);
                            ip[x][1]=clamp_int_u8(ip[x][1]+br);
                            ip[x][2]=clamp_int_u8(ip[x][2]+br/2);
                        }
                    }
                }
            }
            return;
        }
        if (mode == 106) {
            /* PRESENCE HEAT: everywhere you've been slowly glows warm
             * and fades — your movement paints heat into the room.
             * fx = memory length. */
            static cv::Mat heat;
            if (heat.empty() || heat.size() != img.size())
                heat = cv::Mat::zeros(img.size(), CV_32F);
            float decay = 1.0f - 0.02f/fx;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                float *hp = heat.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    hp[x] = std::max(hp[x]*decay, mm[x]);
                    float k = hp[x] * (1.0f - mm[x]);   /* trail only */
                    if (k < 0.03f) continue;
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(k*130));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(k*45));
                    ip[x][0]=(uchar)(ip[x][0]*(1-k*0.35f));
                }
            }
            return;
        }
        if (mode == 107) {
            /* GRAND FINALE: golden rim light, rising sparks, warm
             * vignette — the 100-filter victory lap. fx = celebration. */
            unsigned tk = g_seg.counter;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                float vy = std::fabs(y-img.rows*0.5f)/(img.rows*0.5f);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    float vx2 = std::fabs(x-img.cols*0.5f)/(img.cols*0.5f);
                    float vig = 1.0f - (vx2*vx2+vy*vy)*0.22f;
                    float rim = (a > 0.05f && a < 0.55f)
                        ? (0.55f-a)*2.0f : 0.0f;
                    ip[x][0]=clamp_int_u8((int)(ip[x][0]*vig + rim*40*fx));
                    ip[x][1]=clamp_int_u8((int)(ip[x][1]*vig + rim*150*fx));
                    ip[x][2]=clamp_int_u8((int)(ip[x][2]*vig*1.04f + rim*230*fx));
                }
            }
            for (int i2 = 0; i2 < 46; i2++) {       /* rising gold sparks */
                float n = uhash(i2, 3, 0);
                int x = (int)(n*img.cols);
                int y = (int)(img.rows - ((tk*(2+(int)(n*3)) + i2*37) %
                              (img.rows+40)));
                if (y < 0 || y >= img.rows) continue;
                float tw = 0.5f + 0.5f*uhash(i2, 4, tk/3);
                cv::circle(img, {x, y}, 1 + (i2%2),
                           cv::Scalar(60*tw, 190*tw, 250*tw), cv::FILLED);
            }
            return;
        }
        if (mode == 88) {
            /* VAPORWAVE: pink-cyan sunset duotone on you + drifting
             * horizontal displacement lines. fx = drift. */
            unsigned tk = g_seg.counter;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                int off = ((y/3 + tk/4) % 9 == 0) ? (int)(6*fx) : 0;
                for (int x = 0; x < img.cols; x++) {
                    int sx2 = std::min(img.cols-1, std::max(0, x - off));
                    float a = mm[sx2]; if (a < 0.05f) continue;
                    cv::Vec3b v = img.at<cv::Vec3b>(y, sx2);
                    float lum = (v[0]*0.114f+v[1]*0.587f+v[2]*0.299f)/255.f;
                    cv::Vec3b c((uchar)(200+lum*55),(uchar)(60+lum*120),
                                (uchar)(140+lum*115));
                    ip[x][0]=(uchar)(ip[x][0]+(c[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(c[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(c[2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 89) {
            /* EMBOSS RELIEF: you pressed into gray metal — self-offset
             * difference relief, person only. fx = depth. */
            cv::Mat keep = img.clone();
            int o = std::max(1,(int)(2*fx));
            for (int y = o; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *up = keep.ptr<cv::Vec3b>(y-o);
                const cv::Vec3b *cu = keep.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = o; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    int d = (cu[x][1] - up[x-o][1]) * 2 + 128;
                    uchar u2 = clamp_int_u8(d);
                    ip[x][0]=(uchar)(ip[x][0]+(u2-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(u2-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(u2-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 90) {
            /* PIXEL DRIP: your bottom edge melts — columns of you
             * stretch downward in dripping streaks. fx = drip length. */
            unsigned tk = g_seg.counter;
            for (int x = 0; x < img.cols; x++) {
                int by2 = -1;
                for (int y = img.rows-1; y >= 0; y--)
                    if (g_seg.mask.at<float>(y,x) > 0.5f) { by2 = y; break; }
                if (by2 < 0 || by2 >= img.rows-2) continue;
                float n = uhash(x/4, 0, tk/16);
                int len = (int)(n*n * img.rows * 0.18f * fx);
                cv::Vec3b src = img.at<cv::Vec3b>(by2, x);
                for (int y = by2+1; y < std::min(img.rows, by2+len); y++) {
                    float f2 = 1.0f - (float)(y-by2)/len;
                    cv::Vec3b *p2 = img.ptr<cv::Vec3b>(y)+x;
                    p2[0][0]=(uchar)(p2[0][0]+(src[0]-p2[0][0])*f2);
                    p2[0][1]=(uchar)(p2[0][1]+(src[1]-p2[0][1])*f2);
                    p2[0][2]=(uchar)(p2[0][2]+(src[2]-p2[0][2])*f2);
                }
            }
            return;
        }
        if (mode == 91) {
            /* ANAGLYPH 3D: red/cyan offset ghosts of you — grab the
             * cardboard glasses. fx = eye separation. */
            cv::Mat keep = img.clone();
            int o = std::max(2, (int)(7*fx));
            for (int y = 0; y < img.rows; y++) {
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    int xl = std::max(0, x-o), xr = std::min(img.cols-1, x+o);
                    float al = g_seg.mask.at<float>(y,xl);
                    float ar = g_seg.mask.at<float>(y,xr);
                    if (al < 0.05f && ar < 0.05f) continue;
                    if (al > 0.05f)
                        ip[x][2]=(uchar)(ip[x][2]+(keep.at<cv::Vec3b>(y,xl)[2]-ip[x][2])*al);
                    if (ar > 0.05f) {
                        ip[x][0]=(uchar)(ip[x][0]+(keep.at<cv::Vec3b>(y,xr)[0]-ip[x][0])*ar);
                        ip[x][1]=(uchar)(ip[x][1]+(keep.at<cv::Vec3b>(y,xr)[1]-ip[x][1])*ar);
                    }
                }
            }
            return;
        }
        if (mode == 92) {
            /* CHALK OUTLINE: the room goes crime-scene gray and a rough
             * white chalk line traces where you are. fx = roughness. */
            unsigned tk = g_seg.counter; (void)tk;
            cv::Mat m8, d1, e1;
            g_seg.mask.convertTo(m8, CV_8U, 255.0);
            cv::dilate(m8, d1, cv::Mat(), cv::Point(-1,-1), 3);
            cv::erode (m8, e1, cv::Mat(), cv::Point(-1,-1), 1);
            for (int y = 0; y < img.rows; y++) {
                const uchar *da=d1.ptr<uchar>(y), *ea=e1.ptr<uchar>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float lum = ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f;
                    if (da[x] > 128 && ea[x] < 128) {
                        if (uhash(x/2, y/2, 0) < 0.75f/fx + 0.2f) {
                            ip[x] = cv::Vec3b(235, 235, 238);
                            continue;
                        }
                    }
                    ip[x][0]=(uchar)(lum*0.7f);
                    ip[x][1]=(uchar)(lum*0.7f);
                    ip[x][2]=(uchar)(lum*0.72f);
                }
            }
            return;
        }
        if (mode == 93) {
            /* SCAN BAR: a bright vertical bar sweeps across your body,
             * over-exposing the slice it touches. fx = speed. */
            unsigned tk = g_seg.counter;
            int bar = (int)(tk * 4 * fx) % (img.cols + 160) - 80;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = std::max(0,bar-40); x < std::min(img.cols,bar+40); x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float k = (1.0f - std::fabs(x-bar)/40.0f) * a;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(k*200));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(k*220));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(k*235));
                }
            }
            return;
        }
        if (mode == 94) {
            /* TEST PATTERN: SMPTE color bars fill your silhouette —
             * please stand by. */
            static const int BARS[7][3] = {{180,180,180},{0,180,180},{180,180,0},
                {0,180,0},{180,0,180},{0,0,180},{180,0,0}};
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    const int *b2 = BARS[(x * 7 / img.cols) % 7];
                    ip[x][0]=(uchar)(ip[x][0]+(b2[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(b2[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(b2[2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 95) {
            /* ONE-BIT DITHER: you in pure black-and-white Bayer
             * dithering — 1985 Macintosh you. fx = brightness bias. */
            static const int BAYER[4][4] = {{0,8,2,10},{12,4,14,6},
                                            {3,11,1,9},{15,7,13,5}};
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)
                              / 255.f * fx;
                    uchar v = lum * 16 > BAYER[y&3][x&3] + 0.5f ? 245 : 12;
                    ip[x][0]=(uchar)(ip[x][0]+(v-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(v-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(v-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 96) {
            /* FROST EDGE: ice crystals grow along your outline — white
             * crystalline spikes, cold blue tint creeping inward.
             * fx = frost reach. */
            unsigned tk = g_seg.counter; (void)tk;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a < 0.04f || a > 0.97f) continue;
                    float edge2 = 1.0f - std::fabs(a - 0.5f) * 2.0f;
                    float n = uhash(x/2, y/2, 0);
                    float k = edge2 * (n > 0.45f ? 1.0f : 0.25f) * std::min(1.5f, fx);
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(k*230));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(k*200));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(k*160));
                }
            }
            return;
        }
        if (mode == 97) {
            /* HOLLOW MAN: your interior vanishes into the learned room;
             * only a bright shell of edges remains. fx = shell glow. */
            cv::Mat gray, gx2, gy2;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::Sobel(gray, gx2, CV_32F, 1, 0, 3);
            cv::Sobel(gray, gy2, CV_32F, 0, 1, 3);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *ex = gx2.ptr<float>(y);
                const float *ey = gy2.ptr<float>(y);
                const cv::Vec3b *bp = g_seg.bg_est.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float e = (std::fabs(ex[x])+std::fabs(ey[x])) / 900.0f * fx;
                    if (e > 1) e = 1;
                    float nb = bp[x][0]+(255-bp[x][0])*e;
                    float ng = bp[x][1]+(255-bp[x][1])*e;
                    float nr = bp[x][2]+(200-bp[x][2])*e;
                    ip[x][0]=(uchar)(ip[x][0]+(nb-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(ng-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(nr-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 78) {
            /* BLUEPRINT: you as a technical drawing — deep blue fill,
             * white construction edges and a drafting grid; the room
             * stays photographic. fx = line density. */
            cv::Mat gray, gx2, gy2;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::Sobel(gray, gx2, CV_32F, 1, 0, 3);
            cv::Sobel(gray, gy2, CV_32F, 0, 1, 3);
            float thr = 240.0f / fx;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *ex = gx2.ptr<float>(y);
                const float *ey = gy2.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float e = std::fabs(ex[x]) + std::fabs(ey[x]);
                    bool grid = (x % 24 == 0) || (y % 24 == 0);
                    cv::Vec3b c = e > thr ? cv::Vec3b(245,245,250)
                                : grid    ? cv::Vec3b(150,105,60)
                                          : cv::Vec3b(110,60,20);
                    ip[x][0]=(uchar)(ip[x][0]+(c[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(c[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(c[2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 79) {
            /* BRICK TILES: you built from beveled plastic tiles — lit
             * top-left edge, shadowed bottom-right. fx = tile size. */
            int C = std::max(8, (int)(14 * fx));
            cv::Mat keep = img.clone();
            for (int y0 = 0; y0 + C <= img.rows; y0 += C)
                for (int x0 = 0; x0 + C <= img.cols; x0 += C) {
                    if (g_seg.mask.at<float>(y0+C/2, x0+C/2) < 0.4f) continue;
                    cv::Vec3b c = keep.at<cv::Vec3b>(y0+C/2, x0+C/2);
                    c[0]=(c[0]/48)*48+24; c[1]=(c[1]/48)*48+24; c[2]=(c[2]/48)*48+24;
                    cv::rectangle(img, cv::Rect(x0, y0, C, C),
                                  cv::Scalar(c[0],c[1],c[2]), cv::FILLED);
                    cv::line(img, {x0,y0}, {x0+C-1,y0},
                             cv::Scalar(std::min(255,c[0]+55),std::min(255,c[1]+55),
                                        std::min(255,c[2]+55)), 2);
                    cv::line(img, {x0,y0}, {x0,y0+C-1},
                             cv::Scalar(std::min(255,c[0]+55),std::min(255,c[1]+55),
                                        std::min(255,c[2]+55)), 2);
                    cv::line(img, {x0+1,y0+C-1}, {x0+C-1,y0+C-1},
                             cv::Scalar(c[0]*2/3,c[1]*2/3,c[2]*2/3), 2);
                    cv::circle(img, {x0+C/2, y0+C/2}, C/5,
                               cv::Scalar(std::min(255,c[0]+30),std::min(255,c[1]+30),
                                          std::min(255,c[2]+30)), 1);
                }
            return;
        }
        if (mode == 80) {
            /* SHADOW CALLER: you become a flat black silhouette with a
             * cold rim — witness-protection chic. fx = rim strength. */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.04f) continue;
                    float rim = (a < 0.55f) ? (0.55f-a)/0.51f : 0.0f;
                    int rb = (int)(rim * 150 * fx);
                    ip[x][0]=(uchar)(ip[x][0]*(1-a) + std::min(255, 18+rb)*a);
                    ip[x][1]=(uchar)(ip[x][1]*(1-a) + std::min(255, 14+rb/2)*a);
                    ip[x][2]=(uchar)(ip[x][2]*(1-a) + 12*a);
                }
            }
            return;
        }
        if (mode == 81) {
            /* GLOW IN THE DARK: lights out — you luminesce toxic green,
             * brighter where you were brighter. fx = glow. */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    if (a < 0.05f) {
                        ip[x][0]=(uchar)(ip[x][0]*0.12f);
                        ip[x][1]=(uchar)(ip[x][1]*0.14f);
                        ip[x][2]=(uchar)(ip[x][2]*0.12f);
                        continue;
                    }
                    float g2 = std::min(255.f, (40 + lum*215) * fx);
                    ip[x][0]=(uchar)(ip[x][0]+((g2*0.35f)-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(g2-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+((g2*0.25f)-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 82) {
            /* GRAPHITE: you sketched in pencil (classic dodge-blend:
             * luma over its own blur), room in color. fx = contrast. */
            cv::Mat gray, blur2;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, blur2, cv::Size(0,0), 5.0);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const uchar *gp = gray.ptr<uchar>(y);
                const uchar *bp = blur2.ptr<uchar>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float v = 255.0f * gp[x] / (256.0f - std::min(254, (int)bp[x]));
                    v = 255.0f - (255.0f - std::min(255.f, v)) * fx;
                    if (v < 0) v = 0;
                    uchar u2 = (uchar)v;
                    ip[x][0]=(uchar)(ip[x][0]+(u2-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(u2-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(u2-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 83) {
            /* POP POSTER: you in flat two-tone screen-print; the ink
             * color steps to a new one every couple of seconds.
             * fx = switch speed. */
            unsigned tk = g_seg.counter;
            static const int INK[5][3] = {
                {180,40,230},{40,180,255},{200,180,20},{60,220,90},{230,90,50}};
            int pi = (int)(tk * 0.01f * fx) % 5;
            const int *I2 = INK[pi];
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    cv::Vec3b c = lum > 0.52f
                        ? cv::Vec3b(240, 238, 232)
                        : cv::Vec3b((uchar)I2[0], (uchar)I2[1], (uchar)I2[2]);
                    ip[x][0]=(uchar)(ip[x][0]+(c[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(c[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(c[2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 84) {
            /* RISO OUTLINES: three offset colored contours around your
             * silhouette — misregistered print, on purpose. fx = offset. */
            cv::Mat m8, d1;
            g_seg.mask.convertTo(m8, CV_8U, 255.0);
            cv::dilate(m8, d1, cv::Mat(), cv::Point(-1,-1), 2);
            int off = std::max(2, (int)(5 * fx));
            static const int OC[3][3] = {{60,60,235},{235,180,40},{200,80,200}};
            for (int y = 0; y < img.rows; y++) {
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    for (int k2 = 0; k2 < 3; k2++) {
                        int sx2 = x + (k2-1)*off, sy2 = y + (k2==2?off:(k2==0?-off:0));
                        if (sx2<1||sy2<1||sx2>=img.cols-1||sy2>=img.rows-1) continue;
                        uchar c2 = d1.at<uchar>(sy2, sx2);
                        uchar e2 = m8.at<uchar>(sy2, sx2);
                        if (c2 > 128 && e2 < 128) {
                            ip[x][0]=clamp_int_u8(ip[x][0]/2+OC[k2][0]/2);
                            ip[x][1]=clamp_int_u8(ip[x][1]/2+OC[k2][1]/2);
                            ip[x][2]=clamp_int_u8(ip[x][2]/2+OC[k2][2]/2);
                        }
                    }
                }
            }
            return;
        }
        if (mode == 85) {
            /* HEAT HAZE: you shimmer like air over asphalt — small
             * animated refraction inside your silhouette only.
             * fx = shimmer. */
            unsigned tk = g_seg.counter;
            cv::Mat keep = img.clone();
            float A2 = 3.5f * fx;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    int sx2 = x + (int)(std::sin(y*0.16f + tk*0.22f) * A2 * a);
                    int sy2 = y + (int)(std::sin(x*0.13f + tk*0.17f) * A2 * 0.6f * a);
                    sx2 = std::max(0, std::min(img.cols-1, sx2));
                    sy2 = std::max(0, std::min(img.rows-1, sy2));
                    ip[x] = keep.at<cv::Vec3b>(sy2, sx2);
                }
            }
            return;
        }
        if (mode == 86) {
            /* SECURITY CAM: grainy green-mono CCTV with a tracking box
             * locked on you, timestamp and REC dot. fx = grain. */
            unsigned tk = g_seg.counter;
            int minx=img.cols, miny=img.rows, maxx=0, maxy=0;
            for (int y = 0; y < img.rows; y += 6) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 6)
                    if (mm[x] > 0.4f) {
                        if (x<minx)minx=x;
                        if (x>maxx)maxx=x;
                        if (y<miny)miny=y;
                        if (y>maxy)maxy=y;
                    }
            }
            for (int y = 0; y < img.rows; y++) {
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float lum = ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f;
                    int gr = (int)((uhash(x,y,tk)-0.5f)*26*fx);
                    ip[x][0]=clamp_int_u8((int)(lum*0.55f)+gr);
                    ip[x][1]=clamp_int_u8((int)(lum*0.95f)+gr+10);
                    ip[x][2]=clamp_int_u8((int)(lum*0.55f)+gr);
                }
            }
            if (maxx > minx + 20) {
                cv::rectangle(img, {minx-8,miny-8}, {maxx+8,maxy+8},
                              cv::Scalar(120,255,120), 2);
                cv::putText(img, "SUBJECT 01", {minx-8, std::max(18,miny-14)},
                            cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(120,255,120), 1);
            }
            char ts[40];
            snprintf(ts, sizeof ts, "CAM 04  %02u:%02u:%02u",
                     (tk/108000)%24, (tk/1800)%60, (tk/30)%60);
            cv::putText(img, ts, {14, img.rows-16},
                        cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(140,255,140), 1);
            if ((tk/20)%2) {
                cv::circle(img, {img.cols-34, 26}, 7, cv::Scalar(60,60,255), cv::FILLED);
                cv::putText(img, "REC", {img.cols-92, 32},
                            cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(120,120,255), 1);
            }
            return;
        }
        if (mode == 87) {
            /* NEGATIVE YOU: your colors inverted, film-negative style,
             * with a faint hot rim; the room stays positive. */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float rim = (a < 0.5f) ? (0.5f-a)*2.0f*80*fx : 0.0f;
                    ip[x][0]=(uchar)(ip[x][0]+((255-ip[x][0])-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+((255-ip[x][1])-ip[x][1])*a);
                    ip[x][2]=clamp_int_u8((int)(ip[x][2]+((255-ip[x][2])-ip[x][2])*a+rim));
                }
            }
            return;
        }
        if (mode == 68) {
            /* ASCII YOU: your body rendered as green terminal
             * characters by brightness; the room stays real.
             * fx = character size. */
            int C = std::max(8, (int)(10 * fx));
            static const char *RAMP[] = {".",":","+","o","X","#","@"};
            cv::Mat keep = img.clone();
            for (int y0 = 0; y0 + C <= img.rows; y0 += C)
                for (int x0 = 0; x0 + C <= img.cols; x0 += C) {
                    if (g_seg.mask.at<float>(y0+C/2, x0+C/2) < 0.4f) continue;
                    cv::Vec3b c = keep.at<cv::Vec3b>(y0+C/2, x0+C/2);
                    float lum = (c[0]*0.114f+c[1]*0.587f+c[2]*0.299f)/255.f;
                    cv::rectangle(img, cv::Rect(x0, y0, C, C),
                                  cv::Scalar(8, 14, 8), cv::FILLED);
                    int ri = std::min(6, (int)(lum * 7));
                    cv::putText(img, RAMP[ri], {x0, y0 + C - 2},
                                cv::FONT_HERSHEY_PLAIN, C/10.0,
                                cv::Scalar(40, 100 + (int)(lum*155), 40), 1);
                }
            return;
        }
        if (mode == 69) {
            /* HALFTONE PRINT: you as a proper newspaper dot-screen —
             * dot size by darkness on a print grid; the room keeps its
             * color. fx = dot cell size. */
            int C = std::max(5, (int)(6 * fx));
            cv::Mat keep = img.clone();
            for (int y0 = 0; y0 + C <= img.rows; y0 += C)
                for (int x0 = 0; x0 + C <= img.cols; x0 += C) {
                    if (g_seg.mask.at<float>(y0+C/2, x0+C/2) < 0.4f) continue;
                    cv::Vec3b c = keep.at<cv::Vec3b>(y0+C/2, x0+C/2);
                    float dark = 1.0f - (c[0]*0.114f+c[1]*0.587f+c[2]*0.299f)/255.f;
                    cv::rectangle(img, cv::Rect(x0, y0, C, C),
                                  cv::Scalar(235, 232, 226), cv::FILLED);
                    int r2 = (int)(dark * C * 0.62f);
                    if (r2 > 0)
                        cv::circle(img, {x0 + C/2, y0 + C/2}, r2,
                                   cv::Scalar(30, 26, 24), cv::FILLED, cv::LINE_AA);
                }
            return;
        }
        if (mode == 70) {
            /* LIQUID METAL: chrome you — luminance remapped through a
             * banded metallic reflection curve. T-1000 energy.
             * fx = band frequency. */
            unsigned tk = g_seg.counter;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    float band = std::sin(lum * 9.0f * fx + tk * 0.02f);
                    float v = lum * 0.55f + 0.30f + band * 0.22f;
                    if (v < 0) v = 0;
                    if (v > 1) v = 1;
                    float nb = v*255*1.04f, ng = v*255*1.02f, nr = v*255*0.94f;
                    ip[x][0]=(uchar)(ip[x][0]+(std::min(255.f,nb)-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(std::min(255.f,ng)-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(std::min(255.f,nr)-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 71) {
            /* CONTOUR STRIPES: animated brightness-contour bands sweep
             * across your body — living topography of your own light.
             * fx = sweep speed. */
            unsigned tk = g_seg.counter;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    float band = std::sin(lum * 22.0f - tk * 0.12f * fx);
                    if (band > 0.55f) {
                        float hue = lum * 5.0f + tk * 0.02f;
                        ip[x][0]=clamp_int_u8(ip[x][0]+(int)(a*(std::sin(hue+4.19f)*.5f+.5f)*170));
                        ip[x][1]=clamp_int_u8(ip[x][1]+(int)(a*(std::sin(hue+2.09f)*.5f+.5f)*170));
                        ip[x][2]=clamp_int_u8(ip[x][2]+(int)(a*(std::sin(hue)*.5f+.5f)*170));
                    } else {
                        ip[x][0]=(uchar)(ip[x][0]*(1-a*0.35f));
                        ip[x][1]=(uchar)(ip[x][1]*(1-a*0.35f));
                        ip[x][2]=(uchar)(ip[x][2]*(1-a*0.35f));
                    }
                }
            }
            return;
        }
        if (mode == 72) {
            /* STATIC YOU: signal lost — your silhouette is pure TV
             * white-noise while the broadcast around you is fine.
             * fx = noise coarseness. */
            unsigned tk = g_seg.counter;
            int C = std::max(1, (int)fx);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    int n = (int)(uhash(x/C, y/C, tk) * 255);
                    ip[x][0]=(uchar)(ip[x][0]+(n-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(n-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(n-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 73) {
            /* GOLDEN STATUE: you, cast in gold — warm metal ramp with a
             * specular kick in the highlights. fx = shine. */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    float spec = lum > 0.72f ? (lum-0.72f)/0.28f : 0.0f;
                    spec = spec * spec * 0.9f * fx;
                    float nr = std::min(255.f, (lum*200 + 55) * (1+spec*0.5f));
                    float ng = std::min(255.f, (lum*160 + 30) * (1+spec*0.6f));
                    float nb = std::min(255.f, (lum*60  + 8)  * (1+spec*1.6f));
                    ip[x][0]=(uchar)(ip[x][0]+(nb-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(ng-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(nr-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 74) {
            /* COMIC INK: you, drawn — 4-level posterized color with
             * black ink where your edges are. fx = ink weight. */
            cv::Mat gray, gx2, gy2;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::Sobel(gray, gx2, CV_32F, 1, 0, 3);
            cv::Sobel(gray, gy2, CV_32F, 0, 1, 3);
            float thr = 260.0f / fx;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *ex = gx2.ptr<float>(y);
                const float *ey = gy2.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float e = std::fabs(ex[x]) + std::fabs(ey[x]);
                    if (e > thr) {
                        ip[x][0]=(uchar)(ip[x][0]*(1-a));
                        ip[x][1]=(uchar)(ip[x][1]*(1-a));
                        ip[x][2]=(uchar)(ip[x][2]*(1-a));
                    } else {
                        cv::Vec3b q((ip[x][0]/64)*64+40,
                                    (ip[x][1]/64)*64+40,
                                    (ip[x][2]/64)*64+40);
                        ip[x][0]=(uchar)(ip[x][0]+(q[0]-ip[x][0])*a);
                        ip[x][1]=(uchar)(ip[x][1]+(q[1]-ip[x][1])*a);
                        ip[x][2]=(uchar)(ip[x][2]+(q[2]-ip[x][2])*a);
                    }
                }
            }
            return;
        }
        if (mode == 75) {
            /* VENETIAN: you're seen through sliding blinds — stripes of
             * you alternate with the empty room behind. fx = blind size. */
            unsigned tk = g_seg.counter;
            int B = std::max(8, (int)(14 * fx));
            int off = (int)(tk * 0.8f) % (B * 2);
            for (int y = 0; y < img.rows; y++) {
                bool open2 = (((y + off) / B) & 1) == 0;
                if (open2) continue;
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *bp = g_seg.bg_est.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(bp[x][0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(bp[x][1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(bp[x][2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 76) {
            /* BURNING EDGE: your outline smolders like the rim of
             * burning paper — white-hot line, orange glow, charred
             * band inside. fx = flicker energy. */
            unsigned tk = g_seg.counter;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a < 0.04f || a > 0.97f) continue;
                    float fl = 0.6f + 0.4f * uhash(x/3, y/3, tk/2) * fx;
                    float edge2 = 1.0f - std::fabs(a - 0.5f) * 2.0f;  /* peak mid */
                    float hot = edge2 * edge2 * fl;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(hot*40));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(hot*130));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(hot*255));
                    if (a > 0.75f) {                    /* charred inner band */
                        float ch = (a - 0.75f) / 0.25f * 0.5f;
                        ip[x][0]=(uchar)(ip[x][0]*(1-ch));
                        ip[x][1]=(uchar)(ip[x][1]*(1-ch));
                        ip[x][2]=(uchar)(ip[x][2]*(1-ch));
                    }
                }
            }
            return;
        }
        if (mode == 77) {
            /* GLASS BODY: you become refractive glass — the learned
             * room shows THROUGH you, bent by your body's shape (mask
             * gradient as the surface normal) with a bright rim.
             * fx = refraction strength. */
            cv::Mat gx2, gy2;
            cv::Sobel(g_seg.mask, gx2, CV_32F, 1, 0, 3);
            cv::Sobel(g_seg.mask, gy2, CV_32F, 0, 1, 3);
            float R = 34.0f * fx;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *px2 = gx2.ptr<float>(y);
                const float *py2 = gy2.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    int sx2 = x + (int)(px2[x] * R);
                    int sy2 = y + (int)(py2[x] * R);
                    sx2 = std::max(0, std::min(img.cols-1, sx2));
                    sy2 = std::max(0, std::min(img.rows-1, sy2));
                    const cv::Vec3b &bg = g_seg.bg_est.at<cv::Vec3b>(sy2, sx2);
                    float rim = (a > 0.04f && a < 0.6f) ? (0.6f-a)/0.56f : 0.0f;
                    float nb = std::min(255.f, bg[0]*1.08f + rim*110);
                    float ng = std::min(255.f, bg[1]*1.08f + rim*110);
                    float nr = std::min(255.f, bg[2]*1.05f + rim*100);
                    ip[x][0]=(uchar)(ip[x][0]+(nb-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(ng-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(nr-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 66) {
            /* RETRO SPRITE: you become a chunky 16-color video-game
             * character; the room stays photoreal. fx = chunkiness. */
            int C = std::max(4, (int)(img.cols / 140 * fx * 2));
            for (int y0 = 0; y0 < img.rows; y0 += C)
                for (int x0 = 0; x0 < img.cols; x0 += C) {
                    if (g_seg.mask.at<float>(std::min(img.rows-1, y0+C/2),
                                             std::min(img.cols-1, x0+C/2)) < 0.4f)
                        continue;
                    cv::Vec3b c = img.at<cv::Vec3b>(
                        std::min(img.rows-1, y0+C/2), std::min(img.cols-1, x0+C/2));
                    c[0] = (c[0]/64)*64+32; c[1]=(c[1]/64)*64+32; c[2]=(c[2]/64)*64+32;
                    cv::rectangle(img, cv::Rect(x0, y0,
                        std::min(C, img.cols-x0), std::min(C, img.rows-y0)),
                        cv::Scalar(c[0],c[1],c[2]), cv::FILLED);
                }
            return;
        }
        if (mode == 67) {
            /* ACID YOU: your colors slowly cycle through the spectrum
             * while the room stays sober. fx = cycle speed. */
            unsigned tk = g_seg.counter;
            float ph = tk * 0.03f * fx;
            float c1 = std::cos(ph), s1 = std::sin(ph);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float lum = ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f;
                    float u = ip[x][2]-lum, v2 = ip[x][0]-lum;   /* chroma-ish */
                    float u2 = u*c1 - v2*s1, w2 = u*s1 + v2*c1;  /* rotate */
                    float nr = lum+u2, nb = lum+w2, ng = lum-(u2+w2)*0.4f;
                    ip[x][0]=(uchar)(ip[x][0]+(clamp_int_u8((int)nb)-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(clamp_int_u8((int)ng)-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(clamp_int_u8((int)nr)-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 56) {
            /* COLOR POP: the room falls to a cool, slightly dark
             * monochrome; you stay in full — slightly enriched — color.
             * The classic gallery look. fx = how muted the room goes. */
            float mute = std::min(1.0f, 0.75f * fx);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    float lum = ip[x][0]*0.114f + ip[x][1]*0.587f + ip[x][2]*0.299f;
                    if (a < 0.98f) {
                        float k = (1.0f - a) * mute;
                        float b2 = lum*0.92f + 8, g2 = lum*0.90f, r2 = lum*0.84f;
                        ip[x][0]=(uchar)(ip[x][0]+(b2-ip[x][0])*k);
                        ip[x][1]=(uchar)(ip[x][1]+(g2-ip[x][1])*k);
                        ip[x][2]=(uchar)(ip[x][2]+(r2-ip[x][2])*k);
                    }
                    if (a > 0.02f) {                     /* enrich you a touch */
                        float k = a * 0.18f;
                        ip[x][0]=clamp_int_u8((int)(lum + (ip[x][0]-lum)*(1+k)));
                        ip[x][1]=clamp_int_u8((int)(lum + (ip[x][1]-lum)*(1+k)));
                        ip[x][2]=clamp_int_u8((int)(lum + (ip[x][2]-lum)*(1+k)));
                    }
                }
            }
            return;
        }
        if (mode == 57) {
            /* CINEMA: 2.39:1 letterbox, teal-orange split-tone, gentle
             * S-curve contrast, fine grain. The movie look, done with
             * restraint. fx = grade strength. */
            unsigned tk = g_seg.counter;
            float G = 0.5f * std::min(1.6f, fx);
            for (int y = 0; y < img.rows; y++) {
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    for (int c2 = 0; c2 < 3; c2++) {      /* S-curve */
                        float v = ip[x][c2] / 255.0f;
                        float sc = v*v*(3.0f - 2.0f*v);
                        ip[x][c2] = (uchar)((v + (sc - v)*G*0.7f) * 255.0f);
                    }
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    float sh = (1.0f - lum);              /* shadows -> teal */
                    float hi = lum;                       /* highlights -> warm */
                    ip[x][0]=clamp_int_u8(ip[x][0] + (int)((sh*14 - hi*6)*G*2));
                    ip[x][1]=clamp_int_u8(ip[x][1] + (int)((sh*6)*G*2));
                    ip[x][2]=clamp_int_u8(ip[x][2] + (int)((hi*16 - sh*8)*G*2));
                    int gr = (int)((uhash(x, y, tk) - 0.5f) * 7 * G);
                    ip[x][0]=clamp_int_u8(ip[x][0]+gr);
                    ip[x][1]=clamp_int_u8(ip[x][1]+gr);
                    ip[x][2]=clamp_int_u8(ip[x][2]+gr);
                }
            }
            int barH = std::max(0, (int)((img.rows - img.cols / 2.39f) / 2));
            if (barH > 0 && barH < img.rows/2) {
                img.rowRange(0, barH) = cv::Scalar(0,0,0);
                img.rowRange(img.rows - barH, img.rows) = cv::Scalar(0,0,0);
            }
            return;
        }
        if (mode == 58) {
            /* BOKEH PRO: a real lens-style background — highlights
             * bloom into soft discs instead of dissolving into gaussian
             * mush; you stay tack sharp. fx = aperture. */
            int QW = img.cols/4, QH = img.rows/4;
            cv::Mat q, blur2, bright;
            cv::resize(img, q, cv::Size(QW, QH), 0, 0, cv::INTER_AREA);
            double sig = 2.5 + 2.5 * fx;
            cv::GaussianBlur(q, blur2, cv::Size(0,0), sig);
            /* highlight discs: threshold, disc-ish dilate, soften */
            bright = q.clone();
            for (int y = 0; y < QH; y++) {
                cv::Vec3b *bp = bright.ptr<cv::Vec3b>(y);
                for (int x = 0; x < QW; x++) {
                    int mx2 = std::max(bp[x][0], std::max(bp[x][1], bp[x][2]));
                    if (mx2 < 200) bp[x] = cv::Vec3b(0,0,0);
                }
            }
            int it = 2 + (int)fx;
            cv::dilate(bright, bright, cv::Mat(), cv::Point(-1,-1), it);
            cv::GaussianBlur(bright, bright, cv::Size(0,0), 1.5);
            cv::Mat bg;
            cv::addWeighted(blur2, 1.0, bright, 0.55, 0, bg);
            cv::Mat bgf;
            cv::resize(bg, bgf, img.size(), 0, 0, cv::INTER_LINEAR);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *bp = bgf.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float k = 1.0f - mm[x];
                    if (k < 0.02f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(bp[x][0]-ip[x][0])*k);
                    ip[x][1]=(uchar)(ip[x][1]+(bp[x][1]-ip[x][1])*k);
                    ip[x][2]=(uchar)(ip[x][2]+(bp[x][2]-ip[x][2])*k);
                }
            }
            return;
        }
        if (mode == 59) {
            /* STUDIO LIGHT: quiet three-point relighting — a warm key
             * lift on you, a cool rim from the upper-left, the room
             * eased down with a vignette. The "look better" button.
             * fx = light strength. */
            float K = 0.5f * fx;
            cv::Mat gx2, gy2;
            cv::Sobel(g_seg.mask, gx2, CV_32F, 1, 0, 3);
            cv::Sobel(g_seg.mask, gy2, CV_32F, 0, 1, 3);
            float vcx2 = img.cols/2.f, vcy2 = img.rows/2.f;
            float vmax = vcx2*vcx2 + vcy2*vcy2;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *px2 = gx2.ptr<float>(y);
                const float *py2 = gy2.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                float dy = y - vcy2;
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a > 0.02f) {                      /* warm key */
                        float k = a * 0.14f * K * 2;
                        ip[x][0]=clamp_int_u8((int)(ip[x][0]*(1+k*0.7f)));
                        ip[x][1]=clamp_int_u8((int)(ip[x][1]*(1+k*0.9f)));
                        ip[x][2]=clamp_int_u8((int)(ip[x][2]*(1+k*1.15f)));
                    }
                    if (a > 0.04f && a < 0.96f) {         /* cool rim, up-left */
                        float gl = std::sqrt(px2[x]*px2[x]+py2[x]*py2[x]);
                        if (gl > 0.02f) {
                            float rim = (px2[x]*0.7f + py2[x]*0.7f) / gl;
                            if (rim > 0) {
                                int r3 = (int)(rim*rim*130*K);
                                ip[x][0]=clamp_int_u8(ip[x][0]+r3);
                                ip[x][1]=clamp_int_u8(ip[x][1]+(int)(r3*0.9f));
                                ip[x][2]=clamp_int_u8(ip[x][2]+(int)(r3*0.75f));
                            }
                        }
                    }
                    if (a < 0.9f) {                       /* eased room + vignette */
                        float dx = x - vcx2;
                        float vg = (dx*dx + dy*dy) / vmax;
                        float dim = (1.0f - a) * (0.06f + vg*0.22f) * K * 2;
                        ip[x][0]=(uchar)(ip[x][0]*(1-dim));
                        ip[x][1]=(uchar)(ip[x][1]*(1-dim));
                        ip[x][2]=(uchar)(ip[x][2]*(1-dim));
                    }
                }
            }
            return;
        }
        if (mode == 60) {
            /* NIGHT BOOST: dark-room rescue — shadow lift + gentle
             * denoise on YOU, while the room keeps its evening mood.
             * fx = lift. */
            cv::Mat soft;
            cv::GaussianBlur(img, soft, cv::Size(0,0), 1.6);
            float L = std::min(1.5f, fx);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *sp = soft.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a < 0.03f) continue;
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    /* gamma lift, strongest in the shadows */
                    float lifted = std::pow(lum, 1.0f/(1.0f + 0.55f*L));
                    float gainv = lum > 0.004f ? lifted/lum : 1.0f;
                    if (gainv > 3.5f) gainv = 3.5f;
                    /* dark pixels borrow from the blur = cheap denoise */
                    float dn = (1.0f - lum) * 0.45f * a;
                    for (int c2 = 0; c2 < 3; c2++) {
                        float v = ip[x][c2] + (sp[x][c2]-ip[x][c2])*dn;
                        ip[x][c2] = clamp_int_u8((int)(v * (1 + (gainv-1)*a)));
                    }
                }
            }
            return;
        }
        if (mode == 61) {
            /* GREEN SCREEN OUT: you, cleanly keyed over solid chroma —
             * feed this straight into OBS/your compositor and key it
             * there. fx < 1.5 = green, >= 1.5 = blue. */
            cv::Vec3b chroma = (fx < 1.5f) ? cv::Vec3b(0, 255, 0)
                                           : cv::Vec3b(255, 64, 0);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float k = 1.0f - mm[x];
                    if (k < 0.02f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(chroma[0]-ip[x][0])*k);
                    ip[x][1]=(uchar)(ip[x][1]+(chroma[1]-ip[x][1])*k);
                    ip[x][2]=(uchar)(ip[x][2]+(chroma[2]-ip[x][2])*k);
                }
            }
            return;
        }
        if (mode == 62) {
            /* BRB PRIVACY: step out of frame and after ~1.5 s the feed
             * pixelates and dims with a BRB card; step back and it
             * clears instantly. Nobody watches your empty chair. */
            static int absent = 0;
            double cover = 0;
            for (int y = 0; y < img.rows; y += 12) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 12)
                    if (mm[x] > 0.4f) cover += 1;
            }
            if (cover > 6) absent = 0; else absent++;
            if (absent > 45) {
                cv::Mat tiny;
                cv::resize(img, tiny, cv::Size(img.cols/28, img.rows/28));
                cv::resize(tiny, img, img.size(), 0, 0, cv::INTER_NEAREST);
                img *= 0.55;
                cv::putText(img, "BRB", cv::Point(img.cols/2-95, img.rows/2+25),
                            cv::FONT_HERSHEY_TRIPLEX, 3.0, cv::Scalar(20,20,20), 12);
                cv::putText(img, "BRB", cv::Point(img.cols/2-95, img.rows/2+25),
                            cv::FONT_HERSHEY_TRIPLEX, 3.0, cv::Scalar(235,235,235), 4);
            }
            return;
        }
        if (mode == 63) {
            /* POP-OUT: a soft contact shadow falls from you onto the
             * real room, and you get a whisper of lift — suddenly the
             * flat webcam image has depth. fx = shadow reach. */
            cv::Mat mb;
            cv::GaussianBlur(g_seg.mask, mb, cv::Size(0,0), 7.0);
            int ox2 = (int)(14 * fx), oy2 = (int)(18 * fx);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *sb = (y >= oy2) ? mb.ptr<float>(y - oy2) : NULL;
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a < 0.5f && sb && x >= ox2) {     /* the shadow */
                        float sh = sb[x - ox2] * (1.0f - a) * 0.38f;
                        if (sh > 0.02f) {
                            ip[x][0]=(uchar)(ip[x][0]*(1-sh));
                            ip[x][1]=(uchar)(ip[x][1]*(1-sh));
                            ip[x][2]=(uchar)(ip[x][2]*(1-sh));
                        }
                    }
                    if (a > 0.02f) {                      /* the lift */
                        float k = a * 0.06f;
                        ip[x][0]=clamp_int_u8((int)(ip[x][0]*(1+k)));
                        ip[x][1]=clamp_int_u8((int)(ip[x][1]*(1+k)));
                        ip[x][2]=clamp_int_u8((int)(ip[x][2]*(1+k)));
                    }
                }
            }
            return;
        }
        if (mode == 64) {
            /* DOLLY ZOOM: the Hitchcock shot — the room slowly breathes
             * in and out of scale around your head while you stay
             * exactly the same size. Quietly unsettling. fx = depth. */
            unsigned tk = g_seg.counter;
            static float acx = -1, acy = -1;
            double sx=0, sy=0, sw2=0;
            for (int y = 0; y < img.rows/2; y += 6) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 6)
                    if (mm[x] > 0.4f) { sx+=x; sy+=y; sw2++; }
            }
            float tx2 = sw2>4 ? (float)(sx/sw2) : img.cols/2.f;
            float ty2 = sw2>4 ? (float)(sy/sw2) : img.rows/2.5f;
            if (acx < 0) { acx=tx2; acy=ty2; }
            acx += (tx2-acx)*0.04f; acy += (ty2-acy)*0.04f;
            float sc2 = 1.0f + 0.06f * fx * std::sin(tk * 0.017f);
            cv::Mat keep = img.clone();
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                float dy = (y - acy) / sc2;
                for (int x = 0; x < img.cols; x++) {
                    if (mm[x] > 0.5f) continue;           /* you stay put */
                    int sx2 = (int)(acx + (x - acx) / sc2);
                    int sy2 = (int)(acy + dy);
                    if (sx2<0||sy2<0||sx2>=img.cols||sy2>=img.rows) continue;
                    float a = mm[x];
                    cv::Vec3b v = keep.at<cv::Vec3b>(sy2, sx2);
                    ip[x][0]=(uchar)(v[0]+(ip[x][0]-v[0])*a);
                    ip[x][1]=(uchar)(v[1]+(ip[x][1]-v[1])*a);
                    ip[x][2]=(uchar)(v[2]+(ip[x][2]-v[2])*a);
                }
            }
            return;
        }
        if (mode == 65) {
            /* DUOTONE POSTER: the frame gradient-mapped to a curated
             * two-color palette — Spotify-cover graphic design, with
             * you lifted slightly out of the ink. fx = 1..4 picks:
             * 1 navy/coral  2 plum/amber  3 forest/cream  4 noir. */
            static const int PAL[4][2][3] = {   /* dark BGR, light BGR */
                {{70,25,15},{110,160,255}}, {{60,15,45},{60,190,255}},
                {{40,45,12},{200,235,240}}, {{18,16,16},{225,225,230}}};
            int pi = std::min(3, std::max(0, (int)fx - 1));
            const int *D = PAL[pi][0], *L2 = PAL[pi][1];
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    lum = std::min(1.0f, lum * (1.0f + mm[x]*0.22f));
                    lum = lum*lum*(3.0f - 2.0f*lum);      /* punchy midtones */
                    ip[x][0]=(uchar)(D[0]+(L2[0]-D[0])*lum);
                    ip[x][1]=(uchar)(D[1]+(L2[1]-D[1])*lum);
                    ip[x][2]=(uchar)(D[2]+(L2[2]-D[2])*lum);
                }
            }
            return;
        }
        if (mode == 48) {
            /* GOD RAYS: volumetric light streams past you — screen-space
             * crepuscular rays, occluded by your silhouette. fx = ray
             * strength. */
            unsigned tk = g_seg.counter;
            float lx2 = img.cols * (0.5f + 0.35f * std::sin(tk * 0.004f));
            float ly2 = -img.rows * 0.15f;
            int QW = img.cols/4, QH = img.rows/4;
            cv::Mat qm, ql(QH, QW, CV_32F), rays(QH, QW, CV_32F, cv::Scalar(0));
            cv::resize(g_seg.mask, qm, cv::Size(QW, QH));
            {   /* light source map: bright sky, occluded by you */
                cv::Mat qi; cv::resize(img, qi, cv::Size(QW, QH));
                for (int y = 0; y < QH; y++) {
                    const cv::Vec3b *ip = qi.ptr<cv::Vec3b>(y);
                    const float *mm = qm.ptr<float>(y);
                    float *lp = ql.ptr<float>(y);
                    for (int x = 0; x < QW; x++) {
                        float lum = (ip[x][0]+ip[x][1]+ip[x][2]) / (3.f*255);
                        lp[x] = (0.35f + 0.65f*lum) * (1.0f - mm[x]);
                    }
                }
            }
            const int K = 22;
            float qlx = lx2/4, qly = ly2/4;
            for (int y = 0; y < QH; y++) {
                float *rp = rays.ptr<float>(y);
                for (int x = 0; x < QW; x++) {
                    float dx = (qlx - x) / K, dy = (qly - y) / K;
                    float acc = 0, sxp = (float)x, syp = (float)y, decay = 1.0f;
                    for (int k2 = 0; k2 < K; k2++) {
                        sxp += dx; syp += dy;
                        int ix = (int)sxp, iy = (int)syp;
                        if (ix < 0 || iy < 0 || ix >= QW || iy >= QH) break;
                        acc += ql.at<float>(iy, ix) * decay;
                        decay *= 0.93f;
                    }
                    rp[x] = acc / K;
                }
            }
            cv::Mat rf; cv::resize(rays, rf, img.size(), 0, 0, cv::INTER_LINEAR);
            float st = 220.0f * fx;
            for (int y = 0; y < img.rows; y++) {
                const float *rp = rf.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float v = rp[x]; if (v < 0.03f) continue;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(v*st*0.55f));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(v*st*0.85f));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(v*st));
                }
            }
            return;
        }
        if (mode == 49) {
            /* TRON TRACE: your outline as a glowing neon circuit with a
             * bright pulse RACING around the perimeter. fx = glow. */
            unsigned tk = g_seg.counter;
            img *= 0.45;
            cv::Mat keep2;
            {   /* keep the person mildly lit */
                for (int y = 0; y < img.rows; y++) {
                    const float *mm = g_seg.mask.ptr<float>(y);
                    cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < img.cols; x++) {
                        float a = mm[x] * 0.5f; if (a < 0.02f) continue;
                        ip[x][0]=clamp_int_u8((int)(ip[x][0]*(1+a)));
                        ip[x][1]=clamp_int_u8((int)(ip[x][1]*(1+a)));
                        ip[x][2]=clamp_int_u8((int)(ip[x][2]*(1+a)));
                    }
                }
            }
            (void)keep2;
            cv::Mat q, q8;
            cv::resize(g_seg.mask, q, cv::Size(), 0.25, 0.25);
            q8 = q > 0.5;
            std::vector<std::vector<cv::Point>> ctrs;
            cv::findContours(q8, ctrs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            if (!ctrs.empty()) {
                size_t big = 0;
                for (size_t i = 1; i < ctrs.size(); i++)
                    if (ctrs[i].size() > ctrs[big].size()) big = i;
                std::vector<cv::Point> C2(ctrs[big].size());
                for (size_t i = 0; i < C2.size(); i++)
                    C2[i] = cv::Point(ctrs[big][i].x*4, ctrs[big][i].y*4);
                int gw = std::max(2, (int)(6*fx));
                cv::polylines(img, C2, true, cv::Scalar(120,60,10), gw+6, cv::LINE_AA);
                cv::polylines(img, C2, true, cv::Scalar(255,160,40), gw, cv::LINE_AA);
                cv::polylines(img, C2, true, cv::Scalar(255,240,200), 1, cv::LINE_AA);
                /* the racing pulse: a bright arc of ~8% perimeter */
                size_t N2 = C2.size();
                if (N2 > 24) {
                    size_t head2 = (size_t)(tk * 6) % N2;
                    size_t seg2 = std::max((size_t)8, N2 / 12);
                    for (size_t i = 0; i < seg2; i++) {
                        size_t a2 = (head2 + i) % N2, b2 = (head2 + i + 1) % N2;
                        float br = 1.0f - (float)i / seg2;
                        cv::line(img, C2[a2], C2[b2],
                                 cv::Scalar(255*br, 255*br, 255),
                                 (int)(gw + 3*br), cv::LINE_AA);
                    }
                }
            }
            return;
        }
        if (mode == 50) {
            /* SHOCKWAVE: move fast and an expanding refraction ring
             * bursts out of you. fx = ring power. */
            static float rr2[3] = {-1,-1,-1};
            static float rcx2[3], rcy2[3];
            static float pcx2 = -1, pcy2 = -1;
            unsigned tk = g_seg.counter;
            double sx=0, sy=0, sw2=0;
            for (int y = 0; y < img.rows; y += 8) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 8)
                    if (mm[x] > 0.4f) { sx+=x; sy+=y; sw2++; }
            }
            if (sw2 > 4) {
                float cx2=(float)(sx/sw2), cy2=(float)(sy/sw2);
                if (pcx2 >= 0) {
                    float v = std::sqrt((cx2-pcx2)*(cx2-pcx2)+(cy2-pcy2)*(cy2-pcy2));
                    if (v > img.cols * 0.013f && (tk % 5) == 0) {
                        for (int r3 = 0; r3 < 3; r3++) if (rr2[r3] < 0) {
                            rr2[r3] = 10; rcx2[r3] = cx2; rcy2[r3] = cy2; break;
                        }
                    }
                }
                pcx2 = cx2; pcy2 = cy2;
            }
            cv::Mat keep = img.clone();
            for (int r3 = 0; r3 < 3; r3++) {
                if (rr2[r3] < 0) continue;
                rr2[r3] += 16;
                float R = rr2[r3];
                float band = 26.0f;
                if (R > img.cols) { rr2[r3] = -1; continue; }
                int x0 = std::max(0, (int)(rcx2[r3]-R-band));
                int x1 = std::min(img.cols-1, (int)(rcx2[r3]+R+band));
                int y0 = std::max(0, (int)(rcy2[r3]-R-band));
                int y1 = std::min(img.rows-1, (int)(rcy2[r3]+R+band));
                float fade = 1.0f - R / img.cols;
                for (int y = y0; y <= y1; y++) {
                    cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                    float dy = y - rcy2[r3];
                    for (int x = x0; x <= x1; x++) {
                        float dx = x - rcx2[r3];
                        float d = std::sqrt(dx*dx + dy*dy);
                        float dd = d - R;
                        if (std::fabs(dd) > band) continue;
                        float k2 = (1.0f - std::fabs(dd)/band);
                        float push = k2*k2 * 22.0f * fx * fade;
                        int sx2 = (int)(x + dx/std::max(1.f,d)*push);
                        int sy2 = (int)(y + dy/std::max(1.f,d)*push);
                        if (sx2<0||sy2<0||sx2>=img.cols||sy2>=img.rows) continue;
                        ip[x] = keep.at<cv::Vec3b>(sy2, sx2);
                        int gl = (int)(k2 * 60 * fade);
                        ip[x][0]=clamp_int_u8(ip[x][0]+gl);
                        ip[x][1]=clamp_int_u8(ip[x][1]+gl);
                        ip[x][2]=clamp_int_u8(ip[x][2]+gl);
                    }
                }
            }
            return;
        }
        if (mode == 51) {
            /* RAPTURE: a pillar of light takes you — you slowly levitate
             * inside a golden beam while sparks rise. fx = lift. */
            unsigned tk = g_seg.counter;
            float phase = 0.5f - 0.5f * std::cos(tk * 0.02f);
            int lift = (int)(phase * img.rows * 0.14f * fx);
            double sx=0, sw2=0;
            for (int y = 0; y < img.rows; y += 8) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 8)
                    if (mm[x] > 0.4f) { sx+=x; sw2++; }
            }
            float ccx = sw2>4 ? (float)(sx/sw2) : img.cols/2.f;
            cv::Mat out2; g_seg.bg_est.copyTo(out2);
            for (int y = 0; y < img.rows; y++) {          /* lifted you */
                int sy2 = y + lift;
                if (sy2 >= img.rows) continue;
                const float *mm = g_seg.mask.ptr<float>(sy2);
                const cv::Vec3b *ip = img.ptr<cv::Vec3b>(sy2);
                cv::Vec3b *op = out2.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    op[x][0]=(uchar)(op[x][0]+(ip[x][0]-op[x][0])*a);
                    op[x][1]=(uchar)(op[x][1]+(ip[x][1]-op[x][1])*a);
                    op[x][2]=(uchar)(op[x][2]+(ip[x][2]-op[x][2])*a);
                }
            }
            float bw2 = img.cols * 0.11f;
            for (int y = 0; y < img.rows; y++) {          /* the beam */
                cv::Vec3b *op = out2.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float d = std::fabs(x - ccx) / bw2;
                    if (d >= 1.0f) continue;
                    float k2 = (1.0f - d)*(1.0f - d) * (0.55f + 0.1f*std::sin(tk*0.13f));
                    op[x][0]=clamp_int_u8(op[x][0]+(int)(k2*120));
                    op[x][1]=clamp_int_u8(op[x][1]+(int)(k2*190));
                    op[x][2]=clamp_int_u8(op[x][2]+(int)(k2*235));
                }
            }
            for (int i = 0; i < 40; i++) {                /* rising sparks */
                float px2 = ccx + (uhash(i,5,0)-0.5f)*bw2*1.6f;
                float py2 = (float)((int)(img.rows + uhash(i,9,0)*img.rows
                            - tk*(2+3*uhash(i,3,0))) % (img.rows+40)) - 20;
                int X=(int)px2, Y=(int)py2;
                if (X<1||Y<1||X>=img.cols-1||Y>=img.rows-1) continue;
                cv::circle(out2, {X,Y}, 1, cv::Scalar(200,240,255), cv::FILLED);
            }
            out2.copyTo(img);
            return;
        }
        if (mode == 52) {
            /* KALEIDO-BODY: your matted body folded into 8 radial
             * mirror-wedges — a living mandala of flesh. fx unused. */
            unsigned tk = g_seg.counter;
            int QW = img.cols/2, QH = img.rows/2;
            cv::Mat qi, qm, outq(QH, QW, CV_8UC3);
            cv::resize(img, qi, cv::Size(QW, QH), 0, 0, cv::INTER_AREA);
            cv::resize(g_seg.mask, qm, cv::Size(QW, QH));
            float ccx = QW/2.f, ccy = QH/2.f;
            const float WEDGE = 6.28318f / 8;
            float rot = tk * 0.008f;
            for (int y = 0; y < QH; y++) {
                cv::Vec3b *op = outq.ptr<cv::Vec3b>(y);
                float dy = y - ccy;
                for (int x = 0; x < QW; x++) {
                    float dx = x - ccx;
                    float r = std::sqrt(dx*dx + dy*dy);
                    float th = std::atan2(dy, dx) + rot;
                    th = std::fmod(th + 12.566f, WEDGE);
                    if (th > WEDGE/2) th = WEDGE - th;    /* mirror fold */
                    int sx2 = (int)(ccx + std::cos(th)*r);
                    int sy2 = (int)(ccy + std::sin(th)*r);
                    if (sx2<0||sy2<0||sx2>=QW||sy2>=QH) { op[x]=cv::Vec3b(8,6,10); continue; }
                    float a = qm.at<float>(sy2, sx2);
                    cv::Vec3b v = qi.at<cv::Vec3b>(sy2, sx2);
                    op[x] = cv::Vec3b((uchar)(v[0]*a + 8*(1-a)),
                                      (uchar)(v[1]*a + 6*(1-a)),
                                      (uchar)(v[2]*a + 10*(1-a)));
                }
            }
            cv::resize(outq, img, img.size(), 0, 0, cv::INTER_LINEAR);
            return;
        }
        if (mode == 53) {
            /* LIQUID FLOOR: the bottom of the frame becomes rippling
             * water reflecting the scene — you stand on a dark lake.
             * fx = wave strength. */
            unsigned tk = g_seg.counter;
            int yline = (int)(img.rows * 0.68f);
            cv::Mat keep = img.clone();
            for (int y = yline; y < img.rows; y++) {
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                int depth = y - yline;
                for (int x = 0; x < img.cols; x++) {
                    float wob = std::sin(x*0.05f + tk*0.11f + depth*0.13f)
                              * (2.0f + depth*0.10f) * fx;
                    int ry2 = yline - depth - 1 + (int)wob;
                    int rx2 = x + (int)(std::sin(depth*0.21f + tk*0.07f)*2.5f*fx);
                    ry2 = std::max(0, std::min(img.rows-1, ry2));
                    rx2 = std::max(0, std::min(img.cols-1, rx2));
                    cv::Vec3b v = keep.at<cv::Vec3b>(ry2, rx2);
                    float fade = 1.0f - (float)depth / (img.rows - yline);
                    ip[x][0]=(uchar)(v[0]*0.45f*fade + 30);
                    ip[x][1]=(uchar)(v[1]*0.42f*fade + 18);
                    ip[x][2]=(uchar)(v[2]*0.38f*fade + 8);
                }
            }
            /* waterline shimmer */
            for (int x = 0; x < img.cols; x++) {
                int yy = yline + (int)(std::sin(x*0.09f + tk*0.15f)*1.5f);
                if (yy < 1 || yy >= img.rows-1) continue;
                cv::Vec3b *p2 = img.ptr<cv::Vec3b>(yy) + x;
                p2[0][0]=clamp_int_u8(p2[0][0]+70);
                p2[0][1]=clamp_int_u8(p2[0][1]+70);
                p2[0][2]=clamp_int_u8(p2[0][2]+70);
            }
            return;
        }
        if (mode == 54) {
            /* ASTRAL SOUL: a translucent, cyan-shifted copy of you
             * drifts out of your body in a slow orbit and sinks back.
             * fx = how far the soul strays. */
            unsigned tk = g_seg.counter;
            float gx2 = std::sin(tk*0.013f) * img.cols * 0.055f * fx;
            float gy2 = std::cos(tk*0.009f) * img.rows * 0.035f * fx
                      - img.rows * 0.02f * fx;
            int ox2 = (int)gx2, oy2 = (int)gy2;
            cv::Mat keep = img.clone();
            for (int y = 0; y < img.rows; y++) {
                int sy2 = y - oy2;
                if (sy2 < 0 || sy2 >= img.rows) continue;
                const float *sm2 = g_seg.mask.ptr<float>(sy2);
                const float *cm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *kp = keep.ptr<cv::Vec3b>(sy2);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    int sx2 = x - ox2;
                    if (sx2 < 0 || sx2 >= img.cols) continue;
                    float a = sm2[sx2] * 0.42f * (1.0f - cm[x]);
                    if (a < 0.03f) continue;
                    float lum = (kp[sx2][0]*0.114f+kp[sx2][1]*0.587f+kp[sx2][2]*0.299f);
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(lum*1.2f*a));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(lum*1.05f*a));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(lum*0.45f*a));
                }
            }
            return;
        }
        if (mode == 55) {
            /* MYSTIC FOG: two layers of drifting procedural fog — one
             * rolls BEHIND you, a thinner one in front of everything.
             * Smoothed value noise, scrolling at different speeds.
             * fx = density. */
            unsigned tk = g_seg.counter;
            auto vnoise = [&](float X, float Y, unsigned salt)->float{
                int x0=(int)std::floor(X), y0=(int)std::floor(Y);
                float tx2=X-x0, ty2=Y-y0;
                float a2=uhash(x0,y0,salt),   b2=uhash(x0+1,y0,salt);
                float c2=uhash(x0,y0+1,salt), d2=uhash(x0+1,y0+1,salt);
                float u2=tx2*tx2*(3-2*tx2), v2=ty2*ty2*(3-2*ty2);
                return a2+(b2-a2)*u2+(c2-a2)*v2+(a2-b2-c2+d2)*u2*v2;
            };
            int QW=img.cols/4, QH=img.rows/4;
            cv::Mat fog1(QH,QW,CV_32F), fog2(QH,QW,CV_32F);
            float t1=tk*0.010f, t2=tk*0.021f;
            for(int y=0;y<QH;y++){
                float*f1=fog1.ptr<float>(y);float*f2=fog2.ptr<float>(y);
                for(int x=0;x<QW;x++){
                    float v1=vnoise(x*0.045f+t1, y*0.09f, 11)*0.65f
                            +vnoise(x*0.11f+t1*1.7f, y*0.2f, 12)*0.35f;
                    float v2=vnoise(x*0.03f-t2, y*0.07f+t2*0.3f, 13)*0.7f
                            +vnoise(x*0.09f-t2*1.5f, y*0.18f, 14)*0.3f;
                    f1[x]=std::max(0.f,(v1-0.35f)*1.6f);
                    f2[x]=std::max(0.f,(v2-0.45f)*1.8f);
                }
            }
            cv::Mat F1,F2;
            cv::resize(fog1,F1,img.size(),0,0,cv::INTER_LINEAR);
            cv::resize(fog2,F2,img.size(),0,0,cv::INTER_LINEAR);
            for(int y=0;y<img.rows;y++){
                const float*a2=F1.ptr<float>(y);
                const float*b2=F2.ptr<float>(y);
                const float*mm=g_seg.mask.ptr<float>(y);
                cv::Vec3b*ip=img.ptr<cv::Vec3b>(y);
                for(int x=0;x<img.cols;x++){
                    /* back fog only behind you; front fog over all */
                    float k2=a2[x]*0.55f*fx*(1.0f-mm[x]) + b2[x]*0.28f*fx;
                    if(k2<0.02f)continue;
                    if(k2>0.85f)k2=0.85f;
                    ip[x][0]=(uchar)(ip[x][0]+(190-ip[x][0])*k2);
                    ip[x][1]=(uchar)(ip[x][1]+(185-ip[x][1])*k2);
                    ip[x][2]=(uchar)(ip[x][2]+(175-ip[x][2])*k2);
                }
            }
            return;
        }
        if (mode == 38) {
            /* SNOWFALL: snow that ACCUMULATES on you — flakes land on
             * the upward-facing surfaces of your silhouette (mask
             * gradient says which way is 'up'), pile there, and tumble
             * off when you move. */
            static float fx2[200], fy2[200], fvx[200];
            static float slx[600], sly[600];
            static int sinit2 = 0, nset = 0;
            unsigned tk = g_seg.counter;
            if (!sinit2) {
                for (int i=0;i<200;i++){fx2[i]=uhash(i,1,3)*img.cols;
                    fy2[i]=-uhash(i,2,3)*img.rows;fvx[i]=(uhash(i,3,3)-0.5f)*0.8f;}
                sinit2 = 1;
            }
            auto m_at = [&](int X,int Y)->float{
                if(X<0||Y<0||X>=img.cols||Y>=img.rows)return 0.f;
                return g_seg.mask.at<float>(Y,X);};
            for (int i = 0; i < 200; i++) {               /* falling */
                fx2[i]+=fvx[i]+std::sin((tk+i*7)*0.03f)*0.4f;
                fy2[i]+=(1.2f+uhash(i,9,0)*0.8f)*fx;
                int X=(int)fx2[i],Y=(int)fy2[i];
                if(Y>=img.rows||X<0||X>=img.cols){
                    fx2[i]=uhash(i,tk,4)*img.cols;fy2[i]=-8;continue;}
                /* landing: entering the mask from above on an up-facing
                 * surface -> settle */
                if(m_at(X,Y+1)>0.5f && m_at(X,Y-3)<0.3f){
                    if(nset<600){slx[nset]=fx2[i];sly[nset]=fy2[i];nset++;}
                    fx2[i]=uhash(i,tk,5)*img.cols;fy2[i]=-8;continue;
                }
                if(Y>=0)cv::circle(img,{X,Y},1,cv::Scalar(250,250,255),cv::FILLED);
            }
            for (int i = 0; i < nset; i++) {              /* settled */
                int X=(int)slx[i],Y=(int)sly[i];
                if(m_at(X,Y+1)<0.4f){                     /* you moved: tumble */
                    slx[i]=slx[nset-1];sly[i]=sly[nset-1];nset--;i--;continue;}
                cv::circle(img,{X,Y},1,cv::Scalar(255,255,255),cv::FILLED);
                cv::circle(img,{X,Y-1},1,cv::Scalar(235,240,255),cv::FILLED);
            }
            return;
        }
        if (mode == 39) {
            /* FIRE SILHOUETTE: doom-fire seeded at your outline — real
             * heat diffusion rises off your body. */
            static cv::Mat heat;
            unsigned tk = g_seg.counter; (void)tk;
            int QW=img.cols/4, QH=img.rows/4;
            if(heat.size()!=cv::Size(QW,QH))
                heat=cv::Mat::zeros(QH,QW,CV_32F);
            cv::Mat qm; cv::resize(g_seg.mask,qm,cv::Size(QW,QH));
            for(int y=0;y<QH;y++){                        /* seed the edge */
                const float*mm=qm.ptr<float>(y);float*hp=heat.ptr<float>(y);
                for(int x=0;x<QW;x++)
                    if(mm[x]>0.30f&&mm[x]<0.72f)hp[x]=std::min(1.0f,fx);
            }
            for(int y=0;y<QH-1;y++){                      /* rise + cool */
                float*hp=heat.ptr<float>(y);
                const float*hd=heat.ptr<float>(y+1);
                for(int x=0;x<QW;x++){
                    float a=hd[x];
                    if(x>0)a+=hd[x-1];else a+=hd[x];
                    if(x<QW-1)a+=hd[x+1];else a+=hd[x];
                    float v=(a/3.0f)*0.93f - 0.004f;
                    hp[x]=std::max(hp[x]*0.55f, std::max(0.0f,v));
                }
            }
            cv::Mat hf; cv::resize(heat,hf,img.size(),0,0,cv::INTER_LINEAR);
            for(int y=0;y<img.rows;y++){
                const float*hp=hf.ptr<float>(y);
                cv::Vec3b*ip=img.ptr<cv::Vec3b>(y);
                for(int x=0;x<img.cols;x++){
                    float v=hp[x]; if(v<0.06f)continue;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(v*v*v*220));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(v*v*200));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(v*255));
                }
            }
            return;
        }
        if (mode == 40) {
            /* BUBBLES: they drift up; touch one and it POPS into
             * droplets. Your hands are the pin. */
            static float bx2[24],by3[24],bph[24];
            static float dxp[64],dyp[64],dvx2[64],dvy2[64],dl[64];
            static int binit2=0;
            unsigned tk=g_seg.counter;
            if(!binit2){for(int i=0;i<24;i++){bx2[i]=uhash(i,5,1)*img.cols;
                by3[i]=img.rows+uhash(i,6,1)*img.rows;bph[i]=uhash(i,7,1)*6.28f;}
                for(int i=0;i<64;i++)dl[i]=0;
                binit2=1;}
            for(int i=0;i<24;i++){
                by3[i]-=0.9f+0.4f*uhash(i,8,0);
                bx2[i]+=std::sin(tk*0.03f+bph[i])*0.7f;
                int X=(int)bx2[i],Y=(int)by3[i];
                int R2=8+(int)(uhash(i,9,1)*10);
                if(Y<-R2){by3[i]=img.rows+R2;bx2[i]=uhash(i,tk,2)*img.cols;continue;}
                bool pop=false;
                if(X>=0&&X<img.cols&&Y>=0&&Y<img.rows&&
                   g_seg.mask.at<float>(Y,X)>0.45f)pop=true;
                if(pop){
                    for(int k2=0,sp=0;k2<64&&sp<8;k2++)if(dl[k2]<=0){
                        dxp[k2]=(float)X;dyp[k2]=(float)Y;
                        dvx2[k2]=(uhash(k2,tk,3)-0.5f)*5;
                        dvy2[k2]=-uhash(k2,tk,4)*3;dl[k2]=1;sp++;}
                    by3[i]=img.rows+R2;bx2[i]=uhash(i,tk,6)*img.cols;continue;
                }
                cv::circle(img,{X,Y},R2,cv::Scalar(255,230,200),1,cv::LINE_AA);
                cv::ellipse(img,{X-R2/3,Y-R2/3},cv::Size(R2/3,R2/5),-40,0,180,
                            cv::Scalar(255,255,255),1,cv::LINE_AA);
            }
            for(int i=0;i<64;i++){                        /* droplets */
                if(dl[i]<=0)continue;
                dvy2[i]+=0.3f;dxp[i]+=dvx2[i];dyp[i]+=dvy2[i];dl[i]-=0.05f;
                int X=(int)dxp[i],Y=(int)dyp[i];
                if(X<0||Y<0||X>=img.cols||Y>=img.rows){dl[i]=0;continue;}
                cv::circle(img,{X,Y},1,cv::Scalar(255,240,210),cv::FILLED);
            }
            return;
        }
        if (mode == 41) {
            /* THOR: lightning strikes your highest point — raise a hand
             * and the sky answers. */
            static int strike_t = -999; static unsigned strike_seed = 1;
            unsigned tk = g_seg.counter;
            int topx=-1, topy=-1;
            for (int y = 0; y < img.rows && topx < 0; y += 3) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 3)
                    if (mm[x] > 0.5f) { topx = x; topy = y; break; }
            }
            if (topx >= 0 && (int)tk - strike_t > 55 &&
                uhash(tk, 3, 9) > 0.94f) {
                strike_t = (int)tk; strike_seed = tk * 2654435761u + 1;
            }
            int since = (int)tk - strike_t;
            if (topx >= 0 && since >= 0 && since < 7) {
                unsigned sd = strike_seed;
                float px2 = topx + (uhash(sd,1,0)-0.5f)*img.cols*0.3f;
                float py2 = 0;
                float lx2 = px2, ly2 = py2;
                int flick = (since*strike_seed)%3;
                while (ly2 < topy - 4) {
                    sd = sd*1664525u+1013904223u;
                    float nx2 = lx2 + ((float)(sd%1000)/1000.f-0.5f)*36
                              + (topx-lx2)*0.16f;
                    float ny2 = ly2 + 10 + (sd>>10)%14;
                    if (flick != 2) {
                        cv::line(img,{(int)lx2,(int)ly2},{(int)nx2,(int)ny2},
                                 cv::Scalar(255,200,140),5,cv::LINE_AA);
                        cv::line(img,{(int)lx2,(int)ly2},{(int)nx2,(int)ny2},
                                 cv::Scalar(255,255,255),2,cv::LINE_AA);
                    }
                    lx2=nx2; ly2=ny2;
                }
                if (since < 2) img += cv::Scalar(40,40,55);   /* flash */
                cv::circle(img,{topx,topy},10+since*3,
                           cv::Scalar(255,230,180),2,cv::LINE_AA);
            }
            return;
        }
        if (mode == 42) {
            /* FIREFLIES: thirty glowing agents wander the room but are
             * drawn to your silhouette — they hover and drift around
             * your outline, repelled if they cross inside. */
            static float ax2[30],ay2[30],avx[30],avy[30],aph[30];
            static int ainit=0;
            unsigned tk=g_seg.counter;
            if(!ainit){for(int i=0;i<30;i++){ax2[i]=uhash(i,4,2)*img.cols;
                ay2[i]=uhash(i,5,2)*img.rows;avx[i]=avy[i]=0;
                aph[i]=uhash(i,6,2)*6.28f;}ainit=1;}
            /* centroid for gentle attraction */
            double sx=0,sy=0,sw2=0;
            for(int y=0;y<img.rows;y+=10){const float*mm=g_seg.mask.ptr<float>(y);
                for(int x=0;x<img.cols;x+=10)if(mm[x]>0.4f){sx+=x;sy+=y;sw2++;}}
            float ccx=sw2>2?(float)(sx/sw2):img.cols/2.f;
            float ccy=sw2>2?(float)(sy/sw2):img.rows/2.f;
            auto m_at=[&](int X,int Y)->float{
                if(X<0||Y<0||X>=img.cols||Y>=img.rows)return 0.f;
                return g_seg.mask.at<float>(Y,X);};
            for(int i=0;i<30;i++){
                avx[i]+=(uhash(i,tk,7)-0.5f)*0.5f+(ccx-ax2[i])*0.0012f;
                avy[i]+=(uhash(i,tk,8)-0.5f)*0.5f+(ccy-ay2[i])*0.0012f;
                float m2=m_at((int)ax2[i],(int)ay2[i]);
                if(m2>0.4f){                              /* inside: push out */
                    float gx2=m_at((int)ax2[i]+4,(int)ay2[i])-m_at((int)ax2[i]-4,(int)ay2[i]);
                    float gy2=m_at((int)ax2[i],(int)ay2[i]+4)-m_at((int)ax2[i],(int)ay2[i]-4);
                    avx[i]-=gx2*4; avy[i]-=gy2*4;
                }
                avx[i]*=0.92f;avy[i]*=0.92f;
                ax2[i]+=avx[i];ay2[i]+=avy[i];
                if(ax2[i]<0)ax2[i]=0;
                if(ax2[i]>img.cols-1)ax2[i]=(float)img.cols-1;
                if(ay2[i]<0)ay2[i]=0;
                if(ay2[i]>img.rows-1)ay2[i]=(float)img.rows-1;
                float glow=0.55f+0.45f*std::sin(tk*0.11f+aph[i]);
                int X=(int)ax2[i],Y=(int)ay2[i];
                cv::circle(img,{X,Y},4,cv::Scalar(20,90*glow,90*glow),cv::FILLED,cv::LINE_AA);
                cv::circle(img,{X,Y},2,cv::Scalar(120*glow,255*glow,255*glow),cv::FILLED,cv::LINE_AA);
            }
            return;
        }
        if (mode == 43) {
            /* STARDUST: your moving edges shed glitter — wherever the
             * matte CHANGED since last frame, sparkles are born. */
            static cv::Mat prevq;
            static float gx3[220],gy3[220],gl2[220],gph[220];
            static int ginit=0;
            unsigned tk=g_seg.counter;
            if(!ginit){for(int i=0;i<220;i++)gl2[i]=0;ginit=1;}
            int QW=img.cols/4,QH=img.rows/4;
            cv::Mat qm;cv::resize(g_seg.mask,qm,cv::Size(QW,QH));
            if(!prevq.empty()&&prevq.size()==qm.size()){
                int spawned=0;
                for(int y=0;y<QH&&spawned<10;y+=2){
                    const float*a2=qm.ptr<float>(y);
                    const float*b2=prevq.ptr<float>(y);
                    for(int x=0;x<QW&&spawned<10;x+=2){
                        if(std::fabs(a2[x]-b2[x])>0.30f&&uhash(x,y,tk)>0.85f){
                            for(int e=0;e<220;e++)if(gl2[e]<=0){
                                gx3[e]=x*4.f;gy3[e]=y*4.f;gl2[e]=1;
                                gph[e]=uhash(e,tk,1)*6.28f;spawned++;break;}
                        }
                    }
                }
            }
            qm.copyTo(prevq);
            for(int i=0;i<220;i++){
                if(gl2[i]<=0)continue;
                gx3[i]+=(uhash(i,tk,2)-0.5f)*1.6f;
                gy3[i]-=0.4f; gl2[i]-=0.014f;
                int X=(int)gx3[i],Y=(int)gy3[i];
                if(X<1||Y<1||X>=img.cols-1||Y>=img.rows-1){gl2[i]=0;continue;}
                float tw=(0.5f+0.5f*std::sin(tk*0.4f+gph[i]))*gl2[i];
                int br=(int)(tw*255);
                cv::Vec3b*p2=img.ptr<cv::Vec3b>(Y)+X;
                p2[0][0]=clamp_int_u8(p2[0][0]+br);
                p2[0][1]=clamp_int_u8(p2[0][1]+br);
                p2[0][2]=clamp_int_u8(p2[0][2]+(int)(br*0.85f));
            }
            return;
        }
        if (mode == 44) {
            /* MATRIX YOU: digital rain falls INSIDE your silhouette —
             * you are the code. */
            static float head[240]; static float spd2[240];
            static int minit=0;
            unsigned tk=g_seg.counter;
            const int CW=10, CH=14;
            int ncol=std::min(240,img.cols/CW);
            if(!minit){for(int i=0;i<240;i++){head[i]=uhash(i,3,5)*60;
                spd2[i]=0.4f+uhash(i,4,5)*0.8f;}minit=1;}
            /* darken the body slightly so the rain reads */
            for(int y=0;y<img.rows;y++){
                const float*mm=g_seg.mask.ptr<float>(y);
                cv::Vec3b*ip=img.ptr<cv::Vec3b>(y);
                for(int x=0;x<img.cols;x++){
                    float a=mm[x];if(a<0.05f)continue;
                    ip[x][0]=(uchar)(ip[x][0]*(1-0.75f*a));
                    ip[x][1]=(uchar)(ip[x][1]*(1-0.55f*a));
                    ip[x][2]=(uchar)(ip[x][2]*(1-0.75f*a));
                }
            }
            for(int c2=0;c2<ncol;c2++){
                head[c2]+=spd2[c2]*fx;
                int rows2=img.rows/CH;
                if((int)head[c2]>rows2+8){head[c2]=0;spd2[c2]=0.4f+uhash(c2,tk,6)*0.8f;}
                for(int r2=0;r2<rows2;r2++){
                    float d=head[c2]-r2;
                    if(d<0||d>9)continue;
                    int X=c2*CW+2,Y=r2*CH+2;
                    if(g_seg.mask.at<float>(std::min(img.rows-1,Y+CH/2),
                                            std::min(img.cols-1,X+CW/2))<0.5f)continue;
                    int g2=d<1?255:(int)(220*(1.0f-d/9));
                    cv::Scalar col=d<1?cv::Scalar(200,255,200):cv::Scalar(0,g2,0);
                    if(uhash(c2,r2,tk/6)>0.5f)
                        cv::rectangle(img,cv::Rect(X,Y,CW-4,CH-5),col,cv::FILLED);
                    else
                        cv::rectangle(img,cv::Rect(X,Y+CH/3,CW-4,CH/2-3),col,cv::FILLED);
                }
            }
            return;
        }
        if (mode == 45) {
            /* BLACK HOLE YOU: the room is sucked toward you — radial
             * inward stretch with an accretion glow; you stay solid. */
            unsigned tk=g_seg.counter;
            static float hcx=-1,hcy=-1;
            double sx=0,sy=0,sw2=0;
            for(int y=0;y<img.rows;y+=8){const float*mm=g_seg.mask.ptr<float>(y);
                for(int x=0;x<img.cols;x+=8)if(mm[x]>0.4f){sx+=x;sy+=y;sw2++;}}
            float tx2=sw2>4?(float)(sx/sw2):img.cols/2.f;
            float ty2=sw2>4?(float)(sy/sw2):img.rows/2.f;
            if(hcx<0){hcx=tx2;hcy=ty2;}
            hcx+=(tx2-hcx)*0.05f;hcy+=(ty2-hcy)*0.05f;
            static cv::Mat hpx,hpy;
            if(hpx.size()!=img.size()){hpx.create(img.size(),CV_32F);
                hpy.create(img.size(),CV_32F);}
            float rmax=std::sqrt((float)(img.cols*img.cols+img.rows*img.rows))*0.5f;
            for(int y=0;y<img.rows;y++){
                float*axp=hpx.ptr<float>(y);float*ayp=hpy.ptr<float>(y);
                float dy=y-hcy;
                for(int x=0;x<img.cols;x++){
                    float dx=x-hcx;
                    float r=std::sqrt(dx*dx+dy*dy)+1e-3f;
                    float pull=(1.0f-r/rmax);
                    float rr=r+pull*pull*110.0f*fx;       /* sample further out */
                    axp[x]=hcx+dx/r*rr; ayp[x]=hcy+dy/r*rr;
                }
            }
            cv::Mat keep=img.clone(),warped;
            cv::remap(img,warped,hpx,hpy,cv::INTER_LINEAR,cv::BORDER_REFLECT);
            warped.copyTo(img);
            for(int y=0;y<img.rows;y++){                  /* you stay solid */
                const float*mm=g_seg.mask.ptr<float>(y);
                const cv::Vec3b*kp=keep.ptr<cv::Vec3b>(y);
                cv::Vec3b*ip=img.ptr<cv::Vec3b>(y);
                float dy=y-hcy;
                for(int x=0;x<img.cols;x++){
                    float a=mm[x];
                    if(a>0.05f){
                        ip[x][0]=(uchar)(ip[x][0]+(kp[x][0]-ip[x][0])*a);
                        ip[x][1]=(uchar)(ip[x][1]+(kp[x][1]-ip[x][1])*a);
                        ip[x][2]=(uchar)(ip[x][2]+(kp[x][2]-ip[x][2])*a);
                        continue;
                    }
                    float dx=x-hcx;
                    float r=std::sqrt(dx*dx+dy*dy);
                    float ring=std::exp(-std::fabs(r-rmax*0.32f
                                 -10*std::sin(tk*0.05f))*0.06f);
                    if(ring>0.1f){
                        ip[x][0]=clamp_int_u8(ip[x][0]+(int)(ring*60));
                        ip[x][1]=clamp_int_u8(ip[x][1]+(int)(ring*110));
                        ip[x][2]=clamp_int_u8(ip[x][2]+(int)(ring*200));
                    }
                }
            }
            return;
        }
        if (mode == 46) {
            /* PET MOONS: two tiny LIVE copies of you orbit your head —
             * they mimic you in real time, because they ARE you. */
            unsigned tk=g_seg.counter;
            double sx=0,sy=0,sw2=0;
            for(int y=0;y<img.rows/2;y+=6){const float*mm=g_seg.mask.ptr<float>(y);
                for(int x=0;x<img.cols;x+=6)if(mm[x]>0.4f){sx+=x;sy+=y;sw2++;}}
            float ccx=sw2>4?(float)(sx/sw2):img.cols/2.f;
            float ccy=sw2>4?(float)(sy/sw2):img.rows/3.f;
            int mw2=img.cols/6,mh3=img.rows/6;
            cv::Mat mini,minia;
            cv::resize(img,mini,cv::Size(mw2,mh3),0,0,cv::INTER_AREA);
            cv::resize(g_seg.mask,minia,cv::Size(mw2,mh3));
            float rad=img.cols*0.22f*fx;
            for(int k2=0;k2<2;k2++){
                float aa=tk*0.025f+k2*3.1416f;
                int px2=(int)(ccx+std::cos(aa)*rad)-mw2/2;
                int py2=(int)(ccy+std::sin(aa)*rad*0.45f)-mh3/2;
                for(int yy=0;yy<mh3;yy++){
                    int Y=py2+yy; if(Y<0||Y>=img.rows)continue;
                    const float*am=minia.ptr<float>(yy);
                    const cv::Vec3b*sp=mini.ptr<cv::Vec3b>(yy);
                    cv::Vec3b*ip=img.ptr<cv::Vec3b>(Y);
                    for(int xx=0;xx<mw2;xx++){
                        int X=px2+xx; if(X<0||X>=img.cols)continue;
                        float a=am[xx]; if(a<0.08f)continue;
                        ip[X][0]=(uchar)(ip[X][0]+(sp[xx][0]-ip[X][0])*a);
                        ip[X][1]=(uchar)(ip[X][1]+(sp[xx][1]-ip[X][1])*a);
                        ip[X][2]=(uchar)(ip[X][2]+(sp[xx][2]-ip[X][2])*a);
                    }
                }
            }
            return;
        }
        if (mode == 47) {
            /* PORTALS: a blue ring and an orange ring — whatever part
             * of you enters the blue portal steps out of the orange
             * one, live. */
            unsigned tk=g_seg.counter;
            float Ax=img.cols*0.20f, Ay=img.rows*0.55f;
            float Bx=img.cols*0.80f, By=img.rows*0.55f;
            float rx2=img.cols*0.105f, ry2=img.rows*0.30f;
            for(int y=0;y<img.rows;y++){
                cv::Vec3b*ip=img.ptr<cv::Vec3b>(y);
                float ndy=(y-By)/ry2;
                for(int x=0;x<img.cols;x++){
                    float ndx=(x-Bx)/rx2;
                    if(ndx*ndx+ndy*ndy>=1.0f)continue;
                    int sx2=(int)(x-Bx+Ax), sy2=(int)(y-By+Ay);
                    if(sx2<0||sy2<0||sx2>=img.cols||sy2>=img.rows)continue;
                    float a=g_seg.mask.at<float>(sy2,sx2);
                    if(a<0.05f){                          /* empty: void glow */
                        ip[x][0]=(uchar)(ip[x][0]*0.35f);
                        ip[x][1]=(uchar)(ip[x][1]*0.30f+15);
                        ip[x][2]=(uchar)(ip[x][2]*0.35f+25);
                        continue;
                    }
                    cv::Vec3b v=img.at<cv::Vec3b>(sy2,sx2);
                    ip[x][0]=(uchar)(ip[x][0]+(v[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(v[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(v[2]-ip[x][2])*a);
                }
            }
            float puls=2.0f+std::sin(tk*0.1f)*1.2f;
            cv::ellipse(img,{(int)Ax,(int)Ay},cv::Size((int)rx2,(int)ry2),0,0,360,
                        cv::Scalar(255,140,30),(int)(4+puls),cv::LINE_AA);
            cv::ellipse(img,{(int)Ax,(int)Ay},cv::Size((int)(rx2*1.06f),(int)(ry2*1.06f)),
                        0,0,360,cv::Scalar(200,80,10),2,cv::LINE_AA);
            cv::ellipse(img,{(int)Bx,(int)By},cv::Size((int)rx2,(int)ry2),0,0,360,
                        cv::Scalar(30,140,255),(int)(4+puls),cv::LINE_AA);
            cv::ellipse(img,{(int)Bx,(int)By},cv::Size((int)(rx2*1.06f),(int)(ry2*1.06f)),
                        0,0,360,cv::Scalar(10,80,200),2,cv::LINE_AA);
            return;
        }
        if (mode == 31) {
            /* MANDALA YOU: your silhouette's contour, kaleidoscoped —
             * N rotated neon copies of your outline spin around your
             * centroid while you stay real in the middle. findContours
             * meets rotation matrices. */
            unsigned tk = g_seg.counter;
            cv::Mat keep = img.clone();
            img *= 0.30;                                   /* dim the room */
            /* restore the real you */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *kp = keep.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(kp[x][0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(kp[x][1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(kp[x][2]-ip[x][2])*a);
                }
            }
            cv::Mat q, q8;
            cv::resize(g_seg.mask, q, cv::Size(), 0.25, 0.25);
            q8 = q > 0.5;
            std::vector<std::vector<cv::Point>> ctrs;
            cv::findContours(q8, ctrs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (!ctrs.empty()) {
                size_t big = 0;
                for (size_t i = 1; i < ctrs.size(); i++)
                    if (ctrs[i].size() > ctrs[big].size()) big = i;
                /* centroid at full res */
                double mx2=0,my2=0;
                for (auto &p : ctrs[big]) { mx2+=p.x; my2+=p.y; }
                float ccx=(float)(mx2/ctrs[big].size())*4, ccy=(float)(my2/ctrs[big].size())*4;
                const int NSYM = 6;
                for (int k2 = 0; k2 < NSYM; k2++) {
                    float ang = k2 * 6.2832f/NSYM + tk * 0.012f;
                    float ca=std::cos(ang), sa=std::sin(ang);
                    float hue = tk*0.03f + k2*1.05f;
                    cv::Scalar col((std::sin(hue+4.19f)*.5f+.5f)*255,
                                   (std::sin(hue+2.09f)*.5f+.5f)*255,
                                   (std::sin(hue)*.5f+.5f)*255);
                    std::vector<cv::Point> rp(ctrs[big].size());
                    for (size_t i = 0; i < ctrs[big].size(); i++) {
                        float px2 = ctrs[big][i].x*4 - ccx;
                        float py2 = ctrs[big][i].y*4 - ccy;
                        rp[i] = cv::Point((int)(ccx + px2*ca - py2*sa),
                                          (int)(ccy + px2*sa + py2*ca));
                    }
                    cv::polylines(img, rp, true, col, 2, cv::LINE_AA);
                }
            }
            return;
        }
        if (mode == 32) {
            /* TOPO AURA: animated topographic contour rings radiate
             * outward from your body — distanceTransform turned into a
             * living radar map. */
            unsigned tk = g_seg.counter;
            /* two-pass chamfer distance transform (3-4 weights), done by
             * hand: cv::distanceTransform's DIST_L2 constant moved in
             * the OpenCV 5 module split, and 20 lines of classic code
             * beat chasing headers across versions. */
            cv::Mat q;
            cv::resize(g_seg.mask, q, cv::Size(), 0.25, 0.25);
            cv::Mat dist(q.size(), CV_32F);
            const float INF2 = 1e9f, W1 = 1.0f, W2 = 1.4142f;
            for (int y = 0; y < q.rows; y++) {
                const float *mm = q.ptr<float>(y);
                float *dp = dist.ptr<float>(y);
                for (int x = 0; x < q.cols; x++)
                    dp[x] = mm[x] > 0.4f ? 0.0f : INF2;
            }
            for (int y = 0; y < q.rows; y++) {            /* forward pass */
                float *dp = dist.ptr<float>(y);
                const float *up = y > 0 ? dist.ptr<float>(y-1) : NULL;
                for (int x = 0; x < q.cols; x++) {
                    float d = dp[x];
                    if (x > 0)        d = std::min(d, dp[x-1] + W1);
                    if (up) {
                        d = std::min(d, up[x] + W1);
                        if (x > 0)          d = std::min(d, up[x-1] + W2);
                        if (x < q.cols - 1) d = std::min(d, up[x+1] + W2);
                    }
                    dp[x] = d;
                }
            }
            for (int y = q.rows - 1; y >= 0; y--) {       /* backward pass */
                float *dp = dist.ptr<float>(y);
                const float *dn = y < q.rows-1 ? dist.ptr<float>(y+1) : NULL;
                for (int x = q.cols - 1; x >= 0; x--) {
                    float d = dp[x];
                    if (x < q.cols - 1) d = std::min(d, dp[x+1] + W1);
                    if (dn) {
                        d = std::min(d, dn[x] + W1);
                        if (x < q.cols - 1) d = std::min(d, dn[x+1] + W2);
                        if (x > 0)          d = std::min(d, dn[x-1] + W2);
                    }
                    dp[x] = d;
                }
            }
            cv::Mat dfull;
            cv::resize(dist, dfull, img.size());
            for (int y = 0; y < img.rows; y++) {
                const float *dd = dfull.ptr<float>(y);
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    if (mm[x] > 0.3f) continue;
                    float d = dd[x] * 4.0f;
                    float band = std::sin(d * 0.16f - tk * 0.14f);
                    if (band < 0.86f) continue;
                    float hue = d * 0.02f;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)((std::sin(hue+4.19f)*.5f+.5f)*150));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)((std::sin(hue+2.09f)*.5f+.5f)*150));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)((std::sin(hue)*.5f+.5f)*150));
                }
            }
            return;
        }
        if (mode == 33) {
            /* BALL PIT: physics balls rain down and BOUNCE OFF YOUR
             * BODY — the matte is the collider, its gradient the
             * surface normal. Bat them with your hands. */
            static float bx[90], by2[90], bvx[90], bvy[90];
            static int bc[90], binit = 0;
            unsigned tk = g_seg.counter;
            if (!binit) {
                for (int i = 0; i < 90; i++) {
                    bx[i]=uhash(i,1,7)* (img.cols-20)+10;
                    by2[i]=-uhash(i,2,7)*img.rows*2;
                    bvx[i]=(uhash(i,3,7)-0.5f)*3; bvy[i]=0;
                    bc[i]=i%6;
                }
                binit = 1;
            }
            static const int BC[6][3]={{60,60,235},{60,200,255},{80,220,80},
                {235,180,60},{230,90,200},{250,250,250}};
            auto mask_at = [&](float X, float Y)->float{
                int xi=(int)X, yi=(int)Y;
                if (xi<0||yi<0||xi>=img.cols||yi>=img.rows) return 0.f;
                return g_seg.mask.at<float>(yi,xi);
            };
            const int R = std::max(6, img.cols/90);
            for (int i = 0; i < 90; i++) {
                bvy[i] += 0.32f * fx;                     /* gravity */
                bvx[i] *= 0.999f;
                float nx2 = bx[i]+bvx[i], ny2 = by2[i]+bvy[i];
                if (nx2 < R)            { nx2=(float)R; bvx[i]=std::fabs(bvx[i])*0.8f; }
                if (nx2 > img.cols-R)   { nx2=(float)(img.cols-R); bvx[i]=-std::fabs(bvx[i])*0.8f; }
                if (ny2 > img.rows+R*4) {                 /* fell out: respawn */
                    nx2=uhash(i,tk,1)*(img.cols-20)+10; ny2=-(float)R*2;
                    bvx[i]=(uhash(i,tk,2)-0.5f)*3; bvy[i]=0;
                }
                if (mask_at(nx2, ny2+R) > 0.5f) {         /* hit YOU */
                    /* surface normal from mask gradient */
                    float gx2 = mask_at(nx2+4,ny2) - mask_at(nx2-4,ny2);
                    float gy2 = mask_at(nx2,ny2+4) - mask_at(nx2,ny2-4);
                    float gl = std::sqrt(gx2*gx2+gy2*gy2);
                    float nxn = gl>1e-4f ? -gx2/gl : 0.f;
                    float nyn = gl>1e-4f ? -gy2/gl : -1.f;
                    float dot2 = bvx[i]*nxn + bvy[i]*nyn;
                    bvx[i] = (bvx[i] - 2*dot2*nxn) * 0.72f;
                    bvy[i] = (bvy[i] - 2*dot2*nyn) * 0.72f - 0.2f;
                    for (int st = 0; st < 8 && mask_at(nx2, ny2+R) > 0.5f; st++) {
                        nx2 += nxn*3; ny2 += nyn*3;
                    }
                }
                bx[i]=nx2; by2[i]=ny2;
                int X=(int)bx[i], Y=(int)by2[i];
                if (Y < -R || Y >= img.rows+R) continue;
                const int *cc = BC[bc[i]];
                cv::circle(img, {X,Y}, R, cv::Scalar(cc[0],cc[1],cc[2]), cv::FILLED);
                cv::circle(img, {X,Y}, R, cv::Scalar(cc[0]/2,cc[1]/2,cc[2]/2), 1);
                cv::circle(img, {X-R/3,Y-R/3}, std::max(1,R/3),
                           cv::Scalar(255,255,255), cv::FILLED);
            }
            return;
        }
        if (mode == 34) {
            /* ORBIT RIM LIGHT: a virtual studio light circles you —
             * your silhouette's edge catches it exactly where physics
             * says it should (mask gradient = surface normal). */
            unsigned tk = g_seg.counter;
            float la = tk * 0.03f;
            float ldx = std::cos(la), ldy = std::sin(la) * 0.5f;
            cv::Mat gx2, gy2;
            cv::Sobel(g_seg.mask, gx2, CV_32F, 1, 0, 3);
            cv::Sobel(g_seg.mask, gy2, CV_32F, 0, 1, 3);
            /* light color drifts warm<->cool with the orbit */
            float lr = 200 + 55*std::cos(la), lg = 180, lb = 200 - 55*std::cos(la);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const float *px2 = gx2.ptr<float>(y);
                const float *py2 = gy2.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float m2 = mm[x];
                    if (m2 < 0.04f || m2 > 0.97f) continue;   /* edge band only */
                    float gl = std::sqrt(px2[x]*px2[x] + py2[x]*py2[x]);
                    if (gl < 0.02f) continue;
                    float rim = (-px2[x]*ldx - py2[x]*ldy) / gl;  /* facing the light */
                    if (rim <= 0) continue;
                    float k = rim * rim * 0.9f;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)(lb*k));
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)(lg*k));
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)(lr*k));
                }
            }
            return;
        }
        if (mode == 35) {
            /* MIND VORTEX: the room spirals into your head — a polar
             * swirl remap warps everything EXCEPT you. */
            unsigned tk = g_seg.counter;
            /* head anchor: topmost band of the matte's centroid */
            static float vcx = -1, vcy = -1;
            double sx=0, sy=0, sw2=0;
            for (int y = 0; y < img.rows/2; y += 6) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 6)
                    if (mm[x] > 0.4f) { sx+=x; sy+=y; sw2+=1; }
            }
            float tx2 = sw2>4 ? (float)(sx/sw2) : img.cols/2.f;
            float ty2 = sw2>4 ? (float)(sy/sw2) : img.rows/3.f;
            if (vcx < 0) { vcx=tx2; vcy=ty2; }
            vcx += (tx2-vcx)*0.05f; vcy += (ty2-vcy)*0.05f;
            static cv::Mat mpx, mpy;
            if (mpx.size() != img.size()) {
                mpx.create(img.size(), CV_32F);
                mpy.create(img.size(), CV_32F);
            }
            float rmax = std::sqrt((float)(img.cols*img.cols+img.rows*img.rows))*0.5f;
            for (int y = 0; y < img.rows; y++) {
                float *ax = mpx.ptr<float>(y);
                float *ay = mpy.ptr<float>(y);
                float dy = y - vcy;
                for (int x = 0; x < img.cols; x++) {
                    float dx = x - vcx;
                    float r = std::sqrt(dx*dx + dy*dy);
                    float sw3 = (1.0f - r/rmax);
                    float ang = sw3*sw3 * 2.4f + tk*0.008f;
                    float ca = std::cos(ang), sa = std::sin(ang);
                    ax[x] = vcx + dx*ca - dy*sa;
                    ay[x] = vcy + dx*sa + dy*ca;
                }
            }
            cv::Mat keep = img.clone(), warped;
            cv::remap(img, warped, mpx, mpy, cv::INTER_LINEAR, cv::BORDER_REFLECT);
            warped.copyTo(img);
            for (int y = 0; y < img.rows; y++) {          /* you stay solid */
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *kp = keep.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(kp[x][0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(kp[x][1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(kp[x][2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 36) {
            /* STAINED GLASS: the frame becomes a leaded glass window —
             * Voronoi facets (Subdiv2D), denser and brighter where you
             * are, with dark lead lines between panes. */
            /* jittered-grid Voronoi, hand-rolled: cv::Subdiv2D moved in
             * the OpenCV 5 module split. One seed per grid cell means a
             * pixel's nearest seed is always within its 3x3 cell
             * neighborhood — exact Voronoi, no library, and faster. */
            const int GX = 22, GY = 13;
            float sxp[GY][GX], syp[GY][GX];
            uchar scol[GY][GX][3];
            float cw2 = (float)img.cols / GX, ch2 = (float)img.rows / GY;
            for (int gy = 0; gy < GY; gy++)
                for (int gx = 0; gx < GX; gx++) {
                    sxp[gy][gx] = (gx + 0.2f + 0.6f*uhash(gx, gy, 11)) * cw2;
                    syp[gy][gx] = (gy + 0.2f + 0.6f*uhash(gx, gy, 47)) * ch2;
                    int px2 = std::min(img.cols-1, (int)sxp[gy][gx]);
                    int py2 = std::min(img.rows-1, (int)syp[gy][gx]);
                    cv::Vec3b c = img.at<cv::Vec3b>(py2, px2);
                    float onp = g_seg.mask.at<float>(py2, px2);
                    float sat = 1.15f + 0.35f * onp;      /* person glows more */
                    scol[gy][gx][0]=(uchar)std::min(255.f, c[0]*sat);
                    scol[gy][gx][1]=(uchar)std::min(255.f, c[1]*sat);
                    scol[gy][gx][2]=(uchar)std::min(255.f, c[2]*sat);
                }
            /* ownership at quarter res, then paint full res with lead
             * lines where the owner changes */
            int QW = img.cols/2, QH = img.rows/2;
            cv::Mat own(QH, QW, CV_16U);
            for (int y = 0; y < QH; y++) {
                unsigned short *op = own.ptr<unsigned short>(y);
                float fy = y * 2.0f;
                int gy0 = std::min(GY-1, (int)(fy / ch2));
                for (int x = 0; x < QW; x++) {
                    float fx = x * 2.0f;
                    int gx0 = std::min(GX-1, (int)(fx / cw2));
                    float bd = 1e18f; int best = 0;
                    for (int dy2 = -1; dy2 <= 1; dy2++)
                        for (int dx2 = -1; dx2 <= 1; dx2++) {
                            int gy2 = gy0+dy2, gx2 = gx0+dx2;
                            if (gy2 < 0 || gy2 >= GY || gx2 < 0 || gx2 >= GX) continue;
                            float ddx = fx - sxp[gy2][gx2];
                            float ddy = fy - syp[gy2][gx2];
                            float d2 = ddx*ddx + ddy*ddy;
                            if (d2 < bd) { bd = d2; best = gy2*GX + gx2; }
                        }
                    op[x] = (unsigned short)best;
                }
            }
            for (int y = 0; y < img.rows; y++) {
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                int qy = std::min(QH-1, y/2);
                const unsigned short *op = own.ptr<unsigned short>(qy);
                const unsigned short *od = own.ptr<unsigned short>(std::min(QH-1, qy+1));
                for (int x = 0; x < img.cols; x++) {
                    int qx = std::min(QW-1, x/2);
                    unsigned short o = op[qx];
                    /* lead line where the pane changes */
                    if ((qx+1 < QW && op[qx+1] != o) || od[qx] != o) {
                        ip[x] = cv::Vec3b(25, 20, 20);
                        continue;
                    }
                    ip[x] = cv::Vec3b(scol[o/GX][o%GX][0],
                                      scol[o/GX][o%GX][1],
                                      scol[o/GX][o%GX][2]);
                }
            }
            return;
        }
        if (mode == 37) {
            /* UPSIDE-DOWN ROOM: the entire world flips; you don't. */
            cv::Mat keep = img.clone();
            cv::flip(keep, img, 0);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *kp = keep.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    ip[x][0]=(uchar)(ip[x][0]+(kp[x][0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(kp[x][1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(kp[x][2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 25) {
            /* STAGE SPOTLIGHT: a theater beam follows you from above;
             * the rest of the room falls dark; dust motes drift in the
             * light. */
            unsigned tk = g_seg.counter;
            /* track your centroid cheaply (coarse mask scan) */
            static float scx = -1;
            double sx = 0, sw2 = 0;
            for (int y = 0; y < img.rows; y += 8) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 8)
                    if (mm[x] > 0.4f) { sx += x; sw2 += 1; }
            }
            float tcx = sw2 > 4 ? (float)(sx / sw2) : img.cols / 2.f;
            if (scx < 0) scx = tcx;
            scx += (tcx - scx) * 0.06f;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                float half = img.cols * 0.05f + y * 0.22f;   /* cone widens */
                for (int x = 0; x < img.cols; x++) {
                    float d = std::fabs(x - scx) / half;      /* 0 center */
                    float lit = d < 0.8f ? 1.0f : d < 1.15f ? (1.15f - d)/0.35f : 0.0f;
                    float k = std::max(lit, std::min(1.0f, mm[x] * 1.2f));
                    float dim = 0.10f + 0.90f * k;
                    ip[x][0]=(uchar)(ip[x][0]*dim);
                    ip[x][1]=(uchar)(ip[x][1]*dim);
                    ip[x][2]=(uchar)std::min(255.f, ip[x][2]*dim*(1.0f+0.08f*lit));
                    if (lit > 0.4f && mm[x] < 0.2f &&
                        uhash(x/2, y/2, tk/2) > 0.9975f) {   /* dust motes */
                        ip[x][0]=clamp_int_u8(ip[x][0]+110);
                        ip[x][1]=clamp_int_u8(ip[x][1]+110);
                        ip[x][2]=clamp_int_u8(ip[x][2]+110);
                    }
                }
            }
            return;
        }
        if (mode == 26) {
            /* HAUNTED SHADOW: your shadow on the wall — but it's your
             * silhouette from a second AGO, sheared sideways. It lags
             * you, catches up, and feels deeply wrong. */
            static cv::Mat mring[32];
            static int mri = 0, mqw = 0, mqh = 0;
            int QW = img.cols/4, QH = img.rows/4;
            if (mqw != QW || mqh != QH) {
                for (int i = 0; i < 32; i++) mring[i].release();
                mqw = QW; mqh = QH; mri = 0;
            }
            cv::Mat qm;
            cv::resize(g_seg.mask, qm, cv::Size(QW, QH));
            qm.copyTo(mring[mri]);
            int gi = (mri - 28 + 32) % 32;                /* ~1 s ago */
            mri = (mri + 1) % 32;
            if (!mring[gi].empty()) {
                cv::Mat old;
                cv::resize(mring[gi], old, img.size());
                for (int y = 0; y < img.rows; y++) {
                    const float *cm = g_seg.mask.ptr<float>(y);
                    cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < img.cols; x++) {
                        if (cm[x] > 0.25f) continue;       /* not on you */
                        int sx2 = x + (int)((img.rows - y) * 0.30f) - img.cols/12;
                        if (sx2 < 0 || sx2 >= img.cols) continue;
                        float a = old.at<float>(y, sx2);
                        if (a < 0.25f) continue;
                        float k = 0.72f * std::min(1.0f, a);
                        ip[x][0]=(uchar)(ip[x][0]*(1-k));
                        ip[x][1]=(uchar)(ip[x][1]*(1-k));
                        ip[x][2]=(uchar)(ip[x][2]*(1-k));
                    }
                }
            }
            return;
        }
        if (mode == 27) {
            /* COSMIC YOU: your silhouette becomes a window into deep
             * space — an animated starfield lives inside your body. */
            static float stx[220], sty[220], stz[220];
            static int sinit = 0;
            if (!sinit) {
                for (int i = 0; i < 220; i++) {
                    stx[i]=uhash(i,1,0); sty[i]=uhash(i,2,0);
                    stz[i]=0.2f+uhash(i,3,0)*0.8f;
                }
                sinit = 1;
            }
            for (int y = 0; y < img.rows; y++) {          /* body -> space */
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a < 0.05f) continue;
                    cv::Vec3b space(40 + (y*20/img.rows), 8, 18);
                    ip[x][0]=(uchar)(ip[x][0]+(space[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(space[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(space[2]-ip[x][2])*a);
                }
            }
            for (int i = 0; i < 220; i++) {               /* stars drift */
                stx[i] += 0.0016f * stz[i];
                if (stx[i] > 1.0f) stx[i] -= 1.0f;
                int X = (int)(stx[i]*img.cols), Y = (int)(sty[i]*img.rows);
                if (X < 1 || X >= img.cols-1 || Y < 1 || Y >= img.rows-1) continue;
                if (g_seg.mask.at<float>(Y, X) < 0.5f) continue;
                int br = (int)(120 + stz[i]*135);
                cv::Vec3b *p2 = img.ptr<cv::Vec3b>(Y) + X;
                p2[0][0]=clamp_int_u8(br); p2[0][1]=clamp_int_u8(br);
                p2[0][2]=clamp_int_u8(br - 20);
            }
            return;
        }
        if (mode == 28) {
            /* PAPER STICKER: thick white cut-out border around you plus
             * a drop shadow — you become a stop-motion sticker. */
            cv::Mat m8, dil, dil2;
            g_seg.mask.convertTo(m8, CV_8U, 255.0);
            cv::dilate(m8, dil,  cv::Mat(), cv::Point(-1,-1), 4);
            cv::dilate(m8, dil2, cv::Mat(), cv::Point(-1,-1), 8);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const uchar *d1 = dil.ptr<uchar>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    if (mm[x] > 0.5f) continue;
                    if (d1[x] > 128) {                    /* sticker border */
                        ip[x] = cv::Vec3b(245, 245, 245);
                    } else {
                        /* drop shadow: border mask sampled up-left */
                        int sx2 = x - 10, sy2 = y - 10;
                        if (sx2 >= 0 && sy2 >= 0 &&
                            dil2.at<uchar>(sy2, sx2) > 128) {
                            ip[x][0]=(uchar)(ip[x][0]*0.55f);
                            ip[x][1]=(uchar)(ip[x][1]*0.55f);
                            ip[x][2]=(uchar)(ip[x][2]*0.55f);
                        } else {                          /* calm room */
                            ip[x][0]=(uchar)(ip[x][0]*0.9f + 12);
                            ip[x][1]=(uchar)(ip[x][1]*0.9f + 12);
                            ip[x][2]=(uchar)(ip[x][2]*0.9f + 12);
                        }
                    }
                }
            }
            return;
        }
        if (mode == 29) {
            /* HEAT SIGNATURE: predator-cam, but honest — only YOU emit
             * heat (luma -> thermal palette); the room is cold steel. */
            static const int TP[6][3] = {   /* B,G,R stops cold->hot */
                {80,10,0},{160,40,0},{120,160,0},{40,220,255},{20,120,255},{240,255,255}};
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float lum=(ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    float a = mm[x];
                    if (a < 0.05f) {                      /* cold room */
                        uchar v=(uchar)(lum*70);
                        ip[x]=cv::Vec3b((uchar)(v+25),v,(uchar)(v*0.8f));
                        continue;
                    }
                    float t2 = std::min(0.999f, lum) * 5.0f;
                    int i0 = (int)t2; float fr = t2 - i0;
                    cv::Vec3b hot(
                        (uchar)(TP[i0][0]+(TP[i0+1][0]-TP[i0][0])*fr),
                        (uchar)(TP[i0][1]+(TP[i0+1][1]-TP[i0][1])*fr),
                        (uchar)(TP[i0][2]+(TP[i0+1][2]-TP[i0][2])*fr));
                    ip[x][0]=(uchar)(ip[x][0]+(hot[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(hot[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(hot[2]-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 30) {
            /* AFTERIMAGE BURST: move fast and sharp snapshots of you
             * freeze in the air where you were, fading out. Triggered
             * by your own centroid velocity — dodge like an anime
             * character. */
            static cv::Mat snap_rgb[3], snap_a[3];
            static float snap_life[3] = {0,0,0};
            static float pcx = -1, pcy = -1;
            unsigned tk = g_seg.counter;
            double sx=0, sy=0, sw2=0;
            for (int y = 0; y < img.rows; y += 8) {
                const float *mm = g_seg.mask.ptr<float>(y);
                for (int x = 0; x < img.cols; x += 8)
                    if (mm[x] > 0.4f) { sx += x; sy += y; sw2 += 1; }
            }
            if (sw2 > 4) {
                float cx2=(float)(sx/sw2), cy2=(float)(sy/sw2);
                if (pcx >= 0) {
                    float v = std::sqrt((cx2-pcx)*(cx2-pcx)+(cy2-pcy)*(cy2-pcy));
                    if (v > img.cols * 0.012f && (tk % 4) == 0) {
                        /* fast! freeze a snapshot into the oldest slot */
                        int slot = 0;
                        for (int i2=1;i2<3;i2++) if (snap_life[i2]<snap_life[slot]) slot=i2;
                        cv::resize(img, snap_rgb[slot],
                                   cv::Size(img.cols/2, img.rows/2), 0,0, cv::INTER_AREA);
                        cv::resize(g_seg.mask, snap_a[slot],
                                   cv::Size(img.cols/2, img.rows/2));
                        snap_life[slot] = 1.0f;
                    }
                }
                pcx = cx2; pcy = cy2;
            }
            for (int s2 = 0; s2 < 3; s2++) {
                if (snap_life[s2] <= 0) continue;
                snap_life[s2] -= 0.03f;
                cv::Mat gr, ga;
                cv::resize(snap_rgb[s2], gr, img.size(), 0,0, cv::INTER_LINEAR);
                cv::resize(snap_a[s2],  ga, img.size(), 0,0, cv::INTER_LINEAR);
                float li = snap_life[s2] * 0.6f;
                for (int y = 0; y < img.rows; y++) {
                    const float *cm = g_seg.mask.ptr<float>(y);
                    const float *am = ga.ptr<float>(y);
                    const cv::Vec3b *gp = gr.ptr<cv::Vec3b>(y);
                    cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < img.cols; x++) {
                        float a = am[x] * li * (1.0f - cm[x]);
                        if (a < 0.04f) continue;
                        ip[x][0]=(uchar)(ip[x][0]+(gp[x][0]-ip[x][0])*a);
                        ip[x][1]=(uchar)(ip[x][1]+(gp[x][1]-ip[x][1])*a);
                        ip[x][2]=(uchar)(ip[x][2]+(gp[x][2]-ip[x][2])*a);
                    }
                }
            }
            return;
        }
        if (mode == 17) {
            /* GLITCH BODY: only YOU get datamoshed — per-band horizontal
             * tearing + RGB channel split; the room stays pristine. */
            unsigned tk = g_seg.counter;
            cv::Mat srcimg = img.clone();
            for (int y = 0; y < img.rows; y++) {
                int band = y / 6;
                float n = uhash(band, 0, tk / 3);
                int off = (n < 0.75f) ? 0 : (int)((n - 0.75f) * 4 * 30 * fx) - (int)(15*fx);
                int coff = 2 + (int)(uhash(band, 7, tk / 3) * 4);
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *sp = srcimg.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    int sx = x - off;
                    if (sx < 0 || sx >= img.cols) continue;
                    float a = mm[sx];
                    if (a < 0.05f) continue;
                    int xr = std::min(img.cols - 1, sx + coff);
                    int xb = std::max(0, sx - coff);
                    cv::Vec3b v(sp[xb][0], sp[sx][1], sp[xr][2]);
                    ip[x][0] = (uchar)(ip[x][0] + (v[0] - ip[x][0]) * a);
                    ip[x][1] = (uchar)(ip[x][1] + (v[1] - ip[x][1]) * a);
                    ip[x][2] = (uchar)(ip[x][2] + (v[2] - ip[x][2]) * a);
                }
            }
            return;
        }
        if (mode == 18) {
            /* SLICED PORTRAIT: you split into horizontal slabs that
             * slide apart in alternating directions, gallery-art style,
             * over the learned room. */
            unsigned tk = g_seg.counter;
            cv::Mat out2; g_seg.bg_est.copyTo(out2);
            int nsl = 9, sh2 = (img.rows + nsl - 1) / nsl;
            for (int y = 0; y < img.rows; y++) {
                int sl = y / sh2;
                float ph = std::sin(tk * 0.045f + sl * 1.7f);
                int off = (int)(ph * img.cols * 0.06f) * ((sl & 1) ? -1 : 1);
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                cv::Vec3b *op = out2.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    int sx = x - off;
                    if (sx < 0 || sx >= img.cols) continue;
                    float a = mm[sx];
                    if (a < 0.05f) continue;
                    op[x][0]=(uchar)(op[x][0]+(ip[sx][0]-op[x][0])*a);
                    op[x][1]=(uchar)(op[x][1]+(ip[sx][1]-op[x][1])*a);
                    op[x][2]=(uchar)(op[x][2]+(ip[sx][2]-op[x][2])*a);
                }
            }
            out2.copyTo(img);
            return;
        }
        if (mode == 19) {
            /* DISINTEGRATION: you dissolve into drifting embers and
             * re-form, on a slow loop. Noise decides which pixels go
             * first; freed pixels reveal the learned room; embers spawn
             * at the dissolve frontier and drift up. */
            static float epx[160], epy[160], evx[160], evy[160], elife[160];
            static int einit = 0;
            if (!einit) { for (int i=0;i<160;i++) elife[i]=0; einit=1; }
            unsigned tk = g_seg.counter;
            float phase = 0.5f + 0.5f * std::sin(tk * 0.016f);   /* 0..1 loop */
            int spawned = 0;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *bp = g_seg.bg_est.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a < 0.05f) continue;
                    float n = uhash(x / 3, y / 3, 0);      /* static pattern */
                    float d = (phase - n) * 8.0f * fx;     /* dissolve depth */
                    if (d <= 0) continue;                  /* still solid */
                    float k = std::min(1.0f, d) * a;
                    /* at the frontier: spawn an ember occasionally */
                    if (d < 0.6f && spawned < 6 && uhash(x, y, tk) > 0.995f) {
                        for (int e = 0; e < 160; e++) if (elife[e] <= 0) {
                            epx[e]=(float)x; epy[e]=(float)y;
                            evx[e]=(uhash(x,y,tk+1)-0.5f)*2.0f;
                            evy[e]=-1.0f-uhash(x,y,tk+2)*2.0f;
                            elife[e]=1.0f; spawned++; break;
                        }
                    }
                    ip[x][0]=(uchar)(ip[x][0]+(bp[x][0]-ip[x][0])*k);
                    ip[x][1]=(uchar)(ip[x][1]+(bp[x][1]-ip[x][1])*k);
                    ip[x][2]=(uchar)(ip[x][2]+(bp[x][2]-ip[x][2])*k);
                }
            }
            for (int e = 0; e < 160; e++) {                /* embers */
                if (elife[e] <= 0) continue;
                epx[e]+=evx[e]; epy[e]+=evy[e]; elife[e]-=0.02f;
                int X=(int)epx[e], Y=(int)epy[e];
                if (X<1||X>=img.cols-2||Y<1||Y>=img.rows-2){elife[e]=0;continue;}
                int br=(int)(elife[e]*255);
                for(int yy=0;yy<2;yy++){uchar*p=data+(Y+yy)*stride+X*3;
                    for(int xx=0;xx<2;xx++){
                        p[xx*3+0]=clamp_int_u8(p[xx*3+0]+br/3);
                        p[xx*3+1]=clamp_int_u8(p[xx*3+1]+br*2/3);
                        p[xx*3+2]=clamp_int_u8(p[xx*3+2]+br);}}
            }
            return;
        }
        if (mode == 20) {
            /* LIGHT-PAINT SILHOUETTE: your outline writes glowing,
             * hue-cycling light trails that linger like a long-exposure
             * photograph. Move to paint; hold still to let it fade. */
            static cv::Mat trail;
            if (trail.empty() || trail.size() != img.size())
                trail = cv::Mat::zeros(img.size(), CV_32FC3);
            trail *= 0.94;
            unsigned tk = g_seg.counter;
            cv::Mat m8, d1, e1;
            g_seg.mask.convertTo(m8, CV_8U, 255.0);
            cv::dilate(m8, d1, cv::Mat(), cv::Point(-1,-1), 2);
            cv::erode (m8, e1, cv::Mat(), cv::Point(-1,-1), 1);
            float hue = tk * 0.05f;
            float cb2=(std::sin(hue+4.19f)*0.5f+0.5f)*255;
            float cg2=(std::sin(hue+2.09f)*0.5f+0.5f)*255;
            float cr2=(std::sin(hue)*0.5f+0.5f)*255;
            for (int y = 0; y < img.rows; y++) {
                const uchar *da=d1.ptr<uchar>(y), *ea=e1.ptr<uchar>(y);
                cv::Vec3f *tp = trail.ptr<cv::Vec3f>(y);
                for (int x = 0; x < img.cols; x++)
                    if (da[x] > 128 && ea[x] < 128) {
                        tp[x][0]=std::max(tp[x][0],cb2);
                        tp[x][1]=std::max(tp[x][1],cg2);
                        tp[x][2]=std::max(tp[x][2],cr2);
                    }
            }
            for (int y = 0; y < img.rows; y++) {
                const cv::Vec3f *tp = trail.ptr<cv::Vec3f>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    if (tp[x][0]+tp[x][1]+tp[x][2] < 12) continue;
                    ip[x][0]=clamp_int_u8(ip[x][0]+(int)tp[x][0]);
                    ip[x][1]=clamp_int_u8(ip[x][1]+(int)tp[x][1]);
                    ip[x][2]=clamp_int_u8(ip[x][2]+(int)tp[x][2]);
                }
            }
            return;
        }
        if (mode == 21) {
            /* MIRROR TWIN: you and your horizontally-flipped double,
             * side by side in the learned room. Wave — it waves back
             * with the other hand. */
            cv::Mat out2; g_seg.bg_est.copyTo(out2);
            int dx = (int)(img.cols / 5 * fx);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                cv::Vec3b *op = out2.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    /* original, shifted left */
                    int sx = x + dx;
                    if (sx >= 0 && sx < img.cols && mm[sx] > 0.05f) {
                        float a = mm[sx];
                        op[x][0]=(uchar)(op[x][0]+(ip[sx][0]-op[x][0])*a);
                        op[x][1]=(uchar)(op[x][1]+(ip[sx][1]-op[x][1])*a);
                        op[x][2]=(uchar)(op[x][2]+(ip[sx][2]-op[x][2])*a);
                    }
                    /* mirrored twin, shifted right */
                    int mx = (img.cols - 1 - x) + dx;
                    if (mx >= 0 && mx < img.cols && mm[mx] > 0.05f) {
                        float a = mm[mx] * 0.97f;
                        op[x][0]=(uchar)(op[x][0]+(ip[mx][0]-op[x][0])*a);
                        op[x][1]=(uchar)(op[x][1]+(ip[mx][1]-op[x][1])*a);
                        op[x][2]=(uchar)(op[x][2]+(ip[mx][2]-op[x][2])*a);
                    }
                }
            }
            out2.copyTo(img);
            return;
        }
        if (mode == 22) {
            /* CRYSTAL YOU: your body rendered as a low-poly triangle
             * mesh (each cell split into two flat-shaded triangles);
             * the room stays photographic. */
            const int C = std::max(16, img.cols / 40);
            for (int cy = 0; cy < img.rows - 1; cy += C) {
                for (int cx = 0; cx < img.cols - 1; cx += C) {
                    int x1 = std::min(img.cols - 1, cx + C);
                    int y1 = std::min(img.rows - 1, cy + C);
                    /* cell person coverage */
                    float acc = g_seg.mask.at<float>(cy, cx)
                              + g_seg.mask.at<float>(cy, x1)
                              + g_seg.mask.at<float>(y1, cx)
                              + g_seg.mask.at<float>(y1, x1);
                    if (acc < 0.8f) continue;
                    cv::Vec3b ca = img.at<cv::Vec3b>((cy+y1)/2, (cx*2+x1)/3);
                    cv::Vec3b cb = img.at<cv::Vec3b>((cy+y1)/2, (cx+x1*2)/3);
                    cv::Point t1[3] = {{cx,cy},{x1,cy},{cx,y1}};
                    cv::Point t2[3] = {{x1,cy},{x1,y1},{cx,y1}};
                    cv::fillConvexPoly(img, t1, 3, cv::Scalar(ca[0],ca[1],ca[2]));
                    cv::fillConvexPoly(img, t2, 3, cv::Scalar(cb[0],cb[1],cb[2]));
                    cv::line(img, t1[1], t1[2], cv::Scalar(255,255,255), 1);
                }
            }
            return;
        }
        if (mode == 23) {
            /* X-RAY SCANNER: a bright bar sweeps down; behind it your
             * body shows in inverted blue x-ray while the room dims. */
            unsigned tk = g_seg.counter;
            int bar = (int)((tk * 5) % (unsigned)(img.rows + 120)) - 60;
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                if (y > bar - 4 && y < bar + 4) {          /* the sweep bar */
                    for (int x = 0; x < img.cols; x++) {
                        ip[x][0]=clamp_int_u8(ip[x][0]+180);
                        ip[x][1]=clamp_int_u8(ip[x][1]+200);
                        ip[x][2]=clamp_int_u8(ip[x][2]+120);
                    }
                } else if (y < bar) {                       /* scanned zone */
                    for (int x = 0; x < img.cols; x++) {
                        float a = mm[x];
                        if (a < 0.05f) {                    /* room dims */
                            ip[x][0]=(uchar)(ip[x][0]*0.45f);
                            ip[x][1]=(uchar)(ip[x][1]*0.40f);
                            ip[x][2]=(uchar)(ip[x][2]*0.35f);
                            continue;
                        }
                        float lum=(ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f);
                        float inv2=255.0f-lum;
                        float b2=std::min(255.f,inv2*1.25f), gg=inv2*0.9f, r2=inv2*0.45f;
                        ip[x][0]=(uchar)(ip[x][0]+(b2-ip[x][0])*a);
                        ip[x][1]=(uchar)(ip[x][1]+(gg-ip[x][1])*a);
                        ip[x][2]=(uchar)(ip[x][2]+(r2-ip[x][2])*a);
                    }
                }
            }
            return;
        }
        if (mode == 24) {
            /* WANTED POSTER: sepia room, you in gritty halftone,
             * WANTED above your head and the bounty below. */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float lum=(ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f);
                    float a = mm[x];
                    if (a > 0.5f) {                        /* halftone person */
                        float cell = ((x % 4) < 2) ^ ((y % 4) < 2) ? 30.f : 0.f;
                        float v = (lum > 128 + cell - 15) ? 225.f : 35.f;
                        ip[x] = cv::Vec3b((uchar)(v*0.55f),(uchar)(v*0.75f),(uchar)v);
                    } else {                               /* sepia room */
                        ip[x][0]=(uchar)std::min(255.f,lum*0.55f+20);
                        ip[x][1]=(uchar)std::min(255.f,lum*0.78f+22);
                        ip[x][2]=(uchar)std::min(255.f,lum*0.95f+30);
                    }
                }
            }
            cv::putText(img, "WANTED", cv::Point(img.cols/2-170, 70),
                        cv::FONT_HERSHEY_TRIPLEX, 2.2, cv::Scalar(15,40,80), 8);
            cv::putText(img, "WANTED", cv::Point(img.cols/2-170, 70),
                        cv::FONT_HERSHEY_TRIPLEX, 2.2, cv::Scalar(60,120,200), 3);
            cv::putText(img, "REWARD  $5000", cv::Point(img.cols/2-150, img.rows-30),
                        cv::FONT_HERSHEY_TRIPLEX, 1.2, cv::Scalar(15,40,80), 5);
            cv::putText(img, "REWARD  $5000", cv::Point(img.cols/2-150, img.rows-30),
                        cv::FONT_HERSHEY_TRIPLEX, 1.2, cv::Scalar(60,120,200), 2);
            return;
        }
        if (mode == 15) {
            /* MOTION ECHO: translucent copies of your silhouette shed
             * behind you as you move — a comet-trail of past poses held
             * in a decaying accumulator, composited under the live you. */
            static cv::Mat accum;
            if (accum.empty() || accum.size() != img.size())
                accum = cv::Mat::zeros(img.size(), CV_32FC3);
            /* decay the trail */
            accum *= 0.88;
            /* stamp the current matted person into the accumulator */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                cv::Vec3f *ap = accum.ptr<cv::Vec3f>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    ap[x][0] = std::max(ap[x][0], ip[x][0]*a);
                    ap[x][1] = std::max(ap[x][1], ip[x][1]*a);
                    ap[x][2] = std::max(ap[x][2], ip[x][2]*a);
                }
            }
            /* draw trail where the live person ISN'T, tinted cyan */
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3f *ap = accum.ptr<cv::Vec3f>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float t = (1.0f - mm[x]);
                    float lum = (ap[x][0]+ap[x][1]+ap[x][2]) / 3.0f;
                    if (lum < 8) continue;
                    float a = std::min(0.75f, lum/180.0f) * t;
                    ip[x][0]=(uchar)(ip[x][0]+(std::min(255.f,ap[x][0]*1.3f)-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(std::min(255.f,ap[x][1]*1.1f)-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(ap[x][2]*0.6f-ip[x][2])*a);
                }
            }
            return;
        }
        if (mode == 16) {
            /* DOUBLE EXPOSURE: the classic portrait effect — your
             * silhouette is filled with the room behind you (the learned
             * background shows THROUGH your body), edges kept dark. */
            if (g_seg.bg_est.empty()) { }
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *bp = g_seg.bg_est.empty() ? nullptr
                                      : g_seg.bg_est.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x];
                    if (a < 0.05f) { ip[x] = cv::Vec3b(20,15,15); continue; }
                    /* inside the person: blend the body luma with the
                     * background image for a dreamy double-exposure */
                    float lum = (ip[x][0]*0.114f+ip[x][1]*0.587f+ip[x][2]*0.299f)/255.f;
                    cv::Vec3b bgv = bp ? bp[x] : cv::Vec3b(60,40,80);
                    ip[x][0]=(uchar)(bgv[0]*lum + 30*(1-lum));
                    ip[x][1]=(uchar)(bgv[1]*lum + 20*(1-lum));
                    ip[x][2]=(uchar)(bgv[2]*lum + 40*(1-lum));
                }
            }
            return;
        }
        if (mode == 14) {
            /* GRAVITY FLIP: you, upside down in the learned room */
            cv::Mat out2; g_seg.bg_est.copyTo(out2);
            for(int y=0;y<img.rows;y++){
                int sy2=img.rows-1-y;
                const float *mm=g_seg.mask.ptr<float>(sy2);
                const cv::Vec3b *ip=img.ptr<cv::Vec3b>(sy2);
                cv::Vec3b *op=out2.ptr<cv::Vec3b>(y);
                for(int x=0;x<img.cols;x++){
                    float a=mm[x]; if(a<0.05f) continue;
                    op[x][0]=(uchar)(op[x][0]+(ip[x][0]-op[x][0])*a);
                    op[x][1]=(uchar)(op[x][1]+(ip[x][1]-op[x][1])*a);
                    op[x][2]=(uchar)(op[x][2]+(ip[x][2]-op[x][2])*a);
                }
            }
            out2.copyTo(img);
            return;
        }
        if (mode == 9) {
            /* TINY YOU: you, shrunk to 45%, standing at the bottom of
             * the learned room like a desk gnome. */
            cv::Mat out2; g_seg.bg_est.copyTo(out2);
            int sw2 = (int)(img.cols*0.45), sh2 = (int)(img.rows*0.45);
            cv::Mat si, sm;
            cv::resize(img, si, cv::Size(sw2, sh2), 0, 0, cv::INTER_AREA);
            cv::resize(g_seg.mask, sm, cv::Size(sw2, sh2));
            int ox2 = (img.cols - sw2)/2, oy2 = img.rows - sh2;
            for (int y = 0; y < sh2; y++) {
                const float *mm = sm.ptr<float>(y);
                const cv::Vec3b *ip = si.ptr<cv::Vec3b>(y);
                cv::Vec3b *op = out2.ptr<cv::Vec3b>(oy2+y) + ox2;
                for (int x = 0; x < sw2; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    op[x][0]=(uchar)(op[x][0]+(ip[x][0]-op[x][0])*a);
                    op[x][1]=(uchar)(op[x][1]+(ip[x][1]-op[x][1])*a);
                    op[x][2]=(uchar)(op[x][2]+(ip[x][2]-op[x][2])*a);
                }
            }
            out2.copyTo(img);
            return;
        }
        if (mode == 10) {
            /* HOLOGRAM YOU: only your body becomes a flickering cyan
             * projection with scanlines; the room stays real. */
            unsigned tk = g_seg.counter;
            float flick = 0.8f + 0.2f * ((float)((tk*2654435761u)>>28) / 15.0f);
            for (int y = 0; y < img.rows; y++) {
                const float *mm = g_seg.mask.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                float sl = ((y + (int)tk) % 4 == 0) ? 0.45f : 1.0f;
                for (int x = 0; x < img.cols; x++) {
                    float a = mm[x]; if (a < 0.05f) continue;
                    float lum = (ip[x][0]*0.114f + ip[x][1]*0.587f + ip[x][2]*0.299f);
                    float b2 = std::min(255.f, (lum*1.35f + 40) * sl * flick);
                    float g2 = std::min(255.f, (lum*1.05f + 20) * sl * flick);
                    float r2 = lum * 0.35f * sl * flick;
                    ip[x][0]=(uchar)(ip[x][0]+(b2-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(g2-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(r2-ip[x][2])*a);
                }
            }
            return;
        }

        if (mode == 7) {
            /* WINDOW-3D (head-coupled perspective): your head position
             * steers a virtual camera — the learned background shifts
             * OPPOSITE your head, you shift slightly WITH it, and the
             * flat screen turns into a window with real parallax. */
            float tx = 0.f, ty = 0.f;
            {
                /* run face detection ourselves so window3d works alone.
                 * Lock order seg->face is unique in the codebase (no
                 * path takes face->seg), so no inversion is possible. */
                std::lock_guard<std::mutex> fl(g_face.mtx);
                const int TW = 320;
                double s2 = (img.cols > TW) ? (double)TW / img.cols : 1.0;
                int dw = std::max(16, (int)(img.cols * s2));
                int dh = std::max(16, (int)(img.rows * s2));
                if (ensure_face(dw, dh)) {
                    if (tick(g_face.counter) % INFER_EVERY == 0) {
                        cv::Mat small2;
                        cv::resize(img, small2, cv::Size(dw, dh));
                        cv::Mat faces2;
                        g_face.det->detect(small2, faces2);
                        faces2.copyTo(g_face.faces);
                    }
                    if (!g_face.faces.empty()) {
                        const float *ff = g_face.faces.ptr<float>(0);
                        tx = (ff[0] + ff[2]*0.5f) / (float)dw - 0.5f;
                        ty = (ff[1] + ff[3]*0.5f) / (float)dh - 0.5f;
                    }
                }
            }
            g_seg.hx = 0.85f*g_seg.hx + 0.15f*tx;
            g_seg.hy = 0.85f*g_seg.hy + 0.15f*ty;
            int bgdx = (int)( g_seg.hx * img.cols * 0.16f);
            int bgdy = (int)( g_seg.hy * img.rows * 0.10f);
            int pdx  = (int)(-g_seg.hx * img.cols * 0.05f);
            cv::Mat out2(img.size(), CV_8UC3);
            for (int y = 0; y < img.rows; y++) {
                int by = std::min(img.rows-1, std::max(0, y + bgdy));
                const cv::Vec3b *bp = g_seg.bg_est.ptr<cv::Vec3b>(by);
                const float *mm = g_seg.mask.ptr<float>(y);
                const cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                cv::Vec3b *op = out2.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    int bx = std::min(img.cols-1, std::max(0, x + bgdx));
                    op[x] = bp[bx];
                    int sx = x - pdx;
                    if (sx < 0 || sx >= img.cols) continue;
                    float a = mm[sx];
                    if (a < 0.05f) continue;
                    op[x][0]=(uchar)(op[x][0]+(ip[sx][0]-op[x][0])*a);
                    op[x][1]=(uchar)(op[x][1]+(ip[sx][1]-op[x][1])*a);
                    op[x][2]=(uchar)(op[x][2]+(ip[sx][2]-op[x][2])*a);
                }
            }
            out2.copyTo(img);
            return;
        }

        if (mode == 8) {
            /* CHRONO CLONES: translucent past-selves trail behind you.
             * A quarter-res ring stores your matted cutout; ghosts from
             * ~1.0 s and ~0.5 s ago composite behind the live you,
             * tinted cyan and magenta. */
            static cv::Mat rgbR[45], aR[45];
            static int ridx = 0, rqw = 0, rqh = 0;
            int QW = img.cols/4, QH = img.rows/4;
            if (rqw != QW || rqh != QH) {
                for (int i = 0; i < 45; i++) { rgbR[i].release(); aR[i].release(); }
                rqw = QW; rqh = QH; ridx = 0;
            }
            cv::Mat q, qm;
            cv::resize(img, q, cv::Size(QW, QH), 0, 0, cv::INTER_AREA);
            cv::resize(g_seg.mask, qm, cv::Size(QW, QH));
            q.copyTo(rgbR[ridx]); qm.copyTo(aR[ridx]);
            int gdel[2] = {30, 15};
            int tint[2][3] = {{255,180,40},{160,60,255}};   /* cyan-ish, magenta */
            for (int g2 = 0; g2 < 2; g2++) {
                int gi = (ridx - gdel[g2] + 45) % 45;
                if (rgbR[gi].empty()) continue;
                cv::Mat gb, ga;
                cv::resize(rgbR[gi], gb, img.size(), 0, 0, cv::INTER_LINEAR);
                cv::resize(aR[gi],  ga, img.size(), 0, 0, cv::INTER_LINEAR);
                const int *T = tint[g2];
                for (int y = 0; y < img.rows; y++) {
                    const float *am = ga.ptr<float>(y);
                    const float *cm = g_seg.mask.ptr<float>(y);
                    const cv::Vec3b *gp = gb.ptr<cv::Vec3b>(y);
                    cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < img.cols; x++) {
                        float a = am[x] * 0.55f * (1.0f - cm[x]);  /* behind you */
                        if (a < 0.04f) continue;
                        ip[x][0]=(uchar)(ip[x][0]+((gp[x][0]+T[0])/2-ip[x][0])*a);
                        ip[x][1]=(uchar)(ip[x][1]+((gp[x][1]+T[1])/2-ip[x][1])*a);
                        ip[x][2]=(uchar)(ip[x][2]+((gp[x][2]+T[2])/2-ip[x][2])*a);
                    }
                }
            }
            ridx = (ridx + 1) % 45;
            return;
        }

        if (mode == 6) {
            /* CLONE ARMY: three of you, standing in the learned room.
             * Composite the matted person at -w/3, 0, +w/3 over bg_est. */
            cv::Mat out2;
            g_seg.bg_est.copyTo(out2);
            int offs[3] = { -img.cols/3, 0, img.cols/3 };
            for (int k = 0; k < 3; k++) {
                int dx = offs[k];
                for (int y = 0; y < img.rows; y++) {
                    const float *mm = g_seg.mask.ptr<float>(y);
                    const cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                    cv::Vec3b *op = out2.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < img.cols; x++) {
                        int sx = x - dx;
                        if (sx < 0 || sx >= img.cols) continue;
                        float a = mm[sx];
                        if (a < 0.05f) continue;
                        op[x][0]=(uchar)(op[x][0]+(ip[sx][0]-op[x][0])*a);
                        op[x][1]=(uchar)(op[x][1]+(ip[sx][1]-op[x][1])*a);
                        op[x][2]=(uchar)(op[x][2]+(ip[sx][2]-op[x][2])*a);
                    }
                }
            }
            out2.copyTo(img);
            return;
        }

        cv::Mat bg;
        bool have_user_bg = false;
        if (mode == 3) {
            /* user-supplied background image (cover-scaled + cached) */
            std::lock_guard<std::mutex> bl(g_bgimg.mtx);
            if (!g_bgimg.orig.empty()) {
                if (g_bgimg.scaled.size() != img.size()) {
                    /* cover: scale so both dims fill, center-crop */
                    double sx = (double)img.cols / g_bgimg.orig.cols;
                    double sy = (double)img.rows / g_bgimg.orig.rows;
                    double s2 = std::max(sx, sy);
                    cv::Mat big;
                    cv::resize(g_bgimg.orig, big, cv::Size(), s2, s2,
                               cv::INTER_AREA);
                    int ox = (big.cols - img.cols) / 2;
                    int oy = (big.rows - img.rows) / 2;
                    big(cv::Rect(ox, oy, img.cols, img.rows))
                        .copyTo(g_bgimg.scaled);
                }
                g_bgimg.scaled.copyTo(bg);
                have_user_bg = true;
            }
            /* no image set → fall through to the gradient below */
        }
        if (have_user_bg) {
            /* bg ready */
        } else if (mode == 0) {
            /* portrait: blur the background */
            cv::GaussianBlur(img, bg, cv::Size(0, 0), 8.0);
        } else if (mode == 1) {
            /* spotlight: gray + dim the background */
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, bg, cv::COLOR_GRAY2BGR);
            bg *= 0.5;
        } else if (mode == 4) {
            /* INVISIBILITY CLOAK: paint the person with the learned
             * background. Needs ~1-2s of you moving around (or briefly
             * stepping out) so the room behind you gets learned. */
            g_seg.bg_est.copyTo(bg);
        } else if (mode == 5) {
            /* NEON PERSON: the room goes near-black, you stay lit, and
             * your silhouette gets an additive cyan edge glow. */
            bg = cv::Mat::zeros(img.size(), CV_8UC3);
            cv::Mat m8, dil, ero, edge;
            g_seg.mask.convertTo(m8, CV_8U, 255.0);
            cv::dilate(m8, dil, cv::Mat(), cv::Point(-1,-1), 3);
            cv::erode (m8, ero, cv::Mat(), cv::Point(-1,-1), 3);
            edge = dil - ero;
            for (int y = 0; y < img.rows; y++) {
                const uchar *ep = edge.ptr<uchar>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < img.cols; x++) {
                    if (ep[x] > 40) {
                        ip[x][0] = clamp_int_u8(ip[x][0] + ep[x]);
                        ip[x][1] = clamp_int_u8(ip[x][1] + ep[x]);
                        ip[x][2] = clamp_int_u8(ip[x][2] + ep[x]/3);
                    }
                }
            }
        } else {
            /* replace with a moving diagonal color gradient */
            bg = cv::Mat(img.size(), CV_8UC3);
            static unsigned t = 0; t++;
            for (int y = 0; y < bg.rows; y++) {
                cv::Vec3b *row = bg.ptr<cv::Vec3b>(y);
                for (int x = 0; x < bg.cols; x++) {
                    uchar r = (uchar)(128 + 127 * std::sin((x + y + t*4) * 0.02));
                    uchar g = (uchar)(128 + 127 * std::sin((x + y + t*4) * 0.02 + 2.0));
                    uchar b = (uchar)(128 + 127 * std::sin((x + y + t*4) * 0.02 + 4.0));
                    row[x] = cv::Vec3b(b, g, r);
                }
            }
        }

        /* blend: person from img, elsewhere from bg, feathered by mask.
         * out = img*mask + bg*(1-mask), done per-channel in floating
         * point. On OpenCV 5 this runs in first-class FP16 (cv::hfloat,
         * CV_16F) — half the memory traffic of FP32, with arithmetic
         * guaranteed in 5.x even without native FP16 hardware. Mask is
         * 0..1 and pixels 0..255, comfortably inside half range. */
#if VCAM_CV5
        const int FDEPTH1 = CV_MAKETYPE(CV_16F, 1);
        const int FDEPTH3 = CV_MAKETYPE(CV_16F, 3);
#else
        const int FDEPTH1 = CV_32FC1;
        const int FDEPTH3 = CV_32FC3;
#endif
        cv::Mat maskf;
        g_seg.mask.convertTo(maskf, FDEPTH1);                 /* already 0..1 */
        cv::Mat invf = cv::Scalar::all(1.0) - maskf;          /* Scalar - Mat */

        cv::Mat m3, inv3;
        {
            cv::Mat mc[3] = { maskf, maskf, maskf };
            cv::merge(mc, 3, m3);
            cv::Mat ic[3] = { invf, invf, invf };
            cv::merge(ic, 3, inv3);
        }
        cv::Mat imgf, bgf, outf;
        img.convertTo(imgf, FDEPTH3);
        bg.convertTo(bgf, FDEPTH3);
        outf = imgf.mul(m3) + bgf.mul(inv3);
        outf.convertTo(img, CV_8UC3);
    } catch (const std::exception &e) {
        warn_once("bgblur", std::string("bg fx failed: ") + e.what());
    }
}

void cv_fx_style(unsigned char *data, int w, int h, int stride,
                 int style, float blend, float size_sel)
{
    if (!data || w < 32 || h < 32) return;
    if (style < 0) style = 0;
    if (style > 4) style = 4;
    if (blend <= 0.0f || blend > 1.0f) blend = 0.85f;
    int S = (int)size_sel;
    if (S < 96)  S = 192;         /* quality↔speed knob, default 192 */
    if (S > 384) S = 384;
    try {
        std::lock_guard<std::mutex> lock(g_style.mtx);
        cv::Mat img = wrap_bgr(data, w, h, stride);
        if (!ensure_style(style)) return;

        /* keep aspect; round dims to /8 for conv friendliness */
        double sc = (double)S / std::max(w, h);
        int sw = std::max(64, ((int)(w * sc) / 8) * 8);
        int sh = std::max(64, ((int)(h * sc) / 8) * 8);
        cv::Mat small;
        cv::resize(img, small, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);

        /* fast-neural-style expects raw RGB floats 0..255 (no norm) */
        cv::Mat blob = cv::dnn::blobFromImage(
            small, 1.0, cv::Size(sw, sh), cv::Scalar(), true, false);
        g_style.net[style].setInput(blob);
        cv::Mat out = g_style.net[style].forward();  /* [1,3,sh,sw] RGB */
        if (out.dims != 4 || out.size[1] != 3) return;
        int oh = out.size[2], ow = out.size[3];

        cv::Mat styled(oh, ow, CV_8UC3);
        const float *R = out.ptr<float>(0, 0);
        const float *G = out.ptr<float>(0, 1);
        const float *B = out.ptr<float>(0, 2);
        for (int y = 0; y < oh; y++) {
            cv::Vec3b *row = styled.ptr<cv::Vec3b>(y);
            for (int x = 0; x < ow; x++) {
                int i = y * ow + x;
                row[x] = cv::Vec3b(
                    cv::saturate_cast<uchar>(B[i]),
                    cv::saturate_cast<uchar>(G[i]),
                    cv::saturate_cast<uchar>(R[i]));
            }
        }
        /* temporal EMA at small res: hides inter-frame flicker */
        if (g_style.ema_style == style && !g_style.ema.empty() &&
            g_style.ema.size() == styled.size()) {
            cv::addWeighted(styled, 0.65, g_style.ema, 0.35, 0.0, styled);
        }
        styled.copyTo(g_style.ema);
        g_style.ema_style = style;

        cv::Mat full;
        cv::resize(styled, full, img.size(), 0, 0, cv::INTER_LINEAR);
        cv::addWeighted(full, blend, img, 1.0 - blend, 0.0, img);
    } catch (const std::exception &e) {
        warn_once("style", std::string("styleai failed: ") + e.what());
    }
}

void cv_fx_mood(unsigned char *data, int w, int h, int stride,
                int mode, float gain, float extra)
{
    if (!data || w < 32 || h < 32) return;
    if (extra <= 0.0f) extra = 1.0f;         /* mode-specific: party density /
                                              * border thickness */
    if (extra < 0.3f) extra = 0.3f;
    if (extra > 3.0f) extra = 3.0f;
    if (mode < 0) mode = 0;
    if (mode > 7) mode = 7;
    if (gain <= 0.0f || gain > 2.0f) gain = 1.0f;
    try {
        cv::Mat img = wrap_bgr(data, w, h, stride);

        /* face detection (shares the YuNet state with cv_fx_face) */
        cv::Rect roi;
        double eye_ang = 0.0;
        {
            std::lock_guard<std::mutex> lock(g_face.mtx);
            const int TW = 320;
            double s = (w > TW) ? (double)TW / w : 1.0;
            int dw = std::max(16, (int)(w * s));
            int dh = std::max(16, (int)(h * s));
            if (!ensure_face(dw, dh)) return;
            if (tick(g_face.counter) % INFER_EVERY == 0) {
                cv::Mat small;
                cv::resize(img, small, cv::Size(dw, dh));
                cv::Mat faces;
                g_face.det->detect(small, faces);
                faces.copyTo(g_face.faces);
            }
            if (g_face.faces.empty()) return;
            const float *f = g_face.faces.ptr<float>(0);   /* largest/first */
            double inv = 1.0 / s;
            int x = (int)(f[0]*inv), y = (int)(f[1]*inv);
            int fw = (int)(f[2]*inv), fh = (int)(f[3]*inv);
            x = std::max(0, std::min(x, w-2));
            y = std::max(0, std::min(y, h-2));
            fw = std::max(2, std::min(fw, w-x));
            fh = std::max(2, std::min(fh, h-y));
            roi = cv::Rect(x, y, fw, fh);
            eye_ang = std::atan2((f[7]-f[5]), (f[6]-f[4])) * 180.0 / 3.14159265;
        }

        std::lock_guard<std::mutex> lock(g_mood.mtx);
        if (!ensure_mood()) return;

        /* FER is ~2 ms — run it EVERY frame (the old every-3rd throttle
         * plus slow EMA made reactions feel dead). */
        tick(g_mood.counter);
        {
            /* ALIGNED crop: the model was trained on eye-level faces.
             * Rotate a padded square around the face by the eye angle,
             * then crop the center — accuracy jumps massively vs the
             * raw bounding box we fed it before. */
            int cx2 = roi.x + roi.width/2, cy2 = roi.y + roi.height/2;
            int half = (int)(std::max(roi.width, roi.height) * 0.72);
            int x0 = std::max(0, cx2-half), y0 = std::max(0, cy2-half);
            int x1 = std::min(w, cx2+half), y1 = std::min(h, cy2+half);
            cv::Mat pad = img(cv::Rect(x0, y0, x1-x0, y1-y0));
            /* rotation matrix by hand (getRotationMatrix2D moved into
             * the new 'geometry' module in OpenCV 5 — building the 2x3
             * affine ourselves works identically on every version):
             *   [  c  s  (1-c)*cx - s*cy ]
             *   [ -s  c  s*cx + (1-c)*cy ]   (positive angle = CCW)   */
            double ra = eye_ang * CV_PI / 180.0;
            double c = std::cos(ra), s = std::sin(ra);
            double rcx = (double)(cx2 - x0), rcy = (double)(cy2 - y0);
            double rbuf[6] = {  c, s, (1.0 - c) * rcx - s * rcy,
                               -s, c, s * rcx + (1.0 - c) * rcy };
            cv::Mat rot(2, 3, CV_64F, rbuf);   /* used within this scope only */
            cv::Mat level;
            cv::warpAffine(pad, level, rot, pad.size(),
                           cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            int side = std::min({level.cols, level.rows,
                                 (int)(std::max(roi.width, roi.height)*1.15)});
            int ox = std::max(0, (cx2-x0) - side/2);
            int oy = std::max(0, (cy2-y0) - side/2);
            ox = std::min(ox, level.cols - side);
            oy = std::min(oy, level.rows - side);
            cv::Mat face112;
            cv::resize(level(cv::Rect(ox, oy, side, side)), face112,
                       cv::Size(112, 112));
            /* MobileFaceNet convention: (x/255 - 0.5)/0.5, RGB */
            cv::Mat blob = cv::dnn::blobFromImage(
                face112, 1.0 / 127.5, cv::Size(112, 112),
                cv::Scalar(127.5, 127.5, 127.5), true, false);
            g_mood.net.setInput(blob);
            cv::Mat out = g_mood.net.forward();   /* 7 scores */
            if ((int)out.total() >= 7) {
                const float *o = out.ptr<float>(0);
                /* softmax → EMA smoothing (kills flicker) */
                float mx = o[0];
                for (int i = 1; i < 7; i++) mx = std::max(mx, o[i]);
                float ex[7], sum = 0.f;
                for (int i = 0; i < 7; i++) { ex[i] = std::exp(o[i]-mx); sum += ex[i]; }
                for (int i = 0; i < 7; i++)
                    g_mood.scores[i] = 0.6f*g_mood.scores[i] + 0.4f*(ex[i]/sum);
            }
        }
        int emo = 0;
        for (int i = 1; i < 7; i++)
            if (g_mood.scores[i] > g_mood.scores[emo]) emo = i;
        /* rescale: 1/7 (chance) -> 0, ~0.65 (clear expression) -> 1 */
        float raw = (g_mood.scores[emo] - 0.143f) / 0.5f;
        float conf = std::min(1.0f, std::max(0.0f, raw) * gain);
        const int *C = EMO_COL[emo];

        if (mode == 0) {
            /* AURA: soft colored vignette breathing with confidence.
             * Neutral = no tint. */
            if (emo == 4 || conf < 0.08f) return;
            float cx = w/2.f, cy = h/2.f;
            float maxr = std::sqrt(cx*cx + cy*cy);
            for (int y = 0; y < h; y++) {
                uchar *row = data + y*stride;
                float dy = y - cy;
                for (int x = 0; x < w; x++) {
                    float dx = x - cx;
                    float t = std::sqrt(dx*dx + dy*dy) / maxr;   /* 0..1 */
                    t = t*t * conf * 0.9f;
                    uchar *p = row + x*3;
                    p[0] = (uchar)(p[0] + (C[0]-p[0])*t);
                    p[1] = (uchar)(p[1] + (C[1]-p[1])*t);
                    p[2] = (uchar)(p[2] + (C[2]-p[2])*t);
                }
            }
        } else if (mode == 1) {
            /* HUD: labeled box + confidence bar over the face */
            cv::Scalar col(C[0], C[1], C[2]);
            cv::rectangle(img, roi, col, 2);
            char label[48];
            snprintf(label, sizeof label, "%s %d%%", EMO_NAME[emo],
                     (int)(conf * 100));
            int ty = std::max(18, roi.y - 8);
            cv::putText(img, label, cv::Point(roi.x, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 4);
            cv::putText(img, label, cv::Point(roi.x, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, col, 2);
            int bw = (int)(roi.width * conf);
            cv::rectangle(img, cv::Rect(roi.x, ty + 6, bw, 5), col, cv::FILLED);
        } else if (mode == 7) {
            /* EMOTION FOUNTAIN: particles pour from behind your head,
             * their motion written by your mood — happy erupts golden
             * upward, sad drips slowly, angry bursts radial sparks,
             * surprised rings outward. extra = flow rate. */
            static float qx2[110], qy3[110], qvx[110], qvy[110], ql2[110];
            static int qc2[110], qinit = 0;
            if (!qinit) { for (int i=0;i<110;i++) ql2[i]=0; qinit=1; }
            float ex2 = roi.x + roi.width/2.f, ey2 = (float)roi.y;
            int rate = (int)(conf * 6 * extra);
            for (int i = 0; i < 110 && rate > 0; i++) {
                if (ql2[i] > 0) continue;
                ql2[i] = 1.0f; qc2[i] = emo;
                qx2[i] = ex2 + ((rand()%100)-50) * roi.width / 130.0f;
                qy3[i] = ey2;
                if (emo == 3) {          /* happy: up fountain */
                    qvx[i]=((rand()%100)-50)/22.0f; qvy[i]=-4.0f-(rand()%100)/33.0f;
                } else if (emo == 5) {   /* sad: slow drip */
                    qvx[i]=((rand()%100)-50)/90.0f; qvy[i]=0.6f+(rand()%100)/160.0f;
                } else if (emo == 0) {   /* angry: radial burst */
                    float aa=(rand()%628)/100.0f;
                    qvx[i]=std::cos(aa)*4.5f; qvy[i]=std::sin(aa)*4.5f;
                } else if (emo == 6) {   /* surprised: outward ring */
                    float aa=(rand()%628)/100.0f;
                    qvx[i]=std::cos(aa)*2.6f; qvy[i]=std::sin(aa)*2.6f - 1.0f;
                } else {                 /* neutral-ish wisps */
                    qvx[i]=((rand()%100)-50)/60.0f; qvy[i]=-0.7f;
                }
                rate--;
            }
            for (int i = 0; i < 110; i++) {
                if (ql2[i] <= 0) continue;
                if (qc2[i] == 3) qvy[i] += 0.14f;         /* fountain falls back */
                qx2[i]+=qvx[i]; qy3[i]+=qvy[i]; ql2[i]-=0.018f;
                int X=(int)qx2[i], Y=(int)qy3[i];
                if (X<2||Y<2||X>=w-2||Y>=h-2) { ql2[i]=0; continue; }
                const int *cc = EMO_COL[qc2[i]];
                float li = ql2[i];
                cv::circle(img, {X,Y}, 2,
                           cv::Scalar(cc[0]*li, cc[1]*li, cc[2]*li), cv::FILLED);
            }
        } else if (mode == 6) {
            /* MOOD RING: a breathing colored border frames the call in
             * your current emotion; thickness pulses with confidence. */
            int th = (int)((10 + (int)(conf * 26)) * extra)
                   + (int)(4 * std::sin((float)g_mood.counter * 0.1f));
            for (int y = 0; y < h; y++) {
                uchar *row = data + y * stride;
                for (int x = 0; x < w; x++) {
                    int db = std::min(std::min(x, w-1-x), std::min(y, h-1-y));
                    if (db >= th) { if (x > th && x < w-1-th) { x = w-1-th; } continue; }
                    float k = 1.0f - (float)db / th;
                    k = k * k * (0.35f + conf * 0.65f);
                    uchar *p2 = row + x * 3;
                    p2[0]=(uchar)(p2[0]+(C[0]-p2[0])*k);
                    p2[1]=(uchar)(p2[1]+(C[1]-p2[1])*k);
                    p2[2]=(uchar)(p2[2]+(C[2]-p2[2])*k);
                }
            }
        } else if (mode == 5) {
            /* AUTO-GRADE: cinematic color grade steered by emotion.
             * happy=warm lift, sad=cool desat, angry=hot contrast,
             * surprised=bright pop, fear=purple shadows, disgust=green. */
            static const float G[7][4] = {   /* mulB mulG mulR add */
                {0.85f,0.90f,1.20f, 8},{0.90f,1.15f,0.90f, 0},
                {1.15f,0.90f,1.05f,-8},{0.95f,1.05f,1.15f,12},
                {1.f,1.f,1.f,0},{1.20f,1.00f,0.85f,-10},{1.05f,1.05f,1.05f,18}};
            const float *g4=G[emo];
            float k4=conf;
            for(int y=0;y<h;y++){
                uchar *p2=data+y*stride;
                for(int x=0;x<w;x++){
                    uchar *q2=p2+x*3;
                    q2[0]=clamp_int_u8((int)(q2[0]*(1+(g4[0]-1)*k4)+g4[3]*k4));
                    q2[1]=clamp_int_u8((int)(q2[1]*(1+(g4[1]-1)*k4)+g4[3]*k4));
                    q2[2]=clamp_int_u8((int)(q2[2]*(1+(g4[2]-1)*k4)+g4[3]*k4));
                }
            }
        } else if (mode == 4) {
            /* PAINTMOOD: two neural nets chained — the FER model reads
             * your emotion, which SELECTS the neural style; confidence
             * drives the blend. Happy world is candy, sad world is
             * rain-princess, anger turns everything to mosaic shards.
             * (Lock nesting mood->style is unique; style never locks
             * mood, so no inversion.) Needs the --models-style pack. */
            static const int emo2style[7] = { 0, 3, 2, 1, 3, 2, 4 };
            /*  angry->mosaic disgust->udnie fear->rain happy->candy
             *  neutral->udnie sad->rain-princess surprised->pointilism */
            float sb = 0.30f + 0.55f * conf;
            cv_fx_style(data, w, h, stride, emo2style[emo], sb, 192.0f);
        } else if (mode == 3) {
            /* EMOJI: your face becomes the emotion-matched emoji */
            cv::Point c(roi.x + roi.width/2, roi.y + roi.height/2);
            int R2 = (int)(std::max(roi.width, roi.height) * 0.62);
            cv::circle(img, c, R2, cv::Scalar(0, 200, 255), cv::FILLED);
            cv::circle(img, c, R2, cv::Scalar(0, 140, 200), std::max(2, R2/24));
            int er = std::max(3, R2/7);
            cv::Point el(c.x - R2/3, c.y - R2/4), er2(c.x + R2/3, c.y - R2/4);
            if (emo == 6) {                       /* surprised: O eyes+mouth */
                cv::circle(img, el, er, cv::Scalar(30,30,30), cv::FILLED);
                cv::circle(img, er2, er, cv::Scalar(30,30,30), cv::FILLED);
                cv::circle(img, cv::Point(c.x, c.y + R2/3), er+2,
                           cv::Scalar(30,30,30), cv::FILLED);
            } else {
                cv::circle(img, el, er, cv::Scalar(30,30,30), cv::FILLED);
                cv::circle(img, er2, er, cv::Scalar(30,30,30), cv::FILLED);
                if (emo == 0) {                   /* angry brows */
                    cv::line(img, cv::Point(el.x-er, el.y-er-4),
                             cv::Point(el.x+er, el.y-er+4), cv::Scalar(30,30,30), 5);
                    cv::line(img, cv::Point(er2.x+er, er2.y-er-4),
                             cv::Point(er2.x-er, er2.y-er+4), cv::Scalar(30,30,30), 5);
                }
                int my = c.y + R2/3;
                if (emo == 3)                     /* happy: smile arc */
                    cv::ellipse(img, cv::Point(c.x,my-6), cv::Size(R2/2,R2/3),
                                0, 20, 160, cv::Scalar(30,30,30), 6);
                else if (emo == 5 || emo == 2)    /* sad/fear: frown */
                    cv::ellipse(img, cv::Point(c.x,my+R2/4), cv::Size(R2/2,R2/4),
                                0, 200, 340, cv::Scalar(30,30,30), 6);
                else                              /* neutral-ish: flat */
                    cv::line(img, cv::Point(c.x-R2/3,my), cv::Point(c.x+R2/3,my),
                             cv::Scalar(30,30,30), 6);
                if (emo == 5)                     /* sad: a tear */
                    cv::circle(img, cv::Point(el.x, el.y+er*2),
                               std::max(2,er/2), cv::Scalar(230,160,60), cv::FILLED);
            }
        } else {
            /* PARTY: emotions trigger physical effects.
             * happy → confetti; sad → falling blue rain streaks;
             * surprised → white edge flash; angry → screen shake. */
            {   /* live readout, bottom-right — see what the model sees */
                char lab[40];
                snprintf(lab, sizeof lab, "%s %d%%", EMO_NAME[emo], (int)(conf*100));
                cv::putText(img, lab, cv::Point(w-400, h-22),
                            cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(0,0,0), 6);
                cv::putText(img, lab, cv::Point(w-400, h-22),
                            cv::FONT_HERSHEY_SIMPLEX, 1.1,
                            cv::Scalar(C[0],C[1],C[2]), 2);
            }
            if (emo == 3 && conf > 0.18f) {         /* confetti */
                for (int i = 0; i < 96; i++) {
                    if (g_mood.pl[i] <= 0 && (rand() % 100) < (int)(conf*25*extra)) {
                        g_mood.px[i] = (float)(rand() % w);
                        g_mood.py[i] = -(float)(rand() % 40);
                        g_mood.pvx[i] = ((rand()%100)-50) / 40.0f;
                        g_mood.pvy[i] = 2.0f + (rand()%100)/40.0f;
                        g_mood.pl[i] = 1.0f;
                        g_mood.pc[i] = rand() % 7;
                    }
                }
            }
            for (int i = 0; i < 96; i++) {           /* step + draw */
                if (g_mood.pl[i] <= 0) continue;
                g_mood.px[i] += g_mood.pvx[i];
                g_mood.py[i] += g_mood.pvy[i];
                g_mood.pl[i] -= 0.008f;
                int X = (int)g_mood.px[i], Y = (int)g_mood.py[i];
                if (X < 1 || X >= w-4 || Y < 0 || Y >= h-4) { g_mood.pl[i]=0; continue; }
                const int *cc = EMO_COL[g_mood.pc[i]];
                for (int yy = 0; yy < 4; yy++) {
                    uchar *p = data + (Y+yy)*stride + X*3;
                    for (int xx = 0; xx < 4; xx++) {
                        p[xx*3+0]=(uchar)cc[0]; p[xx*3+1]=(uchar)cc[1]; p[xx*3+2]=(uchar)cc[2];
                    }
                }
            }
            if (emo == 5 && conf > 0.18f) {          /* sad rain */
                unsigned seed = (unsigned)g_mood.counter * 2654435761u;
                int n = (int)(conf * 40);
                for (int i = 0; i < n; i++) {
                    seed = seed*1664525u + 1013904223u;
                    int X = (int)(seed % (unsigned)w);
                    int Y = (int)((seed >> 12) % (unsigned)(h - 20));
                    for (int yy = 0; yy < 14; yy++) {
                        uchar *p = data + (Y+yy)*stride + X*3;
                        p[0] = clamp_int_u8(p[0] + 90); p[1] = clamp_int_u8(p[1] + 40);
                    }
                }
            }
            if (emo == 6 && conf > 0.18f) {          /* surprised flash edges */
                int bh = (int)(conf * 24);
                for (int y = 0; y < bh; y++) {
                    uchar *t2 = data + y*stride;
                    uchar *b2 = data + (h-1-y)*stride;
                    for (int x = 0; x < w*3; x++) {
                        t2[x] = clamp_int_u8(t2[x] + 120);
                        b2[x] = clamp_int_u8(b2[x] + 120);
                    }
                }
            }
            if (emo == 0 && conf > 0.18f) {          /* angry shake */
                int off = ((int)(g_mood.counter % 4) - 2) * (int)(conf * 6);
                if (off != 0) {
                    cv::Mat shifted;
                    img.copyTo(shifted);
                    int ax = std::abs(off);
                    if (ax < w) {
                        cv::Rect src2 = (off > 0) ? cv::Rect(0,0,w-ax,h) : cv::Rect(ax,0,w-ax,h);
                        cv::Rect dst2 = (off > 0) ? cv::Rect(ax,0,w-ax,h) : cv::Rect(0,0,w-ax,h);
                        shifted(src2).copyTo(img(dst2));
                    }
                }
            }
        }
    } catch (const std::exception &e) {
        warn_once("mood", std::string("moodfx failed: ") + e.what());
    }
}

void cv_fx_paintflow(unsigned char *data, int w, int h, int stride,
                     float strength, float decay)
{
    if (!data || w < 16 || h < 16) return;
    if (strength <= 0.0f || strength > 3.0f) strength = 1.0f;
    if (decay < 0.80f || decay > 0.99f) decay = 0.93f;
    try {
        std::lock_guard<std::mutex> lock(g_paint.mtx);
        cv::Mat img = wrap_bgr(data, w, h, stride);
        cv::Mat gray, gs;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gs, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

        if (g_paint.canvas.empty() || g_paint.canvas.size() != img.size() ||
            g_paint.prev_gray.size() != gs.size()) {
            img.copyTo(g_paint.canvas);
            gs.copyTo(g_paint.prev_gray);
            g_paint.mapx.create(h, w, CV_32F);
            g_paint.mapy.create(h, w, CV_32F);
            return;
        }

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(g_paint.prev_gray, gs, flow,
                                     0.5, 2, 13, 3, 5, 1.1, 0);
        gs.copyTo(g_paint.prev_gray);

        /* advect the canvas: each pixel pulls paint from where the
         * motion came from (flow at half res, scaled x2) */
        float k = 2.0f * strength;
        for (int y = 0; y < h; y++) {
            float *mx = g_paint.mapx.ptr<float>(y);
            float *my = g_paint.mapy.ptr<float>(y);
            const cv::Vec2f *fr = flow.ptr<cv::Vec2f>(std::min(y/2, flow.rows-1));
            for (int x = 0; x < w; x++) {
                const cv::Vec2f &v = fr[std::min(x/2, flow.cols-1)];
                mx[x] = x - v[0]*k;
                my[x] = y - v[1]*k;
            }
        }
        cv::Mat moved;
        cv::remap(g_paint.canvas, moved, g_paint.mapx, g_paint.mapy,
                  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        /* refresh with a whisper of live video so stills stay sharp */
        cv::addWeighted(moved, decay, img, 1.0 - decay, 0.0, g_paint.canvas);
        g_paint.canvas.copyTo(img);
    } catch (const std::exception &e) {
        warn_once("paintflow", std::string("paintflow failed: ") + e.what());
    }
}

void cv_fx_speedlines(unsigned char *data, int w, int h, int stride,
                      float sens, float density)
{
    if (!data || w < 32 || h < 32) return;
    if (sens <= 0.0f || sens > 3.0f) sens = 1.0f;
    if (density <= 0.0f || density > 1.0f) density = 0.5f;
    try {
        std::lock_guard<std::mutex> lock(g_classic.mtx);
        cv::Mat img = wrap_bgr(data, w, h, stride);
        cv::Mat gray, gs;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gs, cv::Size(), 0.25, 0.25, cv::INTER_LINEAR);
        if (g_classic.prev_gray.empty() ||
            g_classic.prev_gray.size() != gs.size()) {
            gs.copyTo(g_classic.prev_gray);
            return;
        }
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(g_classic.prev_gray, gs, flow,
                                     0.5, 2, 9, 2, 5, 1.1, 0);
        gs.copyTo(g_classic.prev_gray);
        cv::Scalar mean2 = cv::mean(flow);
        float mx2 = (float)mean2[0]*4, my2 = (float)mean2[1]*4;
        float mag = std::sqrt(mx2*mx2 + my2*my2) * sens;
        if (mag < 3.0f) return;                     /* not moving enough */
        float L = std::sqrt(mx2*mx2 + my2*my2);
        float dx = -mx2/L, dy = -my2/L;             /* opposite motion */
        int N = (int)(10 + density * 50);
        unsigned seed = (unsigned)((w+h) * 2654435761u) ^ (unsigned)(mag*97);
        float inten = std::min(1.0f, mag / 30.0f);
        for (int i = 0; i < N; i++) {
            seed = seed*1664525u + 1013904223u;
            /* start on the border the lines come FROM */
            float t = (float)(seed % 10000u) / 10000.0f;
            int sx2, sy2;
            if (std::fabs(dx) > std::fabs(dy)) {
                sx2 = dx > 0 ? 0 : w-1;
                sy2 = (int)(t * h);
            } else {
                sy2 = dy > 0 ? 0 : h-1;
                sx2 = (int)(t * w);
            }
            seed = seed*1664525u + 1013904223u;
            int len = (int)((0.15f + 0.20f*((seed>>20)&255)/255.0f)
                            * std::min(w,h) * inten) + 8;
            cv::Point a2(sx2, sy2);
            cv::Point b2(sx2 + (int)(dx*len), sy2 + (int)(dy*len));
            int v2 = 120 + (int)(inten * 135);
            cv::line(img, a2, b2, cv::Scalar(v2,v2,v2), 1 + (int)(seed&1));
        }
    } catch (const std::exception &e) {
        warn_once("speedlines", std::string("speedlines failed: ") + e.what());
    }
}

/* ── enhance: retinex-style low-light rescue ────────────────────
 * out = in / blurred_illumination — recovers shadows the way the eye
 * does, plus gray-world white balance and shadow denoise. */
void cv_fx_enhance(unsigned char *data, int w, int h, int stride,
                   float strength, float denoise)
{
    if (!data || w < 32 || h < 32) return;
    if (strength <= 0.0f) strength = 1.0f;
    if (strength < 0.2f) strength = 0.2f;
    if (strength > 2.0f) strength = 2.0f;
    if (denoise < 0.0f) denoise = 0.0f;
    if (denoise > 1.0f) denoise = 1.0f;
    try {
        cv::Mat img(h, w, CV_8UC3, data, stride);
        /* illumination estimate: heavy blur of luma at quarter res */
        cv::Mat q, luma(h/4, w/4, CV_32F);
        cv::resize(img, q, cv::Size(w/4, h/4), 0, 0, cv::INTER_AREA);
        double mb = 0, mg = 0, mr = 0;
        for (int y = 0; y < q.rows; y++) {
            const cv::Vec3b *p2 = q.ptr<cv::Vec3b>(y);
            float *lp = luma.ptr<float>(y);
            for (int x = 0; x < q.cols; x++) {
                lp[x] = (p2[x][0]*0.114f+p2[x][1]*0.587f+p2[x][2]*0.299f)/255.f;
                mb += p2[x][0]; mg += p2[x][1]; mr += p2[x][2];
            }
        }
        double N = (double)q.rows * q.cols;
        mb/=N; mg/=N; mr/=N;
        double avg = (mb+mg+mr)/3.0 + 1e-3;
        float wbB = (float)std::min(1.25, std::max(0.8, avg/(mb+1e-3)));
        float wbG = (float)std::min(1.25, std::max(0.8, avg/(mg+1e-3)));
        float wbR = (float)std::min(1.25, std::max(0.8, avg/(mr+1e-3)));
        cv::GaussianBlur(luma, luma, cv::Size(0,0), 14.0);
        cv::Mat illum;
        cv::resize(luma, illum, img.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat soft;
        if (denoise > 0.01f)
            cv::GaussianBlur(img, soft, cv::Size(0,0), 1.5);
        for (int y = 0; y < h; y++) {
            const float *il = illum.ptr<float>(y);
            const cv::Vec3b *sp = soft.empty() ? NULL : soft.ptr<cv::Vec3b>(y);
            cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < w; x++) {
                float L2 = il[x];
                /* retinex gain: dark regions lifted toward mid gray */
                float gain = (0.45f + 0.1f*strength) / (L2 + 0.06f);
                gain = 1.0f + (gain - 1.0f) * strength * 0.8f;
                if (gain < 1.0f) gain = 1.0f;
                if (gain > 4.5f) gain = 4.5f;
                float dark = 1.0f - std::min(1.0f, L2*2.2f);
                for (int c2 = 0; c2 < 3; c2++) {
                    float v = ip[x][c2];
                    if (sp) v += (sp[x][c2]-v) * denoise * dark;   /* shadow DN */
                    float wb2 = c2==0?wbB:(c2==1?wbG:wbR);
                    v = v * gain * wb2;
                    /* soft shoulder so highlights don't clip hard */
                    if (v > 210) v = 210 + (v-210)*0.5f;
                    ip[x][c2] = clamp_int_u8((int)v);
                }
            }
        }
    } catch (const std::exception &e) {
        warn_once("enh_fail", std::string("enhance failed: ") + e.what());
    }
}

/* ── posefx: BlazePose body landmarks via the ORT backend ──────── */
struct PoseState {
    std::string path;
    int state = 0;
    float lm[33][3];              /* x,y (frame px), visibility */
    int have = 0;                 /* landmarks valid this session */
    int good = 0;                 /* consecutive confident frames */
    float trailx[2][24], traily[2][24];   /* wrist trails */
    int trail_n = 0;
    double ema_ms = 0;
    int skip_left = 0;
    unsigned tick = 0;
    std::mutex mtx;
};
static PoseState g_pose_s;

int cv_set_pose_model(const char *path)
{
    if (!path || !*path || !path_exists(path)) return 0;
    std::lock_guard<std::mutex> l(g_pose_s.mtx);
    g_pose_s.path = path;
    g_pose_s.state = 0;
    ort_pose_shutdown();
    g_pose_s.have = 0;
    return 1;
}

/* BlazePose-33 bone list */
static const int POSE_BONES[][2] = {
    {11,12},{11,13},{13,15},{12,14},{14,16},{11,23},{12,24},{23,24},
    {23,25},{25,27},{24,26},{26,28},{0,11},{0,12}};

void cv_fx_pose(unsigned char *data, int w, int h, int stride,
                int mode, float fx)
{
    if (!data || w < 64 || h < 64) return;
    if (mode < 0) mode = 0;
    if (mode > 5) mode = 5;
    if (fx <= 0.0f) fx = 1.0f;
    if (fx < 0.2f) fx = 0.2f;
    if (fx > 3.0f) fx = 3.0f;
    if (!ort_available()) {
        warn_once("pose_ort", "posefx needs the ONNX Runtime backend — "
                  "vcam-setup --ort, then make OPENCV=1 ORT=1");
        return;
    }
    try {
        std::lock_guard<std::mutex> l(g_pose_s.mtx);
        if (g_pose_s.state < 0) return;
        if (g_pose_s.state == 0) {
            std::string p = g_pose_s.path.empty()
                ? model_dir() + "/pose_estimation_mediapipe_2023mar.onnx"
                : g_pose_s.path;
            if (!path_exists(p)) {
                g_pose_s.state = -1;
                warn_once("pose_model",
                    "posefx: no pose model — run 'vcam-setup --models-pose'");
                return;
            }
            if (!ort_pose_init(p.c_str())) { g_pose_s.state = -1; return; }
            g_pose_s.state = 1;
            warn_once("pose_on", "posefx: body landmarks live — vcam can "
                      "see your skeleton now");
        }
        cv::Mat img(h, w, CV_8UC3, data, stride);
        g_pose_s.tick++;
        bool infer_now = !(g_pose_s.skip_left > 0 && g_pose_s.have);
        if (infer_now) {
            auto t0 = std::chrono::steady_clock::now();
            /* body crop: from the face box when we have one (the crop
             * convention BlazePose expects: whole body, square) */
            int bx = 0, by = 0, bs = std::min(w, h);
            {
                std::lock_guard<std::mutex> fl(g_face.mtx);
                const int TW = 320;
                double fs = (img.cols > TW) ? (double)TW / img.cols : 1.0;
                int dw = std::max(16, (int)(img.cols * fs));
                int dh = std::max(16, (int)(img.rows * fs));
                if (ensure_face(dw, dh)) {
                    if (tick(g_face.counter) % INFER_EVERY == 0) {
                        cv::Mat sm2; cv::resize(img, sm2, cv::Size(dw, dh));
                        cv::Mat fc2; g_face.det->detect(sm2, fc2);
                        fc2.copyTo(g_face.faces);
                    }
                }
                if (!g_face.faces.empty()) {
                    const float *ff = g_face.faces.ptr<float>(0);
                    double iv2 = 1.0 / fs;
                    int fx2 = (int)(ff[0]*iv2), fy2 = (int)(ff[1]*iv2);
                    int fw2 = (int)(ff[2]*iv2), fh2 = (int)(ff[3]*iv2);
                    /* size FIRST (clamped to what fits), THEN center on
                     * the face — the old order shrank the box after
                     * placing it, sliding you half out of the crop */
                    int S = (int)(fw2 * 5.4f);
                    if (S < h/3) S = h/3;
                    bs = std::min(S, std::min(w, h));
                    bx = fx2 + fw2/2 - bs/2;
                    by = fy2 - (int)(fh2*0.9f);
                } else {
                    bs = std::min(w, h);
                    bx = (w - bs)/2;
                    by = 0;
                }
                bs = std::max(64, std::min(bs, std::min(w, h)));
                bx = std::max(0, std::min(w - bs, bx));
                by = std::max(0, std::min(h - bs, by));
            }
            cv::Mat crop = img(cv::Rect(bx, by, bs, bs));
            cv::Mat small2, rgb;
            cv::resize(crop, small2, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
            cv::cvtColor(small2, rgb, cv::COLOR_BGR2RGB);
            std::vector<float> in((size_t)3*256*256);
            for (int i = 0; i < 256*256; i++) {           /* NHWC [0,1] */
                const cv::Vec3b &p2 = rgb.at<cv::Vec3b>(i/256, i%256);
                in[(size_t)i*3+0] = p2[0]/255.0f;
                in[(size_t)i*3+1] = p2[1]/255.0f;
                in[(size_t)i*3+2] = p2[2]/255.0f;
            }
            float lm195[195], conf = 0;
            if (!ort_pose_infer(in.data(), lm195, &conf)) {
                g_pose_s.state = -1;
                return;
            }
            if (conf < 0.0f || conf > 1.0f)               /* raw logit */
                conf = 1.0f / (1.0f + std::exp(-conf));
            if (conf >= 0.50f) {
                float sx2 = (float)bs / 256.0f;
                float nx2[33], ny2[33], nv2[33];
                for (int k2 = 0; k2 < 33; k2++) {
                    nx2[k2] = bx + lm195[k2*5+0] * sx2;
                    ny2[k2] = by + lm195[k2*5+1] * sx2;
                    nv2[k2] = 1.0f/(1.0f+std::exp(-lm195[k2*5+3]));
                }
                /* outlier rejection: if the whole set teleported, it's a
                 * bad frame — don't let it through */
                bool jumped = false;
                if (g_pose_s.have) {
                    float acc = 0;
                    for (int k2 = 0; k2 < 33; k2++)
                        acc += std::fabs(nx2[k2]-g_pose_s.lm[k2][0])
                             + std::fabs(ny2[k2]-g_pose_s.lm[k2][1]);
                    float allow = w * 0.10f * (1 + g_pose_s.skip_left);
                    if (acc / 33.0f > std::max(w*0.18f, allow)) jumped = true;
                }
                if (jumped) {
                    if (g_pose_s.good > 0) g_pose_s.good--;
                } else {
                    for (int k2 = 0; k2 < 33; k2++) {
                        if (g_pose_s.have) {              /* EMA smooth */
                            g_pose_s.lm[k2][0] += (nx2[k2]-g_pose_s.lm[k2][0])*0.35f;
                            g_pose_s.lm[k2][1] += (ny2[k2]-g_pose_s.lm[k2][1])*0.35f;
                            g_pose_s.lm[k2][2] += (nv2[k2]-g_pose_s.lm[k2][2])*0.35f;
                        } else {
                            g_pose_s.lm[k2][0]=nx2[k2];
                            g_pose_s.lm[k2][1]=ny2[k2];
                            g_pose_s.lm[k2][2]=nv2[k2];
                        }
                    }
                    g_pose_s.have = 1;
                    if (g_pose_s.good < 6) g_pose_s.good++;
                }
            } else {
                if (g_pose_s.good > 0) g_pose_s.good--;
                if (g_pose_s.good <= 0) { g_pose_s.good = 0; g_pose_s.have = 0; }
            }
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            g_pose_s.ema_ms = g_pose_s.ema_ms <= 0 ? ms
                            : 0.7*g_pose_s.ema_ms + 0.3*ms;
            g_pose_s.skip_left = (g_pose_s.ema_ms <= 30) ? 0
                               : (g_pose_s.ema_ms <= 60) ? 1
                               : (g_pose_s.ema_ms <= 110) ? 2 : 4;
        } else {
            g_pose_s.skip_left--;
        }
        if (!g_pose_s.have || g_pose_s.good < 2) return;   /* debounced */
        float (*L)[3] = g_pose_s.lm;
        auto P = [&](int i){ return cv::Point((int)L[i][0], (int)L[i][1]); };
        auto vis = [&](int i){
            return L[i][2] > 0.55f &&
                   L[i][0] > -w*0.02f && L[i][0] < w*1.02f &&
                   L[i][1] > -h*0.02f && L[i][1] < h*1.02f;   /* in frame */
        };

        if (mode == 1) img *= 0.28;                       /* neon bones: dark */
        if (mode == 0 || mode == 1) {
            /* SKELETON: glowing bones + joints over you (0) or in the
             * dark (1). fx = glow size. */
            float hue = g_pose_s.tick * 0.03f;
            cv::Scalar col((std::sin(hue+4.19f)*.5f+.5f)*255,
                           (std::sin(hue+2.09f)*.5f+.5f)*255,
                           (std::sin(hue)*.5f+.5f)*255);
            int th = std::max(2, (int)(4*fx));
            for (unsigned b2 = 0; b2 < sizeof POSE_BONES/sizeof *POSE_BONES; b2++) {
                int a2 = POSE_BONES[b2][0], c2 = POSE_BONES[b2][1];
                if (!vis(a2) || !vis(c2)) continue;
                cv::line(img, P(a2), P(c2), col*0.35, th+6, cv::LINE_AA);
                cv::line(img, P(a2), P(c2), col, th, cv::LINE_AA);
            }
            for (int k2 = 0; k2 < 33; k2++)
                if (vis(k2))
                    cv::circle(img, P(k2), std::max(2,(int)(3*fx)),
                               cv::Scalar(255,255,255), cv::FILLED, cv::LINE_AA);
        } else if (mode == 2) {
            /* ENERGY HANDS: charged orbs in your actual palms — bring
             * your wrists together to merge them into one big charge. */
            unsigned tk = g_pose_s.tick;
            int wl2 = 15, wr2 = 16;
            float dx = L[wl2][0]-L[wr2][0], dy = L[wl2][1]-L[wr2][1];
            float d = std::sqrt(dx*dx + dy*dy);
            bool merged = vis(wl2) && vis(wr2) && d < w*0.09f;
            for (int s2 = 0; s2 < (merged ? 1 : 2); s2++) {
                int i2 = s2 ? wr2 : wl2;
                if (!vis(i2) && !merged) continue;
                cv::Point c2 = merged
                    ? cv::Point((int)((L[wl2][0]+L[wr2][0])/2),
                                (int)((L[wl2][1]+L[wr2][1])/2))
                    : P(i2);
                float R = (merged ? w*0.075f : w*0.035f) * fx
                        * (1.0f + 0.12f*std::sin(tk*0.25f + s2*2));
                for (int g2 = 4; g2 >= 1; g2--)
                    cv::circle(img, c2, (int)(R*g2/2.5f),
                               cv::Scalar(255,140+g2*20,40), 2, cv::LINE_AA);
                cv::circle(img, c2, (int)(R*0.45f),
                           cv::Scalar(255,240,200), cv::FILLED, cv::LINE_AA);
                for (int k2 = 0; k2 < 7; k2++) {          /* crackle */
                    float aa = uhash(k2, tk/2, s2)*6.283f;
                    float rr = R*(0.6f+0.5f*uhash(k2, tk/2, s2+9));
                    cv::line(img, c2,
                        {c2.x+(int)(std::cos(aa)*rr), c2.y+(int)(std::sin(aa)*rr)},
                        cv::Scalar(255,220,120), 1, cv::LINE_AA);
                }
            }
        } else if (mode == 4) {
            /* LIGHT WINGS: glowing wings of light unfold from your
             * shoulders, slowly beating. fx = span. */
            unsigned tk = g_pose_s.tick;
            if (vis(11) && vis(12)) {
                float flap = 0.82f + 0.18f * std::sin(tk * 0.07f);
                for (int s2 = 0; s2 < 2; s2++) {
                    cv::Point sh = P(s2 ? 12 : 11);
                    float dir = s2 ? 1.0f : -1.0f;
                    float span = w * 0.16f * fx * flap;
                    for (int fth = 0; fth < 7; fth++) {   /* feather strokes */
                        float t2 = fth / 6.0f;
                        float ang = -1.15f + t2 * 1.35f;  /* fan downward */
                        float len = span * (1.0f - 0.45f*t2);
                        cv::Point tip(sh.x + (int)(dir*std::cos(ang)*len),
                                      sh.y + (int)(std::sin(ang)*len*0.9f));
                        cv::Point mid(sh.x + (int)(dir*std::cos(ang-0.25f)*len*0.55f),
                                      sh.y + (int)(std::sin(ang-0.25f)*len*0.5f));
                        int br = (int)(200 - t2*90);
                        cv::line(img, sh, mid, cv::Scalar(br, br, 255), 5, cv::LINE_AA);
                        cv::line(img, mid, tip, cv::Scalar(br/2, br/2, 230), 3, cv::LINE_AA);
                        cv::line(img, sh, mid, cv::Scalar(255,255,255), 1, cv::LINE_AA);
                    }
                }
            }
        } else if (mode == 5) {
            /* PUPPET STRINGS: fine strings from above hold your head
             * and hands — you're the marionette. */
            unsigned tk = g_pose_s.tick;
            int anchors[3] = {0, 15, 16};                 /* head, wrists */
            for (int a2 = 0; a2 < 3; a2++) {
                int j = anchors[a2];
                if (!vis(j)) continue;
                cv::Point p2 = P(j);
                int topx = p2.x + (int)(std::sin(tk*0.04f + a2*2.1f) * 14);
                cv::line(img, {topx, 0}, p2, cv::Scalar(220,220,225), 1, cv::LINE_AA);
                cv::line(img, {topx+1, 0}, {p2.x+1, p2.y},
                         cv::Scalar(120,120,125), 1, cv::LINE_AA);
                /* little cross-bar at the top */
                cv::line(img, {topx-14, 6}, {topx+14, 6},
                         cv::Scalar(60,80,120), 4, cv::LINE_AA);
            }
        } else {
            /* WRIST TRAILS: your hands write comet trails through the
             * air — conduct, draw, dance. fx = trail thickness. */
            int nT = 24;
            if (g_pose_s.trail_n < nT) g_pose_s.trail_n++;
            for (int s2 = 0; s2 < 2; s2++) {
                for (int i2 = nT-1; i2 > 0; i2--) {
                    g_pose_s.trailx[s2][i2] = g_pose_s.trailx[s2][i2-1];
                    g_pose_s.traily[s2][i2] = g_pose_s.traily[s2][i2-1];
                }
                int wj = s2 ? 16 : 15;
                g_pose_s.trailx[s2][0] = L[wj][0];
                g_pose_s.traily[s2][0] = L[wj][1];
                for (int i2 = 0; i2 < g_pose_s.trail_n-1; i2++) {
                    float k2 = 1.0f - (float)i2/nT;
                    cv::Scalar col = s2 ? cv::Scalar(80,160*k2,255*k2)
                                        : cv::Scalar(255*k2,180*k2,60);
                    cv::line(img,
                        {(int)g_pose_s.trailx[s2][i2],(int)g_pose_s.traily[s2][i2]},
                        {(int)g_pose_s.trailx[s2][i2+1],(int)g_pose_s.traily[s2][i2+1]},
                        col, std::max(1,(int)(k2*7*fx)), cv::LINE_AA);
                }
            }
        }
    } catch (const std::exception &e) {
        g_pose_s.state = -1;
        warn_once("pose_fail", std::string("posefx failed: ") + e.what());
    }
}

/* ── depthfx: monocular-depth effects via the ORT backend ──────── */
struct DepthState {
    std::string path;
    int state = 0;                 /* 0 untried, 1 up, -1 dead */
    cv::Mat map;                   /* full-size CV_32F, 0=far..1=near */
    float lo = 0, hi = 1;          /* EMA-smoothed normalization range */
    double ema_ms = 0;
    int skip_left = 0;
    unsigned tick = 0;
    std::mutex mtx;
};
static DepthState g_depth;

int cv_set_depth_model(const char *path)
{
    if (!path || !*path || !path_exists(path)) return 0;
    std::lock_guard<std::mutex> l(g_depth.mtx);
    g_depth.path = path;
    g_depth.state = 0;
    ort_depth_shutdown();
    g_depth.map.release();
    return 1;
}

void cv_fx_depth(unsigned char *data, int w, int h, int stride,
                 int mode, float fx)
{
    if (!data || w < 64 || h < 64) return;
    if (mode < 0) mode = 0;
    if (mode > 3) mode = 3;
    if (fx <= 0.0f) fx = 1.0f;
    if (fx < 0.2f) fx = 0.2f;
    if (fx > 3.0f) fx = 3.0f;
    if (!ort_available()) {
        warn_once("depth_ort", "depthfx needs the ONNX Runtime backend — "
                  "vcam-setup --ort, then make OPENCV=1 ORT=1");
        return;
    }
    try {
        std::lock_guard<std::mutex> l(g_depth.mtx);
        if (g_depth.state < 0) return;
        if (g_depth.state == 0) {
            std::string p = g_depth.path.empty()
                ? model_dir() + "/model-small.onnx" : g_depth.path;
            if (!path_exists(p)) {
                g_depth.state = -1;
                warn_once("depth_model",
                    "depthfx: no depth model — run 'vcam-setup --models-depth' "
                    "(MiDaS v2.1 small) or 'vcam-ctl filter depthmodel <path>'");
                return;
            }
            if (!ort_depth_init(p.c_str())) { g_depth.state = -1; return; }
            g_depth.state = 1;
            warn_once("depth_on",
                "depthfx: monocular depth live — every pixel now has a distance");
        }
        cv::Mat img(h, w, CV_8UC3, data, stride);
        g_depth.tick++;
        bool infer_now = !(g_depth.skip_left > 0 && !g_depth.map.empty()
                           && g_depth.map.size() == img.size());
        if (infer_now) {
            auto t0 = std::chrono::steady_clock::now();
            cv::Mat small2, rgb;
            cv::resize(img, small2, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
            cv::cvtColor(small2, rgb, cv::COLOR_BGR2RGB);
            std::vector<float> in((size_t)3*256*256), out((size_t)256*256);
            for (int i = 0; i < 256*256; i++) {           /* RGB [0,1] CHW */
                const cv::Vec3b &p2 = rgb.at<cv::Vec3b>(i/256, i%256);
                in[0*65536+i] = p2[0]/255.0f;
                in[1*65536+i] = p2[1]/255.0f;
                in[2*65536+i] = p2[2]/255.0f;
            }
            if (!ort_depth_infer(in.data(), out.data())) {
                g_depth.state = -1;
                return;
            }
            /* normalize with EMA'd min/max so the scene doesn't flicker */
            float mn = out[0], mx = out[0];
            for (int i = 1; i < 65536; i++) {
                if (out[i] < mn) mn = out[i];
                if (out[i] > mx) mx = out[i];
            }
            if (g_depth.map.empty()) { g_depth.lo = mn; g_depth.hi = mx; }
            g_depth.lo += (mn - g_depth.lo) * 0.15f;
            g_depth.hi += (mx - g_depth.hi) * 0.15f;
            float rng = std::max(1e-4f, g_depth.hi - g_depth.lo);
            cv::Mat d256(256, 256, CV_32F);
            for (int i = 0; i < 65536; i++) {
                float v = (out[i] - g_depth.lo) / rng;
                d256.at<float>(i/256, i%256) =
                    v < 0 ? 0.0f : (v > 1 ? 1.0f : v);
            }
            cv::resize(d256, g_depth.map, img.size(), 0, 0, cv::INTER_LINEAR);
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            g_depth.ema_ms = g_depth.ema_ms <= 0 ? ms
                           : 0.7*g_depth.ema_ms + 0.3*ms;
            g_depth.skip_left = (g_depth.ema_ms <= 30) ? 0
                              : (g_depth.ema_ms <= 60) ? 1
                              : (g_depth.ema_ms <= 110) ? 2 : 4;
        } else {
            g_depth.skip_left--;
        }
        if (g_depth.map.empty() || g_depth.map.size() != img.size()) return;
        const cv::Mat &D = g_depth.map;               /* 1 = near, 0 = far */

        if (mode == 0) {
            /* DEPTH BLUR: true depth-of-field — blur grows with
             * distance, continuously, no person mask needed. */
            cv::Mat b1, b2;
            cv::GaussianBlur(img, b1, cv::Size(0,0), 2.0 + 1.5*fx);
            cv::GaussianBlur(b1, b2, cv::Size(0,0), 3.0 + 2.5*fx);
            for (int y = 0; y < h; y++) {
                const float *dd = D.ptr<float>(y);
                const cv::Vec3b *p1 = b1.ptr<cv::Vec3b>(y);
                const cv::Vec3b *p2 = b2.ptr<cv::Vec3b>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < w; x++) {
                    float far2 = 1.0f - dd[x];            /* 0 near, 1 far */
                    if (far2 < 0.25f) continue;           /* near: sharp */
                    float k = (far2 - 0.25f) / 0.75f;
                    const cv::Vec3b &src = (k > 0.55f) ? p2[x] : p1[x];
                    float a = std::min(1.0f, k * 1.6f);
                    ip[x][0]=(uchar)(ip[x][0]+(src[0]-ip[x][0])*a);
                    ip[x][1]=(uchar)(ip[x][1]+(src[1]-ip[x][1])*a);
                    ip[x][2]=(uchar)(ip[x][2]+(src[2]-ip[x][2])*a);
                }
            }
        } else if (mode == 1) {
            /* DEPTH FOG: atmosphere thickens with distance — the room
             * recedes into haze the way real air does. */
            for (int y = 0; y < h; y++) {
                const float *dd = D.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < w; x++) {
                    float k = (1.0f - dd[x]);
                    k = k * k * 0.85f * std::min(1.5f, fx);
                    if (k < 0.02f) continue;
                    if (k > 0.9f) k = 0.9f;
                    ip[x][0]=(uchar)(ip[x][0]+(205-ip[x][0])*k);
                    ip[x][1]=(uchar)(ip[x][1]+(200-ip[x][1])*k);
                    ip[x][2]=(uchar)(ip[x][2]+(190-ip[x][2])*k);
                }
            }
        } else if (mode == 2) {
            /* DEPTH VIEW: the raw superpower, colormapped — near glows
             * warm, far falls to deep blue. */
            for (int y = 0; y < h; y++) {
                const float *dd = D.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < w; x++) {
                    float v = dd[x];
                    float r2 = std::min(1.0f, v*1.8f);
                    float g2 = v > 0.5f ? (v-0.5f)*2.0f : v*0.6f;
                    float b2 = std::max(0.0f, 1.0f - v*1.6f);
                    ip[x][0]=(uchar)(b2*235);
                    ip[x][1]=(uchar)(g2*225);
                    ip[x][2]=(uchar)(r2*245);
                }
            }
        } else {
            /* PARALLAX WIGGLE: near pixels sway more than far ones —
             * the fake-3D wiggle-gram, live. */
            float ph = std::sin(g_depth.tick * 0.06f) * 7.0f * fx;
            cv::Mat keep = img.clone();
            for (int y = 0; y < h; y++) {
                const float *dd = D.ptr<float>(y);
                cv::Vec3b *ip = img.ptr<cv::Vec3b>(y);
                for (int x = 0; x < w; x++) {
                    int sx2 = x + (int)(ph * (dd[x] - 0.45f));
                    if (sx2 < 0) sx2 = 0;
                    if (sx2 >= w) sx2 = w - 1;
                    ip[x] = keep.at<cv::Vec3b>(y, sx2);
                }
            }
        }
    } catch (const std::exception &e) {
        g_depth.state = -1;
        warn_once("depth_fail", std::string("depthfx failed: ") + e.what());
    }
}

/* ── genai: generic img2img via the ORT backend ────────────────── */
struct GenAiState {
    std::string path;         /* model file; empty = use default */
    int state = 0;            /* 0 untried, 1 up, -1 dead */
    cv::Mat last;             /* last transformed frame (full size) */
    double ema_ms = 0;
    int skip_left = 0;
    std::mutex mtx;
};
static GenAiState g_genai;

int cv_set_gen_model(const char *path)
{
    if (!path || !*path) return 0;
    if (!path_exists(path)) return 0;
    std::lock_guard<std::mutex> l(g_genai.mtx);
    g_genai.path = path;
    g_genai.state = 0;            /* (re)load lazily */
    ort_i2i_shutdown();
    g_genai.last.release();
    return 1;
}

void cv_clear_gen_model(void)
{
    std::lock_guard<std::mutex> l(g_genai.mtx);
    g_genai.path.clear();
    g_genai.state = 0;
    ort_i2i_shutdown();
    g_genai.last.release();
}

void cv_fx_genai(unsigned char *data, int w, int h, int stride,
                 float blend, float size, float range01)
{
    if (!data || w < 32 || h < 32) return;
    bool r01 = range01 >= 0.5f;      /* p3=1: [0,1] convention (Zero-DCE etc.) */
    if (blend <= 0.0f) blend = 1.0f;
    if (blend < 0.2f) blend = 0.2f;
    if (blend > 1.0f) blend = 1.0f;
    int iw0 = (int)size;
    if (iw0 <= 0) iw0 = 288;
    if (iw0 < 128) iw0 = 128;
    if (iw0 > 512) iw0 = 512;
    if (!ort_available()) {
        warn_once("genai_ort", "genai needs the ONNX Runtime backend — "
                  "vcam-setup --ort, then make OPENCV=1 ORT=1");
        return;
    }
    try {
        std::lock_guard<std::mutex> l(g_genai.mtx);
        if (g_genai.state < 0) return;
        if (g_genai.state == 0) {
            std::string p = g_genai.path.empty()
                ? model_dir() + "/img2img.onnx" : g_genai.path;
            if (!path_exists(p)) {
                g_genai.state = -1;
                warn_once("genai_model",
                    "genai: no model — drop an img2img ONNX (e.g. AnimeGAN) "
                    "at " + p + " or set one with "
                    "'vcam-ctl filter genmodel <path>'");
                return;
            }
            if (!ort_i2i_init(p.c_str())) { g_genai.state = -1; return; }
            g_genai.state = 1;
            warn_once("genai_on",
                "genai: img2img model live on ONNX Runtime — "
                "every frame is now the network's opinion of you");
        }
        cv::Mat img(h, w, CV_8UC3, data, stride);
        bool infer_now = !(g_genai.skip_left > 0 && !g_genai.last.empty()
                           && g_genai.last.size() == img.size());
        if (infer_now) {
            auto t0 = std::chrono::steady_clock::now();
            int iw = std::max(64, (iw0/8)*8);
            int ih = std::max(64, ((int)((double)iw*h/w)/8)*8);
            cv::Mat small2, rgb;
            cv::resize(img, small2, cv::Size(iw, ih), 0, 0, cv::INTER_AREA);
            cv::cvtColor(small2, rgb, cv::COLOR_BGR2RGB);
            int N = iw*ih;
            std::vector<float> in((size_t)3*N), out((size_t)3*N);
            if (ort_i2i_layout() == 1) {                  /* NHWC */
                for (int i = 0; i < N; i++) {
                    const cv::Vec3b &p2 = rgb.at<cv::Vec3b>(i/iw, i%iw);
                    float d2 = r01 ? 255.0f : 127.5f, o2 = r01 ? 0.0f : 1.0f;
                    in[(size_t)i*3+0] = p2[0]/d2 - o2;
                    in[(size_t)i*3+1] = p2[1]/d2 - o2;
                    in[(size_t)i*3+2] = p2[2]/d2 - o2;
                }
            } else {                                      /* NCHW */
                for (int i = 0; i < N; i++) {
                    const cv::Vec3b &p2 = rgb.at<cv::Vec3b>(i/iw, i%iw);
                    float d2 = r01 ? 255.0f : 127.5f, o2 = r01 ? 0.0f : 1.0f;
                    in[0*(size_t)N+i] = p2[0]/d2 - o2;
                    in[1*(size_t)N+i] = p2[1]/d2 - o2;
                    in[2*(size_t)N+i] = p2[2]/d2 - o2;
                }
            }
            if (!ort_i2i_infer(in.data(), iw, ih, out.data())) {
                g_genai.state = -1;
                return;
            }
            cv::Mat res(ih, iw, CV_8UC3);
            if (ort_i2i_layout() == 1) {
                for (int i = 0; i < N; i++) {
                    cv::Vec3b &p2 = res.at<cv::Vec3b>(i/iw, i%iw);
                    float m2 = r01 ? 255.0f : 127.5f, a2 = r01 ? 0.0f : 1.0f;
                    p2[2]=clamp_int_u8((int)((out[(size_t)i*3+0]+a2)*m2));
                    p2[1]=clamp_int_u8((int)((out[(size_t)i*3+1]+a2)*m2));
                    p2[0]=clamp_int_u8((int)((out[(size_t)i*3+2]+a2)*m2));
                }
            } else {
                for (int i = 0; i < N; i++) {
                    cv::Vec3b &p2 = res.at<cv::Vec3b>(i/iw, i%iw);
                    float m2 = r01 ? 255.0f : 127.5f, a2 = r01 ? 0.0f : 1.0f;
                    p2[2]=clamp_int_u8((int)((out[0*(size_t)N+i]+a2)*m2));
                    p2[1]=clamp_int_u8((int)((out[1*(size_t)N+i]+a2)*m2));
                    p2[0]=clamp_int_u8((int)((out[2*(size_t)N+i]+a2)*m2));
                }
            }
            cv::resize(res, g_genai.last, img.size(), 0, 0, cv::INTER_LINEAR);
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            g_genai.ema_ms = g_genai.ema_ms <= 0 ? ms
                           : 0.7*g_genai.ema_ms + 0.3*ms;
            g_genai.skip_left = (g_genai.ema_ms <= 30) ? 0
                              : (g_genai.ema_ms <= 60) ? 1
                              : (g_genai.ema_ms <= 110) ? 2 : 4;
        } else {
            g_genai.skip_left--;
        }
        if (!g_genai.last.empty() && g_genai.last.size() == img.size()) {
            if (blend >= 0.99f) g_genai.last.copyTo(img);
            else cv::addWeighted(g_genai.last, blend, img, 1.0-blend, 0, img);
        }
    } catch (const std::exception &e) {
        g_genai.state = -1;
        warn_once("genai_fail", std::string("genai failed: ") + e.what());
    }
}

int cv_set_face_image(const char *path)
{
    if (!path || !*path) return 0;
    try {
        cv::Mat m = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (m.empty()) return 0;
        if (m.channels() == 3)                  /* no alpha: make opaque */
            cv::cvtColor(m, m, cv::COLOR_BGR2BGRA);
        if (m.channels() != 4) return 0;
        std::lock_guard<std::mutex> l(g_faceimg.mtx);
        m.copyTo(g_faceimg.rgba);
        return 1;
    } catch (const std::exception &e) {
        warn_once("faceimg", std::string("face image load failed: ") + e.what());
        return 0;
    }
}

void cv_clear_face_image(void)
{
    try {
        std::lock_guard<std::mutex> l(g_faceimg.mtx);
        g_faceimg.rgba.release();
    } catch (...) { }
}

int cv_set_bg_image(const char *path)
{
    if (!path || !*path) return 0;
    try {
        cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
        if (m.empty()) return 0;
        std::lock_guard<std::mutex> l(g_bgimg.mtx);
        m.copyTo(g_bgimg.orig);
        g_bgimg.scaled.release();     /* re-cover on next frame */
        return 1;
    } catch (const std::exception &e) {
        warn_once("bgimg", std::string("bg image load failed: ") + e.what());
        return 0;
    }
}

void cv_clear_bg_image(void)
{
    try {
        std::lock_guard<std::mutex> l(g_bgimg.mtx);
        g_bgimg.orig.release();
        g_bgimg.scaled.release();
    } catch (...) { }
}

void cv_shutdown(void)
{
    ort_shutdown();
    try {
        { std::lock_guard<std::mutex> l(g_face.mtx); g_face.det.reset(); g_face.faces.release(); }
        { std::lock_guard<std::mutex> l(g_seg.mtx);  g_seg.net = cv::dnn::Net(); g_seg.rvm = cv::dnn::Net(); g_seg.mask.release(); g_seg.bg_est.release(); g_seg.r1.release(); g_seg.r2.release(); g_seg.r3.release(); g_seg.r4.release(); }
        { std::lock_guard<std::mutex> l(g_classic.mtx);
          g_classic.mog2.reset(); g_classic.prev_gray.release(); }
        { std::lock_guard<std::mutex> l(g_style.mtx);
          for (int i = 0; i < 5; i++) g_style.net[i] = cv::dnn::Net();
          g_style.ema.release(); }
        { std::lock_guard<std::mutex> l(g_mood.mtx);
          g_mood.net = cv::dnn::Net(); }
        { std::lock_guard<std::mutex> l(g_paint.mtx);
          g_paint.canvas.release(); g_paint.prev_gray.release();
          g_paint.mapx.release(); g_paint.mapy.release(); }
        { std::lock_guard<std::mutex> l(g_bgimg.mtx);
          g_bgimg.orig.release(); g_bgimg.scaled.release(); }
        { std::lock_guard<std::mutex> l(g_faceimg.mtx);
          g_faceimg.rgba.release(); }
    } catch (...) { /* shutdown must never throw */ }
}

} // extern "C"
