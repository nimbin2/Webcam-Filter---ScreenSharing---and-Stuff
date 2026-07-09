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
cv::Mat rvm_infer(const cv::Mat &bgr)
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
                const int IW = 512;   /* ORT computes correctly; afford quality */
                double sc = (double)IW / bgr.cols;
                int iw = std::max(64, ((int)(bgr.cols*sc)/16)*16);
                int ih = std::max(64, ((int)(bgr.rows*sc)/16)*16);
                cv::Mat small2;
                cv::resize(bgr, small2, cv::Size(iw, ih), 0, 0, cv::INTER_AREA);
                cv::Mat src = cv::dnn::blobFromImage(small2, 1.0/255.0,
                                  cv::Size(iw, ih), cv::Scalar(), true, false);
                cv::Mat pha(ih, iw, CV_32F);
                if (ort_rvm_infer((const float*)src.ptr<float>(0), iw, ih,
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

void cv_fx_face(unsigned char *data, int w, int h, int stride, int mode)
{
    if (!data || w < 16 || h < 16) return;
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
                float scale = 1.45f;
                if (mode == 6)
                    scale = 1.30f + 0.28f * std::sin((float)g_face.counter * 0.35f);
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
                    cv::line(img, eye, tip, cv::Scalar(20,20,140), std::max(6, fw/8));
                    cv::line(img, eye, tip, cv::Scalar(40,40,255), std::max(3, fw/16));
                    cv::line(img, eye, tip, cv::Scalar(220,220,255), std::max(1, fw/40));
                    cv::circle(img, eye, std::max(4, fw/10),
                               cv::Scalar(60,60,255), cv::FILLED);
                    cv::circle(img, eye, std::max(2, fw/20),
                               cv::Scalar(255,255,255), cv::FILLED);
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
                float a3=(float)(g_face.counter%90)/90.0f*6.2832f;
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

void cv_fx_bgblur(unsigned char *data, int w, int h, int stride, int mode,
                  float edge, float smooth)
{
    if (!data || w < 16 || h < 16) return;
    if (edge <= 0.0f) edge = 0.22f;          /* mask generosity: lower = more hair */
    if (edge < 0.05f) edge = 0.05f;
    if (edge > 0.50f) edge = 0.50f;
    if (smooth <= 0.0f) smooth = 0.80f;      /* responsiveness: 1 = instant */
    if (smooth < 0.40f) smooth = 0.40f;
    if (smooth > 1.00f) smooth = 1.00f;
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
            cv::Mat m = rvm_infer(img);
            if (!m.empty()) {
                /* RVM alpha is already clean; light remap honors 'edge' */
                m = (m - edge*0.5) * (1.0/(1.0 - edge*0.5));
                cv::min(m, 1.0, m); cv::max(m, 0.0, m);
            } else if (ensure_seg()) {
                m = seg_infer(img, body, edge);
            } else {
                return;   /* RVM just died and pphumanseg is absent */
            }
            if (!m.empty()) {
                if (!g_seg.mask.empty() && g_seg.mask.size() == m.size())
                    cv::addWeighted(m, smooth, g_seg.mask, 1.0 - smooth, 0.0, g_seg.mask);
                else
                    m.copyTo(g_seg.mask);
            }
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
                int mode, float gain)
{
    if (!data || w < 32 || h < 32) return;
    if (mode < 0) mode = 0;
    if (mode > 5) mode = 5;
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
                    if (g_mood.pl[i] <= 0 && (rand() % 100) < (int)(conf*25)) {
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
    } catch (...) { /* shutdown must never throw */ }
}

} // extern "C"
