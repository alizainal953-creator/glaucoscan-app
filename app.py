import os, io, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlaucoScan · IDSC 2026",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:       #0a0f1a;
    --bg-card:       #111827;
    --bg-card2:      #1a2436;
    --accent-teal:   #00d4c8;
    --accent-blue:   #3b82f6;
    --accent-amber:  #f59e0b;
    --accent-rose:   #f43f5e;
    --accent-green:  #10b981;
    --text-primary:  #f0f6ff;
    --text-muted:    #8899bb;
    --border:        rgba(59,130,246,0.18);
    --glow-teal:     0 0 30px rgba(0,212,200,0.15);
    --glow-blue:     0 0 30px rgba(59,130,246,0.2);
}

/* ── App background ── */
.stApp {
    background: var(--bg-base);
    background-image:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(0,212,200,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 85% 110%, rgba(59,130,246,0.08) 0%, transparent 55%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 39px,
            rgba(59,130,246,0.025) 39px,
            rgba(59,130,246,0.025) 40px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 39px,
            rgba(59,130,246,0.025) 39px,
            rgba(59,130,246,0.025) 40px
        );
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a1020 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Hide default header/footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main typography ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text-primary) !important;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue));
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--glow-teal);
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
}
.metric-value {
    font-size: 34px;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
    color: var(--accent-teal);
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section title ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: var(--text-primary);
    margin: 8px 0 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
    margin-left: 12px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1528 50%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(0,212,200,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 42px;
    line-height: 1.15;
    color: #f0f6ff;
    margin: 0 0 10px;
    position: relative; z-index: 1;
}
.hero-title span { color: var(--accent-teal); }
.hero-sub {
    font-size: 15px;
    color: var(--text-muted);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    max-width: 520px;
    position: relative; z-index: 1;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,212,200,0.1);
    border: 1px solid rgba(0,212,200,0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-teal);
    margin-bottom: 16px;
    position: relative; z-index: 1;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(0,212,200,0.3) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-teal) !important;
}
/* Make the uploader's "Browse files" button have black text for better visibility */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] [role="button"] {
    color: #0a0f1a !important;
    background: #ffffff !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    font-weight: 600 !important;
}

/* ── Result box ── */
.result-pos {
    background: linear-gradient(135deg, rgba(244,63,94,0.12) 0%, rgba(244,63,94,0.05) 100%);
    border: 1px solid rgba(244,63,94,0.4);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
}
.result-neg {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
}
.result-label-pos {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    color: #f43f5e;
}
.result-label-neg {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    color: #10b981;
}

/* ── Fold table ── */
.fold-table {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: var(--text-muted) !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(0,212,200,0.2), rgba(59,130,246,0.2)) !important;
    color: var(--accent-teal) !important;
    border: 1px solid rgba(0,212,200,0.3) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue)) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4c8, #3b82f6) !important;
    color: #f0f6ff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    transition: all 0.25s !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,200,0.35) !important;
}
/* keep sidebar buttons dark for readability */
[data-testid="stSidebar"] .stButton > button { color: #0a0f1a !important; }

/* ── Info/warning boxes ── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card2) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Separator ── */
.custom-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 24px 0;
}

/* ── Progress bar ── */
.prob-bar-wrap { margin: 6px 0; }
.prob-bar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.prob-bar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill-pos {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f43f5e, #fb7185);
    transition: width 0.8s ease;
}
.prob-bar-fill-neg {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.8s ease;
}

/* ── Sidebar nav items ── */
.sidebar-info {
    background: rgba(0,212,200,0.06);
    border: 1px solid rgba(0,212,200,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: var(--text-muted);
    font-family: 'DM Sans', sans-serif;
}
.sidebar-fold-badge {
    display: inline-flex;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--accent-blue);
    margin: 2px 2px;
}
/* ── Checkboxes ── */
/* Make checkbox labels (e.g. "Show Grad-CAM++ heatmap") white for readability */
[data-testid="stCheckbox"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    vertical-align: middle !important;
}
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] div, [data-testid="stCheckbox"] span {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BACKBONE         = "tf_efficientnet_b3"
ATTENTION_REDUCE = 256
DROPOUT          = 0.4
IMG_SIZE         = 300
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
QUALITY_MIN      = 1.0
QUALITY_MAX      = 10.0
N_FOLDS          = 5
THRESHOLD        = 0.5
DEVICE_TYPE      = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE           = torch.device(DEVICE_TYPE)

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
        )
    def forward(self, x):
        avg   = F.adaptive_avg_pool2d(x, 1)
        mx    = F.adaptive_max_pool2d(x, 1)
        scale = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * scale.unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx    = x.max(dim=1, keepdim=True).values
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale

class HybridAttention(nn.Module):
    def __init__(self, in_channels, reduced_channels=ATTENTION_REDUCE):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )
        self.channel_att = ChannelAttention(reduced_channels)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        x_r = self.reduce(x)
        return torch.cat([self.channel_att(x_r), self.spatial_att(x_r)], dim=1)

class NFResNetHybrid(nn.Module):
    def __init__(self, backbone_name=BACKBONE, pretrained=False,
                 reduced_channels=ATTENTION_REDUCE, dropout=DROPOUT):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            c_out = self.backbone(dummy).shape[1]
        self.attention  = HybridAttention(c_out, reduced_channels)
        self.gap_dim    = 2 * reduced_channels
        self.fusion_dim = self.gap_dim + 1
        self.head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 2),
        )
    def extract_features(self, x, q_norm):
        feat_map = self.backbone(x)
        attn_out = self.attention(feat_map)
        f_gap    = attn_out.mean(dim=[2, 3])
        return torch.cat([f_gap, q_norm.unsqueeze(1)], dim=1)
    def forward(self, x, q_norm):
        v_fusion = self.extract_features(x, q_norm)
        return self.head(v_fusion), v_fusion

# ─── IMAGE PREPROCESSING ──────────────────────────────────────────────────────
import cv2

def crop_black_border(img_np, threshold=10):
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mask   = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img_np
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img_np[y0:y1, x0:x1]

def square_pad(img_np):
    h, w   = img_np.shape[:2]
    diff   = abs(h - w)
    p1, p2 = diff // 2, diff - diff // 2
    if h < w:
        img_np = np.pad(img_np, ((p1, p2), (0, 0), (0, 0)), mode='constant')
    elif w < h:
        img_np = np.pad(img_np, ((0, 0), (p1, p2), (0, 0)), mode='constant')
    return img_np

def preprocess_pil(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black_border(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = square_pad(img)
    return Image.fromarray(img)

from torchvision import transforms as T

def get_val_transforms():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ─── GRAD-CAM++ ──────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    def __init__(self, model, device):
        self.model  = model.eval()
        self.device = device
        self._hooks = []
        self._fmaps = None
        self._grads = None

    def _register_hooks(self, target_layer):
        for h in self._hooks: h.remove()
        self._hooks = []
        def fwd(m, i, o): self._fmaps = o.detach()
        def bwd(m, gi, go): self._grads = go[0].detach()
        self._hooks.append(target_layer.register_forward_hook(fwd))
        self._hooks.append(target_layer.register_full_backward_hook(bwd))

    def generate(self, img_tensor, q_norm_tensor, class_idx=1):
        # Hook ke output attention (attn_out) — ini feature map spatial
        # yang benar-benar dipakai untuk klasifikasi SVM
        # Caranya: hook ke self.model.attention (output-nya masih H×W)
        target_layer = self.model.attention
        self._register_hooks(target_layer)

        img_t = img_tensor.to(self.device)
        q_t   = q_norm_tensor.to(self.device)

        with torch.enable_grad():
            # Forward penuh — kita butuh gradient mengalir dari head ke attention
            logits, _ = self.model(img_t, q_t)
            self.model.zero_grad()
            # Backward dari head NN (class_idx=1 = GON+)
            # Head NN dan SVM dilatih pada fitur yang sama (extract_features)
            # jadi gradient dari head NN terhadap attention = proxy yang valid
            logits[0, class_idx].backward()

        grads = self._grads
        fmaps = self._fmaps
        g2 = grads ** 2
        g3 = grads ** 3
        alpha  = g2 / (2*g2 + (fmaps*g3).sum(dim=[2,3], keepdim=True) + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=[2,3], keepdim=True)
        cam = F.relu((weights * fmaps).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
        mn, mx = cam.min(), cam.max()
        cam = (cam - mn) / (mx - mn + 1e-8)
        for h in self._hooks: h.remove()
        self._hooks = []
        return cam

    def overlay(self, cam, original_np, alpha=0.45):
        h, w = original_np.shape[:2]
        cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        hm = (cm_module.jet(cam_r)[:,:,:3]*255).astype(np.uint8)
        return ((1-alpha)*original_np.astype(np.float32) +
                alpha*hm.astype(np.float32)).clip(0,255).astype(np.uint8)

# ─── LOAD MODELS (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(checkpoint_dir: str):
    import joblib
    cnn_models, svm_pipelines = [], []
    missing = []
    for fold in range(N_FOLDS):
        cnn_path = os.path.join(checkpoint_dir, f"fold{fold}_cnn.pth")
        svm_path = os.path.join(checkpoint_dir, f"fold{fold}_svm.joblib")
        if not os.path.exists(cnn_path) or not os.path.exists(svm_path):
            missing.append(fold)
            continue
        model = NFResNetHybrid(pretrained=False).to(DEVICE)
        state = torch.load(cnn_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        cnn_models.append((fold, model))
        clf = joblib.load(svm_path)
        svm_pipelines.append((fold, clf))
    return cnn_models, svm_pipelines, missing

# ─── PREDICT ─────────────────────────────────────────────────────────────────
def predict_ensemble(pil_img, q_score, cnn_models, svm_pipelines):
    q_norm = float(np.clip((q_score - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN), 0, 1))
    processed = preprocess_pil(pil_img)
    img_t = get_val_transforms()(processed).unsqueeze(0).to(DEVICE)
    q_t   = torch.tensor([q_norm], dtype=torch.float32).to(DEVICE)

    all_probs = []
    per_fold  = []

    for (fold_idx, model), (_, clf) in zip(cnn_models, svm_pipelines):
        with torch.no_grad():
            feats = model.extract_features(img_t, q_t).float().cpu().numpy()
        prob_pos = clf.predict_proba(feats)[0, 1]
        all_probs.append(prob_pos)
        per_fold.append({"fold": fold_idx, "P(GON+)": round(prob_pos, 4)})

    ensemble_prob = float(np.mean(all_probs))
    label = "GON+" if ensemble_prob >= THRESHOLD else "GON-"
    return ensemble_prob, label, per_fold, processed, q_norm

def get_gradcam(model, pil_processed, q_norm):
    img_t = get_val_transforms()(pil_processed).unsqueeze(0)
    q_t   = torch.tensor([q_norm], dtype=torch.float32)

    # Tetap eval() — persis seperti Colab
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    gc  = GradCAMPlusPlus(model, DEVICE)
    cam = gc.generate(img_t, q_t, class_idx=1)

    orig = np.array(pil_processed)
    ov   = gc.overlay(cam, orig)
    return cam, ov, orig

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px;'>
        <div style='font-size:42px'>👁️</div>
        <div style='font-family:"DM Serif Display",serif; font-size:20px; color:#f0f6ff;'>GlaucoScan AI</div>
        <div style='font-size:11px; color:#8899bb; font-family:"JetBrains Mono",monospace; margin-top:4px;'>IDSC 2026 · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Auto-load models on startup (no user action needed) ──
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_models")

    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.cnn_models    = []
        st.session_state.svm_pipelines = []

    if not st.session_state.models_loaded:
        with st.spinner("Loading models…"):
            try:
                cnns, svms, missing = load_models(checkpoint_dir)
                st.session_state.cnn_models    = cnns
                st.session_state.svm_pipelines = svms
                st.session_state.models_loaded = len(cnns) > 0
            except Exception as e:
                st.error(f"Failed to load models: {e}")

    st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)

    # Fold status badges
    st.markdown("<div style='font-size:12px; font-weight:600; letter-spacing:1.2px; text-transform:uppercase; color:#8899bb; margin-bottom:8px;'>Model Status</div>", unsafe_allow_html=True)
    loaded_folds = [f for f, _ in st.session_state.cnn_models]
    badges = ""
    for i in range(N_FOLDS):
        if i in loaded_folds:
            badges += f"<span class='sidebar-fold-badge'>Fold {i} ✓</span>"
        else:
            badges += f"<span style='display:inline-flex;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);border-radius:6px;padding:2px 10px;font-family:JetBrains Mono,monospace;font-size:12px;color:#555;margin:2px;'>Fold {i}</span>"
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='sidebar-info'>
        <b>Device:</b> <code style='color:#00d4c8'>{DEVICE_TYPE.upper()}</code><br>
        <b>Backbone:</b> <code style='font-size:11px'>EfficientNet-B3</code><br>
        <b>Classifier:</b> <code style='font-size:11px'>SVM (RBF)</code><br>
        <b>Ensemble:</b> <code style='font-size:11px'>Mean of {N_FOLDS} folds</code>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class='hero-banner'>
    <div class='hero-badge'>● IDSC 2026 · Glaucoma Detection </div>
    <div class='hero-title'>Glaucomatous <span>Optic</span><br>Neuropathy Detection</div>
    <div style='font-size:15px; color:#ffffff; font-family:"DM Sans",sans-serif; font-weight:300; max-width:520px; position: relative; z-index: 1; margin-bottom:10px;'>
        Dashboard made by abjrit team from Institut Teknologi Sepuluh Nopember that consist of Ali, Rafidah, Najwa, and Maul
    </div>
    <div class='hero-sub'>
        Ensemble model based on EfficientNet-B3 + Hybrid Attention + SVM.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab3, tab2 = st.tabs([
    "🔬  Prediction & Analysis",
    "ℹ️  About Model",
    "📊  Model Performance",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not st.session_state.models_loaded:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.3);
             border-radius:14px; padding:24px 28px; text-align:center; margin: 20px 0;'>
            <div style='font-size:32px'>⏳</div>
            <div style='font-family:"DM Serif Display",serif; font-size:20px; color:#f59e0b; margin:8px 0;'>
                Preparing models…
            </div>
            <div style='color:#8899bb; font-size:14px;'>
                Please wait while the models are being loaded.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_upload, col_cfg = st.columns([3, 2], gap="large")

        with col_upload:
            st.markdown(
                """
                <style>
                .tt_wrapper { display:inline-flex; align-items:center; position:relative; margin-left:8px; }
                .tt_icon { display:inline-flex; align-items:center; justify-content:center; font-size:14px; color:#8899bb; padding:4px 8px; height:28px; line-height:1; border-radius:999px; background:rgba(255,255,255,0.02); cursor:help; }
                .tt_text {
                    position:absolute; left:0; top:110%; width:340px; background:#0a0f1a; color:#f0f6ff;
                    border:1px solid rgba(255,255,255,0.06); padding:10px; font-size:13px; border-radius:8px;
                    box-shadow:0 6px 18px rgba(0,0,0,0.6); visibility:hidden; opacity:0; transition:opacity 0s ease;
                    z-index:999;
                }
                .tt_wrapper:hover .tt_text { visibility:visible; opacity:1; }
                </style>

                <div class='section-title'>Upload Fundus Photo
                    <span class='tt_wrapper' aria-label='info'>
                        <span class='tt_icon'>?</span>
                        <div class='tt_text'>
                            Only standardized fundus images are supported. Images should be captured using a TOPCON DRI OCT Triton (45° FOV) or equivalent settings. Non-compliant images may lead to inaccurate predictions.
                        </div>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Choose fundus retina image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            # reset predict state when a new file is uploaded
            if uploaded is not None:
                if ("uploaded_name" not in st.session_state) or (st.session_state.uploaded_name != getattr(uploaded, 'name', None)):
                    st.session_state.uploaded_name = getattr(uploaded, 'name', None)
                    st.session_state.predict_clicked = False
            # uploaded filename display removed

        with col_cfg:
            st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)
            # Image Quality input: label + numeric input (inline)
            left, right = st.columns([0.6, 0.4])
            with left:
                st.markdown(
                    "<div style='color:#8899bb; display:flex; align-items:center; height:38px;'>Image Quality Score</div>",
                    unsafe_allow_html=True,
                )
            with right:
                q_score = st.number_input(
                    "",
                    value=7.0,
                    format="%.2f",
                    step=0.01,
                    help="Estimate image quality: 1 = very poor, 10 = perfect",
                    label_visibility="collapsed",
                )

            # Show warning only when out of allowed range
            if q_score < QUALITY_MIN or q_score > QUALITY_MAX:
                st.warning(f"Image Quality Score must be between {QUALITY_MIN} and {QUALITY_MAX}. You entered {q_score}.")

            st.markdown(f"""
            <div style='font-family:"JetBrains Mono",monospace; font-size:12px;
                 color:#8899bb; margin-top:-8px;'>
                Q-Norm: {(q_score-1)/9:.3f} &nbsp;|&nbsp; Threshold: {THRESHOLD}
            </div>
            """, unsafe_allow_html=True)

            # Predict button (user explicitly triggers analysis)
            if uploaded is not None:
                if st.button("Predict", key="predict_btn", use_container_width=True):
                    st.session_state.predict_clicked = True

            # Ensemble always uses all folds (option removed)

        if uploaded is not None:
            st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)

            pil_img = Image.open(uploaded).convert("RGB")

            # Always show original and processed previews immediately after upload
            processed_preview = preprocess_pil(pil_img)
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(pil_img, caption="Original input", use_container_width=True)
            with col_b:
                st.image(processed_preview, caption="After preprocessing", use_container_width=True)

            # Only run prediction after user clicks Predict
            if not st.session_state.get("predict_clicked", False):
                st.info("Set Image Quality Score and press \"Predict\" to run the analysis.")

            if st.session_state.get("predict_clicked", False):
                with st.spinner("Analyzing image"):
                    cnns = st.session_state.cnn_models
                    svms = st.session_state.svm_pipelines
                    # always use all folds for ensemble

                    ensemble_prob, label, per_fold, processed_img, q_norm = predict_ensemble(
                        pil_img, q_score, cnns, svms
                    )

            if st.session_state.get("predict_clicked", False):
                is_pos = label == "GON+"
                color_main = "#f43f5e" if is_pos else "#10b981"

                # ── Result header ──
                c1, c2, c3 = st.columns([1.4, 1, 1.6], gap="large")

                with c1:
                    # Compact result card to match neighboring metric cards
                    short_label = 'GON+' if is_pos else 'GON−'
                    short_desc = 'Glaucomatous Optic Neuropathy detected' if is_pos else 'No signs of Glaucomatous Optic Neuropathy'
                    st.markdown(f"""
                    <div class='metric-card' style='height:100%; min-height:120px; text-align:center;'>
                        <div class='metric-label'>Result</div>
                        <div style='font-size:28px; font-weight:700; color:{color_main}; font-family:"DM Serif Display",serif;'>
                            {short_label}
                        </div>
                        <div style='color:#8899bb; font-size:12px; margin-top:6px;'>{short_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown(f"""
                    <div class='metric-card' style='height:100%; min-height:120px;'>
                        <div class='metric-label'>Probability of GON+</div>
                        <div class='metric-value' style='color:{color_main};'>{ensemble_prob*100:.1f}<span style='font-size:18px;'>%</span></div>
                        <div class='metric-sub'>Ensemble {len(cnns)} folds · thr={THRESHOLD}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c3:
                    prob_neg = 1 - ensemble_prob
                    st.markdown(f"""
                    <div class='metric-card' style='height:100%;'>
                        <div class='metric-label'>Probability Distribution</div>
                        <div class='prob-bar-wrap'>
                            <div class='prob-bar-label'>
                                <span>GON+</span><span style='color:#f43f5e;'>{ensemble_prob*100:.1f}%</span>
                            </div>
                            <div class='prob-bar-track'>
                                <div class='prob-bar-fill-pos' style='width:{ensemble_prob*100:.1f}%;'></div>
                            </div>
                        </div>
                        <div class='prob-bar-wrap' style='margin-top:10px;'>
                            <div class='prob-bar-label'>
                                <span>GON−</span><span style='color:#10b981;'>{prob_neg*100:.1f}%</span>
                            </div>
                            <div class='prob-bar-track'>
                                <div class='prob-bar-fill-neg' style='width:{prob_neg*100:.1f}%;'></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)

                # Grad-CAM checkbox will be shown below per-fold details (deferred generation)

                # ── Per-fold breakdown ──
                st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title' style='font-size:18px;'>Per-Fold Details</div>", unsafe_allow_html=True)

                fold_cols = st.columns(len(per_fold))
                for i, fd in enumerate(per_fold):
                    prob = fd["P(GON+)"]
                    clr = "#f43f5e" if prob >= THRESHOLD else "#10b981"
                    with fold_cols[i]:
                        st.markdown(f"""
                        <div class='metric-card' style='text-align:center; padding:16px;'>
                            <div class='metric-label'>Fold {fd['fold']}</div>
                            <div style='font-size:24px; font-weight:700; color:{clr};
                                 font-family:"DM Serif Display",serif;'>
                                {prob*100:.1f}%
                            </div>
                                <div style='font-size:11px; font-family:"JetBrains Mono",monospace;
                                 color:#8899bb; margin-top:4px;'>
                                {'GON+' if prob >= THRESHOLD else 'GON−'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                # ── Grad-CAM (deferred) ──
                st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)
                col_check, col_help = st.columns([0.06, 0.94], gap="small")
                with col_check:
                    show_gradcam = st.checkbox("", value=False, key="show_gradcam", label_visibility="collapsed")
                with col_help:
                    st.markdown(
                        """
                        <style>
                        .custom-gradcam-label { display:inline-flex; align-items:center; gap:8px; vertical-align:middle; height:100%; }
                        .tt_wrapper { display:inline-flex; align-items:center; position:relative; margin-left:8px; }
                        .tt_icon { display:inline-flex; align-items:center; justify-content:center; font-size:12px; color:#8899bb; padding:4px 8px; height:22px; line-height:1; border-radius:999px; background:rgba(255,255,255,0.02); cursor:help; }
                        .tt_text {
                            position:absolute; left:0; top:110%; width:320px; background:#0a0f1a; color:#f0f6ff;
                            border:1px solid rgba(255,255,255,0.06); padding:10px; font-size:13px; border-radius:8px;
                            box-shadow:0 6px 18px rgba(0,0,0,0.6); visibility:hidden; opacity:0; transition:opacity 0s ease;
                            z-index:999;
                        }
                        .tt_wrapper:hover .tt_text { visibility:visible; opacity:1; }
                        </style>
                        <div class='custom-gradcam-label' style='display:flex; align-items:center;'>
                            <div style='font-size:14px; color:#f0f6ff;'>Show Grad-CAM++ heatmap</div>
                            <span class='tt_wrapper' aria-label='info'>
                                <span class='tt_icon'>?</span>
                                <div class='tt_text'>
                                    Grad-CAM++ Heatmap highlights the regions of the image that most influenced the model’s prediction, helping you understand how the decision was made.
                                </div>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if show_gradcam and st.session_state.cnn_models and st.session_state.get("predict_clicked", False):
                    with st.spinner("Generating Grad-CAM++ heatmap"):
                        try:
                            cam, overlay = None, None
                            # prefer processed_img returned by prediction if available
                            proc_for_gc = processed_img if 'processed_img' in locals() else processed_preview
                            cam, overlay, orig_np = get_gradcam(
                                st.session_state.cnn_models[0][1], proc_for_gc, q_norm
                            )
                            # Layout: show only Grad-CAM heatmap and overlay (separate from previews)
                            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                            fig.patch.set_facecolor('#111827')
                            for ax in axes:
                                ax.set_facecolor('#111827')
                                ax.axis('off')

                            axes[0].imshow(cam, cmap='jet')
                            axes[0].set_title('Grad-CAM++ Heatmap', color='#8899bb', fontsize=11, pad=8)
                            axes[1].imshow(overlay)
                            axes[1].set_title('Overlay', color='#8899bb', fontsize=11, pad=8)

                            fig.suptitle(f"{uploaded.name}  ·  P(GON+) = {ensemble_prob:.3f}  ·  Q={q_score:.2f}",
                                         color='#f0f6ff', fontsize=13, y=0.99, fontfamily='serif')
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        except Exception as e:
                            st.warning(f"Grad-CAM not available: {e}")

                # ── Disclaimer ──
                st.markdown("""
                <div style='background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.2);
                     border-radius:10px; padding:14px 18px; margin-top:16px;
                     font-size:12px; color:#8899bb; font-family:"DM Sans",sans-serif;'>
                    <b style='color:#f59e0b;'>Disclaimer:</b>
                     This system aims to expand access to early glaucoma screening through AI-assisted fundus image analysis, particularly in resource-limited settings. 
                     It is designed to support clinical decision-making, not replace professional diagnosis. 
                     Evaluation by an ophthalmologist remains essential for accurate diagnosis and care
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMA MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>5-Fold CV Performance Summary</div>", unsafe_allow_html=True)

    ASSETS = os.path.join(os.path.dirname(__file__), "assets")

    # Fold results table
    fold_csv = os.path.join(ASSETS, "fold_results.csv")
    if os.path.exists(fold_csv):
        df_res = pd.read_csv(fold_csv)
        df_res.columns = [c.strip() for c in df_res.columns]

        # Rename xgb_auc → svm_auc if needed
        if "xgb_auc" in df_res.columns:
            df_res = df_res.rename(columns={"xgb_auc": "val_auc"})
        if "svm_auc" in df_res.columns:
            df_res = df_res.rename(columns={"svm_auc": "val_auc"})

        mean_auc = df_res["val_auc"].mean()
        std_auc  = df_res["val_auc"].std()
        max_auc  = df_res["val_auc"].max()
        best_fold = int(df_res.loc[df_res["val_auc"].idxmax(), "fold"])

        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, sub in [
            (m1, "Mean AUC", f"{mean_auc:.4f}", "Mean of 5 folds"),
            (m2, "Std AUC",  f"±{std_auc:.4f}", "Standard deviation"),
            (m3, "Best AUC", f"{max_auc:.4f}", f"Fold {best_fold}"),
            (m4, "N Folds",  f"{len(df_res)}", "Total CV folds"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='font-size:28px;'>{val}</div>
                    <div class='metric-sub'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)

        # AUC bar chart per fold
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')

        colors_bar = ['#00d4c8' if v == max_auc else '#3b82f6'
                      for v in df_res["val_auc"]]
        bars = ax.bar(
            [f"Fold {int(r)}" for r in df_res["fold"]],
            df_res["val_auc"],
            color=colors_bar, edgecolor='none', width=0.55
        )
        ax.axhline(mean_auc, color='#f59e0b', linewidth=1.5, linestyle='--',
                   label=f'Mean = {mean_auc:.4f}')
        ax.set_ylim(max(0, df_res["val_auc"].min() - 0.02), 1.02)
        ax.set_ylabel("Validation AUC", color='#8899bb', fontsize=11)
        ax.tick_params(colors='#8899bb')
        ax.spines[:].set_visible(False)
        for b, v in zip(bars, df_res["val_auc"]):
            ax.text(b.get_x() + b.get_width()/2, v + 0.003,
                    f"{v:.4f}", ha='center', va='bottom',
                    color='white', fontsize=11, fontweight='bold',
                    fontfamily='monospace')
        ax.legend(loc='lower right', facecolor='#1a2436',
                  edgecolor='#2a3f6f', labelcolor='#8899bb')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Raw table
        with st.expander("📋 View full table"):
            st.dataframe(df_res, use_container_width=True)

    st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)

    # Training history & Confusion matrices (per-fold tabs)
    st.markdown("<div class='section-title'>Training History</div>", unsafe_allow_html=True)

    history_files = sorted([f for f in os.listdir(ASSETS) if f.startswith("history_fold")])
    cm_files = sorted([f for f in os.listdir(ASSETS) if f.startswith("confusion_matrix_fold")])

    # map fold index -> filename
    def extract_fold(fname, prefix):
        base = fname.replace(prefix, "")
        base = base.split('.')[0]
        try:
            return int(''.join([c for c in base if c.isdigit()]))
        except:
            return None

    hist_map = {extract_fold(f, 'history_fold'): f for f in history_files}
    cm_map = {extract_fold(f, 'confusion_matrix_fold'): f for f in cm_files}

    available_folds = sorted(set(list(hist_map.keys()) + list(cm_map.keys())))
    if available_folds:
        fold_tabs = st.tabs([f"Fold {i}" for i in available_folds])
        for tab, fold in zip(fold_tabs, available_folds):
            with tab:
                col_left, col_right = st.columns(2)
                # show training history (if available)
                if fold in hist_map:
                    with col_left:
                        img = Image.open(os.path.join(ASSETS, hist_map[fold]))
                        st.image(img, caption=f"Training history - Fold {fold}", use_container_width=True)
                else:
                    with col_left:
                        st.info(f"No training history image for Fold {fold}")

                # show confusion matrix for this fold (if available)
                if fold in cm_map:
                    with col_right:
                        img = Image.open(os.path.join(ASSETS, cm_map[fold]))
                        st.image(img, caption=f"Confusion Matrix - Fold {fold}", use_container_width=True)
                else:
                    with col_right:
                        st.info(f"No confusion matrix image for Fold {fold}")

    st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)

    # GradCAM sample
    gc_files = [f for f in os.listdir(ASSETS) if f.startswith("gradcam")]
    if gc_files:
        st.markdown("<div class='custom-sep'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Sample Grad-CAM++ (Training)</div>", unsafe_allow_html=True)
        gcols = st.columns(min(len(gc_files), 3))
        for i, (col, gf) in enumerate(zip(gcols * 5, gc_files)):
            with col:
                img = Image.open(os.path.join(ASSETS, gf))
                st.image(img, caption=gf.replace("gradcam_","").replace(".png",""),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TENTANG
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        st.markdown("""
        <div class='section-title'>About Model</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:24px;'>
        <div style='font-family:"DM Serif Display",serif; font-size:18px;
             color:#f0f6ff; margin-bottom:12px;'>Model Architecture</div>
        """, unsafe_allow_html=True)

        steps = [
            ("1", "EfficientNet-B3 Backbone", "Feature extractor pretrained on ImageNet. Backbone initially frozen, then unfrozen from epoch 5."),
            ("2", "Hybrid Attention Module", "Channel Attention (SE-Net style) + Spatial Attention (CBAM style) → focuses features on the optic disc area."),
            ("3", "Quality Score Fusion", "Image quality score is normalized and fused as an additional feature before the classifier."),
            ("4", "SVM + StandardScaler", "Backbone features are classified by an RBF-kernel SVM — more stable on small datasets than an MLP head."),
            ("5", "5-Fold Ensemble", "Predictions from 5 models are averaged for a more robust and stable result."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; gap:16px; margin-bottom:16px; align-items:flex-start;'>
                <div style='flex-shrink:0; width:32px; height:32px; border-radius:50%;
                     background:linear-gradient(135deg,#00d4c8,#3b82f6);
                     display:flex; align-items:center; justify-content:center;
                     font-weight:700; font-size:14px; color:#0a0f1a;'>{num}</div>
                <div>
                    <div style='font-weight:600; color:#f0f6ff; font-size:14px;'>{title}</div>
                    <div style='color:#8899bb; font-size:13px; margin-top:3px; line-height:1.6;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # How to Use (moved under Model Architecture)
        st.markdown("""
        <div style='margin-top:18px;'>
        <div class='section-title' style='font-size:18px;'>How to Use</div>
        <div style='color:#8899bb; font-size:13px; font-family:"DM Sans",sans-serif; line-height:1.8; margin-top:8px;'>
            1. Open the <b>Prediction &amp; Analysis</b> tab.<br>
            2. Upload a fundus photo (.jpg / .png).<br>
            3. Set the image quality score (1–10).<br>
            4. Click <b>Predict</b> — results and Grad-CAM++ visualizations appear automatically.
        </div>
        </div>

        <div style='margin-top:18px;'>
        <div class='section-title' style='font-size:18px;'>Citation</div>
        <div style='color:#8899bb; font-size:13px; font-family:"DM Sans",sans-serif; line-height:1.6; margin-top:8px;'>
            <b>Data Citation</b> : Abramovich, O., Pizem, H., Fhima, J., Berkowitz, E., Gofrit, B., Van Eijgen, J., Blumenthal, E., & Behar, J. (2025). Hillel Yaffe Glaucoma Dataset (HYGD): A Gold-Standard Annotated Fundus Dataset for Glaucoma Detection (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/z0ak-km33<br><br>
            <b>Platform Citation</b> : Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='section-title' style='font-size:18px;'>Technical Specifications</div>
        """, unsafe_allow_html=True)

        specs = [
            ("Backbone",         "EfficientNet-B3"),
            ("Image Size",       "300 × 300 px"),
            ("Attention",        "Channel + Spatial (Hybrid)"),
            ("Fusion Dim",       "513 (512 + quality)"),
            ("Classifier",       "SVM · kernel=RBF"),
            ("Preprocessing",    "CLAHE + Black Crop + Square Pad"),
            ("Augmentation",       "Flip, Rotate, ColorJitter, Crop"),
            ("Optimizer",        "AdamW · lr=3e-4"),
            ("Scheduler",        "Warmup + CosineAnnealing"),
            ("Early Stopping",   "Patience = 7 epoch"),
            ("Unfreeze Epoch",   "Epoch 5"),
            ("Max Epochs",       "30"),
            ("K-Fold",           "5 folds · GroupKFold"),
            ("Holdout Test",     "20% (patient-level split)"),
        ]

        for k, v in specs:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                 padding:9px 14px; border-bottom:1px solid rgba(59,130,246,0.1);
                 font-size:13px;'>
                <span style='color:#8899bb; font-family:"DM Sans",sans-serif;'>{k}</span>
                 <span style='color:#f0f6ff; font-family:"JetBrains Mono",monospace;
                     font-size:12px; text-align:right; display:block;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

        # 'How to Use' and 'Citation' will be inserted below the Model Architecture section