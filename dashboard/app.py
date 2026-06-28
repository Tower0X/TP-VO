"""
dashboard/app.py — Interface Opérationnelle CSRNet
===================================================
Groupe 8 : Estimation de la Densité de Trafic Urbain

Dashboard de démonstration avec superposition temps-réel
de la carte de densité sur l'image source.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import streamlit as st

# Ajouter la racine du projet au sys.path pour les imports absolus
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import CKPT_DIR, inference_cfg
from inference.predictor import TrafficPredictor
from dashboard.heatmap import render_heatmap_overlay
from dashboard.thresholds import get_congestion_info

# ─────────────────────────────────────────────────────────────
# CONFIGURATION PAGE
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CSRNet — Densité de Trafic",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# PARAMÈTRES FIXES (plus de sliders inutiles)
# ─────────────────────────────────────────────────────────────

HEATMAP_ALPHA = 0.45
HEATMAP_COLORMAP = "turbo"

# ─────────────────────────────────────────────────────────────
# CACHE & CHARGEMENT
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_predictor():
    """Charge le modèle de la camarade (Configuration B) et le met en cache."""
    model_path = ROOT_DIR / "best_csrnet.pth"
        
    if model_path.exists():
        return TrafficPredictor(model_path=model_path)
    else:
        st.error(f"Modèle non trouvé dans {model_path}.")
        return TrafficPredictor(model_path=None)

# ─────────────────────────────────────────────────────────────
# CSS PREMIUM — DARK MODE
# ─────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Titre principal ── */
    .main-title {
        text-align: center;
        padding: 1.2rem 0 0.2rem 0;
        margin-bottom: 0;
    }
    .main-title h1 {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 50%, #FF9D6C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .main-subtitle {
        text-align: center;
        color: #888;
        font-size: 0.92rem;
        font-weight: 300;
        letter-spacing: 0.4px;
        margin-top: 0;
        padding-bottom: 0.6rem;
    }

    /* ── Bandeau KPI ── */
    .kpi-band {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 1rem auto 1.2rem auto;
        max-width: 900px;
    }
    .kpi-card {
        flex: 1;
        text-align: center;
        padding: 1.4rem 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(255, 75, 43, 0.3);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        color: #999;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #aaa;
        margin-top: 0.3rem;
        font-weight: 300;
    }

    /* ── Couleurs sémantiques ── */
    .status-fluide  { color: #00E676; text-shadow: 0 0 20px rgba(0,230,118,0.4); }
    .status-modere  { color: #FFAB00; text-shadow: 0 0 20px rgba(255,171,0,0.4);  }
    .status-congestion { color: #FF1744; text-shadow: 0 0 20px rgba(255,23,68,0.5); }

    .border-fluide  { border-color: rgba(0,230,118,0.25); }
    .border-modere  { border-color: rgba(255,171,0,0.25); }
    .border-congestion { border-color: rgba(255,23,68,0.30); }

    /* ── Image overlay container ── */
    .overlay-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 48px rgba(0,0,0,0.5);
        margin: 0 auto;
        max-width: 960px;
    }

    /* ── Placeholder ── */
    .premium-placeholder {
        padding: 80px 40px;
        text-align: center;
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
        border: 2px dashed rgba(255, 75, 43, 0.35);
        border-radius: 20px;
        color: #777;
        font-weight: 300;
        letter-spacing: 0.5px;
        transition: all 0.4s;
        max-width: 700px;
        margin: 2rem auto;
    }
    .premium-placeholder:hover {
        border-color: #FF4B2B;
        background: rgba(255, 75, 43, 0.04);
        color: #aaa;
    }

    /* ── Sidebar tweaks ── */
    [data-testid="stSidebar"] {
        background: rgba(30, 30, 40, 0.97);
        border-right: 1px solid rgba(255, 75, 43, 0.15);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #e0e0e0;
    }

    /* ── Distribution chart ── */
    .chart-section-title {
        text-align: center;
        font-size: 1.15rem;
        font-weight: 600;
        color: #ddd;
        margin-bottom: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KPI RENDERING
# ─────────────────────────────────────────────────────────────

def render_kpi_banner(count: int, info: dict, alert_pct: int):
    """Affiche le bandeau de 3 KPIs avec couleurs sémantiques."""
    # Déterminer la classe CSS de couleur
    status_lower = info["status"].lower()
    if status_lower == "fluide":
        css_class = "status-fluide"
        border_class = "border-fluide"
    elif "mod" in status_lower:
        css_class = "status-modere"
        border_class = "border-modere"
    else:
        css_class = "status-congestion"
        border_class = "border-congestion"

    st.markdown(f"""
    <div class="kpi-band">
        <div class="kpi-card {border_class}">
            <div class="kpi-label">🚗 Véhicules Comptés</div>
            <div class="kpi-value" style="color: #FFFFFF;">{count}</div>
            <div class="kpi-sub">estimés par CSRNet</div>
        </div>
        <div class="kpi-card {border_class}">
            <div class="kpi-label">🚦 Statut du Trafic</div>
            <div class="kpi-value {css_class}">{info['icon']} {info['status']}</div>
            <div class="kpi-sub">{info['message']}</div>
        </div>
        <div class="kpi-card {border_class}">
            <div class="kpi-label">⚠️ Niveau d'Alerte</div>
            <div class="kpi-value {css_class}">{alert_pct}%</div>
            <div class="kpi-sub">seuil de congestion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# APPLICATION PRINCIPALE
# ─────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Titre ──
    st.markdown('<div class="main-title"><h1>Estimation de Densité de Trafic Urbain</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Projet de Master — Groupe 8 &nbsp;|&nbsp; Architecture CSRNet (VGG-16 + Convolutions Dilatées)</p>', unsafe_allow_html=True)

    # ── Sidebar : Upload + Seuils uniquement ──
    with st.sidebar:
        st.header("📷 Source d'Image")
        uploaded_file = st.file_uploader(
            "Téléchargez une image de trafic",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.divider()

        st.header("🚦 Seuils de Congestion")
        threshold_fluid = st.number_input(
            "Fluide (max véhicules)",
            value=int(inference_cfg.threshold_fluid),
            min_value=1,
            help="En dessous de ce seuil, le trafic est considéré fluide."
        )
        threshold_mod = st.number_input(
            "Modéré (max véhicules)",
            value=int(inference_cfg.threshold_moderate),
            min_value=threshold_fluid + 1,
            help="Au-delà de ce seuil, on passe en congestion."
        )

        # Mise à jour live des seuils
        inference_cfg.threshold_fluid = threshold_fluid
        inference_cfg.threshold_moderate = threshold_mod

        st.divider()

        st.caption("""
        **Mode d'emploi**  
        1. Uploadez une image de caméra urbaine.  
        2. CSRNet génère la carte de densité.  
        3. Le comptage et le niveau de congestion s'affichent instantanément.
        """)

    # ── Chargement du modèle ──
    predictor = load_predictor()

    # ── Contenu principal ──
    if uploaded_file is not None:
        with st.spinner("🔬 Analyse de la scène par CSRNet..."):
            # 1. Lecture de l'image
            pil_image = Image.open(uploaded_file).convert('RGB')
            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 2. Inférence
            density_map, count_raw = predictor.predict(image_bgr)

            # 3. Arrondir à l'entier le plus proche
            count = int(round(count_raw))

            # 4. Superposition heatmap sur l'image source (UN SEUL visuel)
            overlay_bgr = render_heatmap_overlay(
                image_bgr,
                density_map,
                alpha=HEATMAP_ALPHA,
                colormap_name=HEATMAP_COLORMAP
            )
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            # 5. Informations de Congestion
            info = get_congestion_info(count)

        # ── KPI Banner ──
        alert_pct = min(int(count / (threshold_mod * 1.5) * 100), 100)
        render_kpi_banner(count, info, alert_pct)

        st.divider()

        # ── Image unique : overlay superposé ──
        st.image(
            overlay_rgb,
            caption=f"Vision CSRNet — {count} véhicules détectés",
            use_container_width=True
        )

        # ── Analyse de la Distribution Spatiale ──
        st.divider()
        st.markdown('<p class="chart-section-title">📈 Analyse de la Distribution Spatiale</p>', unsafe_allow_html=True)

        # Profil horizontal : somme de la densité le long de chaque colonne (axe X)
        horizontal_profile = density_map.sum(axis=0)
        # Profil vertical : somme le long de chaque ligne (axe Y)
        vertical_profile = density_map.sum(axis=1)

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig1, ax1 = plt.subplots(figsize=(7, 2.8), facecolor='#0E1117')
            ax1.set_facecolor('#0E1117')
            x_pos = np.arange(len(horizontal_profile))
            ax1.fill_between(x_pos, horizontal_profile, alpha=0.35, color='#FF4B2B')
            ax1.plot(x_pos, horizontal_profile, color='#FF6F61', linewidth=1.5)
            ax1.set_xlabel('Position horizontale (pixels)', color='#aaa', fontsize=9)
            ax1.set_ylabel('Densité cumulée', color='#aaa', fontsize=9)
            ax1.set_title('Profil Horizontal — Zones de concentration', color='#ddd', fontsize=10, fontweight='bold', pad=10)
            ax1.tick_params(colors='#888', labelsize=8)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#444')
            ax1.spines['bottom'].set_color('#444')
            ax1.set_xlim(0, len(horizontal_profile) - 1)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with col_chart2:
            fig2, ax2 = plt.subplots(figsize=(7, 2.8), facecolor='#0E1117')
            ax2.set_facecolor('#0E1117')
            y_pos = np.arange(len(vertical_profile))
            ax2.fill_between(y_pos, vertical_profile, alpha=0.35, color='#FFAB00')
            ax2.plot(y_pos, vertical_profile, color='#FFD54F', linewidth=1.5)
            ax2.set_xlabel('Position verticale (pixels)', color='#aaa', fontsize=9)
            ax2.set_ylabel('Densité cumulée', color='#aaa', fontsize=9)
            ax2.set_title('Profil Vertical — Effet de perspective', color='#ddd', fontsize=10, fontweight='bold', pad=10)
            ax2.tick_params(colors='#888', labelsize=8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#444')
            ax2.spines['bottom'].set_color('#444')
            ax2.set_xlim(0, len(vertical_profile) - 1)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    else:
        # ── État vide : placeholder élégant ──
        st.markdown("""
        <div class="premium-placeholder">
            <h3 style="margin-bottom:8px; color:#ddd; font-weight:600;">
                Prêt pour l'Analyse
            </h3>
            <p style="margin:0; font-size: 0.95rem;">
                Uploadez une image de caméra urbaine dans la barre latérale pour lancer la détection.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
