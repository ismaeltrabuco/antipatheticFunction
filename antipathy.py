import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="A-PINN · Antipathy Function", layout="wide", page_icon="◈")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,700;1,300&display=swap');
html,body,[class*="css"]{ font-family:'DM Mono',monospace; background:#07000f; color:#d0b8c8; }
.stApp{ background:#07000f; }
h1,h2,h3{ font-family:'Fraunces',serif; font-weight:300; color:#f0d8e8; }
.pipe-header{
  font-size:10px; letter-spacing:0.15em; color:#6a3a5a; text-transform:uppercase;
  border-bottom:0.5px solid #2a0a1a; padding-bottom:5px; margin:20px 0 10px;
}
.formula-box{
  background:#0d0415; border:0.5px solid #4a1a3a; border-radius:8px;
  padding:16px 20px; font-size:12px; color:#9a6a8a; margin:10px 0; line-height:2.2;
}
.formula-term{ display:inline-block; text-align:center; margin:0 10px; }
.formula-term .name{ font-size:9px; letter-spacing:0.1em; display:block; margin-bottom:2px; }
.insight-card{
  background:#0d0415; border-left:2px solid #c04080; border-radius:0 6px 6px 0;
  padding:10px 14px; margin:5px 0; font-size:11px; line-height:1.7;
}
.insight-card.attract{ border-left-color:#40a080; }
.insight-card.collapse{ border-left-color:#804000; }
.metric-card{
  background:#110820; border:0.5px solid #3a1a3a; border-radius:10px;
  padding:16px 18px 12px; margin-bottom:10px;
}
.metric-label{ font-size:10px; letter-spacing:0.1em; color:#6a3a5a; margin-bottom:4px; }
.metric-value{ font-size:24px; font-weight:500; color:#ff80a0; }
</style>
""", unsafe_allow_html=True)

DEFAULT_CSV = """id,age,device,timestamp,geoLock,bought,cart,read5min,interaction,session_value,traffic_source
1,23,mobile,2026-03-01 10:15,BR,Y,Y,Y,Y,120,organic
2,35,desktop,2026-03-01 11:20,BR,N,Y,Y,Y,80,ads
3,29,mobile,2026-03-01 12:05,US,N,N,N,Y,0,social
4,41,tablet,2026-03-01 13:40,BR,Y,Y,Y,Y,200,email
5,22,mobile,2026-03-01 14:10,IN,N,N,Y,Y,0,organic
6,31,desktop,2026-03-01 15:30,BR,Y,Y,Y,Y,150,ads
7,27,mobile,2026-03-01 16:45,US,N,Y,Y,Y,60,social
8,38,desktop,2026-03-01 17:10,BR,N,N,N,N,0,organic
9,45,tablet,2026-03-01 18:20,BR,Y,Y,Y,Y,220,email
10,19,mobile,2026-03-01 19:05,IN,N,N,N,Y,0,social
11,33,desktop,2026-03-01 20:30,BR,Y,Y,Y,Y,170,ads
12,28,mobile,2026-03-01 21:00,US,N,Y,N,Y,40,organic
13,50,desktop,2026-03-01 21:45,BR,Y,Y,Y,Y,300,email
14,26,mobile,2026-03-01 22:10,BR,N,N,Y,Y,0,ads
15,37,tablet,2026-03-02 09:15,US,N,Y,Y,Y,90,social
16,24,mobile,2026-03-02 10:00,IN,N,N,N,N,0,organic
17,42,desktop,2026-03-02 11:20,BR,Y,Y,Y,Y,210,email
18,30,mobile,2026-03-02 12:40,US,N,Y,Y,Y,70,ads
19,21,mobile,2026-03-02 13:10,BR,N,N,Y,Y,0,organic
20,39,desktop,2026-03-02 14:55,BR,Y,Y,Y,Y,190,email
21,34,tablet,2026-03-02 15:30,US,N,Y,Y,Y,85,social
22,27,mobile,2026-03-02 16:05,IN,N,N,N,Y,0,ads
23,46,desktop,2026-03-02 17:25,BR,Y,Y,Y,Y,260,email
24,25,mobile,2026-03-02 18:10,US,N,Y,N,Y,50,organic
25,32,desktop,2026-03-02 19:45,BR,Y,Y,Y,Y,140,ads
26,28,mobile,2026-03-02 20:20,IN,N,N,Y,Y,0,social
27,36,tablet,2026-03-02 21:00,BR,N,Y,Y,Y,75,organic
28,48,desktop,2026-03-02 21:40,BR,Y,Y,Y,Y,310,email
29,23,mobile,2026-03-02 22:15,US,N,N,N,Y,0,social
30,40,desktop,2026-03-03 09:10,BR,Y,Y,Y,Y,180,ads
31,29,mobile,2026-03-03 10:35,IN,N,Y,Y,Y,65,organic
32,52,desktop,2026-03-03 11:50,BR,Y,Y,Y,Y,330,email
33,20,mobile,2026-03-03 12:25,US,N,N,N,N,0,social"""

# ══════════════════════════════════════════════════════════════════════════════
# PIPE 1 — NORMALIZAÇÃO UNIVERSAL
# ══════════════════════════════════════════════════════════════════════════════
def pipe1_normalize(df):
    df = df.copy()
    drop_cols = [c for c in df.columns
                 if c.lower() in ('id','timestamp','index')]
    df = df.drop(columns=drop_cols, errors='ignore')
    numeric_feats, binary_feats, cat_feats = [], [], []
    for col in list(df.columns):
        vals = df[col].dropna().astype(str).str.upper()
        if vals.isin({'Y','N','YES','NO','TRUE','FALSE','1','0'}).all():
            df[col] = df[col].apply(
                lambda x: 1.0 if str(x).upper() in ('Y','YES','TRUE','1') else 0.0)
            binary_feats.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            rng = df[col].max() - df[col].min()
            df[col] = (df[col]-df[col].min())/rng if rng>0 else 0.0
            numeric_feats.append(col)
        else:
            dummies = pd.get_dummies(df[col], prefix=col).astype(float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            cat_feats.append(col)
    meta = dict(numeric=numeric_feats, binary=binary_feats,
                categorical=cat_feats, all_feats=list(df.columns))
    return df, meta

# ══════════════════════════════════════════════════════════════════════════════
# PIPE 2 — BIG BANG
# ══════════════════════════════════════════════════════════════════════════════
def pipe2_bigbang(df_norm, projection='pca'):
    feats = df_norm.columns.tolist()
    X = df_norm.values.T          # (n_feats, n_samples)
    # handle zero-variance rows
    stds = X.std(axis=1)
    X = np.where(stds[:,None]>0, X, X+np.random.randn(*X.shape)*1e-6)

    if projection == 'pca':
        Xc = X - X.mean(axis=1, keepdims=True)
        cov_mat = Xc @ Xc.T / max(X.shape[1]-1, 1)
        vals, vecs = np.linalg.eigh(cov_mat)
        idx = np.argsort(vals)[::-1]
        # eigenvectors directly give feat positions in PC space
        coords = vecs[:, idx[:2]]
    else:
        corr = np.corrcoef(X)
        np.fill_diagonal(corr, 1.0)
        dist = np.sqrt(np.clip(1 - corr, 0, 2))
        n = dist.shape[0]
        D2 = dist**2
        H = np.eye(n) - np.ones((n,n))/n
        B = -0.5 * H @ D2 @ H
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        coords = vecs[:, idx[:2]] * np.sqrt(np.maximum(vals[idx[:2]], 0))

    # normalise coords to [-1,1]
    for ax in range(2):
        rng = (coords[:,ax].max() - coords[:,ax].min())
        if rng > 0:
            coords[:,ax] = (coords[:,ax]-coords[:,ax].min())/rng*2-1

    np.random.seed(7)
    vel = np.random.randn(len(feats), 2) * 0.015   # big bang velocity
    corr_matrix = np.corrcoef(X)
    np.fill_diagonal(corr_matrix, 0.0)
    return feats, coords, vel, corr_matrix

# ══════════════════════════════════════════════════════════════════════════════
# PIPE 3 — SIMULAÇÃO COMPLETA A-PINN
#
# Estado interno:
#   y_{t+1} = y_t − λ · |cov(y_t, o_t)|
#
# Movimento (Physics-Informed):
#   ẋ = [proteção]  y_t · F_rep
#     + [coesão]    α · (v̄_nbr − v_i)
#     + [espaço]    β · sep
#     + [vida]      η(t)          ← Ornstein-Uhlenbeck
# ══════════════════════════════════════════════════════════════════════════════
def pipe3_apinn(feats, coords_init, vel_init, corr_matrix,
                lam=0.18, alpha=0.055, beta=0.025,
                theta_ou=0.05, sigma_ou=0.009,
                anti_mode='both', anti_threshold=0.1,
                n_steps=50, var_sizes=None):

    n = len(feats)
    coords = coords_init.copy()
    vel    = vel_init.copy()
    y_int  = np.ones(n) * 0.72

    # Ornstein-Uhlenbeck global (vida)
    ou = np.zeros(2)

    # classify pairs
    antipathy_pairs, sympathy_pairs = [], []
    for i in range(n):
        for j in range(i+1, n):
            c  = corr_matrix[i,j]
            fi, fj = feats[i], feats[j]
            is_opp = any([
                fi.endswith('_Y') and fj == fi[:-2]+'_N',
                fi.endswith('_N') and fj == fi[:-2]+'_Y',
                fj.endswith('_Y') and fi == fj[:-2]+'_N',
                fj.endswith('_N') and fi == fj[:-2]+'_Y',
            ])
            use_anti = (anti_mode=='opposites' and is_opp) or \
                       (anti_mode=='auto' and c < -anti_threshold) or \
                       (anti_mode=='both' and (is_opp or c < -anti_threshold))
            use_sym  = (not use_anti) and c > anti_threshold

            if use_anti:
                strength = max(abs(c), 0.65) if is_opp else abs(c)
                antipathy_pairs.append((i, j, strength))
            elif use_sym:
                sympathy_pairs.append((i, j, c))

    traj   = [coords.copy()]
    y_hist = [y_int.copy()]
    WIN    = 8

    for step in range(n_steps):

        # ── Vida: η(t) Ornstein-Uhlenbeck ────────────────────────────
        ou = ou*(1-theta_ou) + sigma_ou*np.random.randn(2)

        forces = np.zeros((n, 2))

        for i in range(n):
            # observable: mean position as external signal
            o_t = np.mean(coords[i])
            past_y = np.array([h[i] for h in y_hist[-WIN:]])
            past_o = np.full(len(past_y), o_t)  # simplified scalar obs
            if len(past_y) >= 2:
                ym = past_y.mean(); om = past_o.mean()
                cov_val = np.mean((past_y-ym)*(past_o-om))
            else:
                cov_val = 0.0

            # ── ANTIEMPATIA: y_{t+1} = y_t − λ·|cov(y_t, o_t)| ─────
            y_int[i] = max(0.0, y_int[i] - lam * abs(cov_val))

        # ── Proteção: y_t · F_rep ────────────────────────────────────
        for (i, j, strength) in antipathy_pairs:
            dx = coords[j] - coords[i]
            d  = np.linalg.norm(dx) + 1e-6
            # força proporcional ao estado interno (proteção)
            f_rep = strength * y_int[i] * (1.0/(d*d+0.1)) * 0.08
            forces[i] -= (dx/d) * f_rep
            forces[j] += (dx/d) * f_rep * (y_int[j]/(y_int[i]+1e-6))

        # ── Coesão: α·(v̄_nbr − v_i) — alinhamento de velocidade ────
        for i in range(n):
            nbr_vx, nbr_vy, cnt = 0.0, 0.0, 0
            for (ii, jj, s) in sympathy_pairs:
                partner = jj if ii==i else (ii if jj==i else -1)
                if partner >= 0:
                    nbr_vx += vel[partner,0]
                    nbr_vy += vel[partner,1]
                    cnt += 1
            if cnt > 0:
                forces[i,0] += alpha * (nbr_vx/cnt - vel[i,0])
                forces[i,1] += alpha * (nbr_vy/cnt - vel[i,1])

        # ── Espaço: β·sep — separação dura ──────────────────────────
        R_SEP = 0.30
        for i in range(n):
            for j in range(i+1, n):
                dx = coords[j] - coords[i]
                d  = np.linalg.norm(dx) + 1e-6
                if d < R_SEP:
                    sep_f = beta * (R_SEP - d) / d
                    forces[i] -= dx * sep_f
                    forces[j] += dx * sep_f

        # ── Vida: η(t) — mundo move quando tudo se esgota ───────────
        for i in range(n):
            forces[i] += ou * (0.4 + 0.6*(1-y_int[i]))  # mais forte quando y→0

        # simpatia: atração suave
        for (i, j, strength) in sympathy_pairs:
            dx = coords[j] - coords[i]
            d  = np.linalg.norm(dx) + 1e-6
            forces[i] += (dx/d) * strength * 0.025
            forces[j] -= (dx/d) * strength * 0.025

        vel = vel*0.88 + forces
        # speed cap
        spds = np.linalg.norm(vel, axis=1, keepdims=True)
        vel  = np.where(spds>0.15, vel/spds*0.15, vel)

        coords = coords + vel
        # soft boundary [-1.5, 1.5]
        coords = np.clip(coords, -1.5, 1.5)

        traj.append(coords.copy())
        y_hist.append(y_int.copy())

    return traj, y_hist, antipathy_pairs, sympathy_pairs, y_int.copy()

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO MATPLOTLIB
# ══════════════════════════════════════════════════════════════════════════════
def make_figure(feats, traj, y_hist, antipathy_pairs, sympathy_pairs,
                show_mode='both', var_sizes=None):
    BG='#07000f'; ANTI='#ff4060'; SYM='#40d0a0'; TXT='#c0a8b8'
    if var_sizes is None:
        var_sizes = np.ones(len(feats))

    panels_spec = {
        'both':   [('antes do A-PINN',   traj[0],  y_hist[0]),
                   ('após A-PINN',        traj[-1], y_hist[-1])],
        'before': [('antes do A-PINN',   traj[0],  y_hist[0])],
        'after':  [('após A-PINN',        traj[-1], y_hist[-1])],
    }[show_mode]

    ncols = len(panels_spec)
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6), facecolor=BG)
    if ncols == 1:
        axes = [axes]

    for ax, (title, coords, y_cur) in zip(axes, panels_spec):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a1a3a')
        ax.tick_params(colors='#3a2a4a', labelsize=7)

        # linhas antipáticas
        for (i, j, s) in antipathy_pairs:
            x0,y0 = coords[i]; x1,y1 = coords[j]
            ax.plot([x0,x1],[y0,y1], color=ANTI,
                    alpha=min(s*0.65,0.55), lw=0.9, ls='--', zorder=1)
            mid = np.array([(x0+x1)/2,(y0+y1)/2])
            dv  = np.array([x0-x1, y0-y1])
            dn  = np.linalg.norm(dv)+1e-6
            ax.annotate('', xy=mid+dv/dn*0.09, xytext=mid,
                        arrowprops=dict(arrowstyle='->',color=ANTI,lw=0.7,alpha=0.5))

        # linhas simpáticas
        for (i, j, s) in sympathy_pairs:
            x0,y0=coords[i]; x1,y1=coords[j]
            ax.plot([x0,x1],[y0,y1], color=SYM,
                    alpha=min(s*0.35,0.3), lw=0.5, ls=':', zorder=1)

        # nós — cor encoda y_int
        sizes = 70 + var_sizes*280
        node_colors = []
        for k in range(len(feats)):
            s = y_cur[k]
            r = int(min(255, 180+s*75))
            g = int(min(255, 40+s*60))
            b = int(min(255, 90+s*100))
            node_colors.append(f'#{r:02x}{g:02x}{b:02x}')

        ax.scatter(coords[:,0], coords[:,1], s=sizes, c=node_colors,
                   edgecolors='#2a1030', linewidths=0.5, zorder=3, alpha=0.93)

        for k, feat in enumerate(feats):
            lbl = feat[:13]+('…' if len(feat)>13 else '')
            ax.text(coords[k,0], coords[k,1]+0.07, lbl,
                    fontsize=6.5, color=TXT, ha='center', va='bottom',
                    fontfamily='monospace', zorder=4)
            ax.text(coords[k,0], coords[k,1]-0.08,
                    f'y={y_cur[k]:.2f}',
                    fontsize=5.5, color='#7a507a', ha='center', va='top', zorder=4)

        ax.set_title(title, color='#9a6080', fontsize=9,
                     fontfamily='monospace', pad=8)
        ax.set_xlabel('componente 1', color='#4a3050', fontsize=7)
        ax.set_ylabel('componente 2', color='#4a3050', fontsize=7)
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)

    patches = [
        mpatches.Patch(color=ANTI, label='antipatia  y_t·F_rep  (proteção)'),
        mpatches.Patch(color=SYM,  label='simpatia  α(v̄−v) + β·sep  (coesão+espaço)'),
        mpatches.Patch(color='#b060d0', label='nó: tamanho=variância  |  cor=y_t'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               facecolor='#0d0420', edgecolor='#2a1030',
               labelcolor=TXT, fontsize=7.5, framealpha=0.85)
    fig.tight_layout(rect=[0,0.07,1,1])
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# INSIGHTS HEURÍSTICOS
# ══════════════════════════════════════════════════════════════════════════════
def make_insights(feats, antipathy_pairs, sympathy_pairs, y_final):
    out = []
    for i,j,s in sorted(antipathy_pairs, key=lambda x:-x[2])[:8]:
        out.append(('anti',
            f"SEPARE  {feats[i]}  ↔  {feats[j]}",
            f"força antiempática = {s:.3f}  |  y_final: {y_final[i]:.2f} / {y_final[j]:.2f}"))
    for i,j,s in sorted(sympathy_pairs, key=lambda x:-x[2])[:5]:
        out.append(('sym',
            f"APROXIME  {feats[i]}  ↔  {feats[j]}",
            f"correlação = +{s:.3f}"))
    for k,feat in enumerate(feats):
        if y_final[k] < 0.12:
            out.append(('collapse',
                f"COLAPSOU  {feat}",
                f"y_t = {y_final[k]:.3f}  — coerência esgotada pela antiempatia"))
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◈ A-PINN")
    st.markdown("*Antipathy Physics-Informed Network*")

    st.markdown('<div class="pipe-header">dados</div>', unsafe_allow_html=True)
    uploaded    = st.file_uploader("upload CSV (qualquer)", type=['csv'])
    use_default = st.checkbox("usar CSV de exemplo (33 linhas)", value=True)

    st.markdown('<div class="pipe-header">pipe 2 — projeção inicial</div>',
                unsafe_allow_html=True)
    projection = st.radio("big bang", ['pca','correlation'],
        format_func=lambda x: 'PCA 2D' if x=='pca' else 'Correlação como distância')

    st.markdown('<div class="pipe-header">pipe 3 — A-PINN</div>',
                unsafe_allow_html=True)
    lam   = st.slider("λ — antiempatia",   0.01, 0.50, 0.18, 0.01)
    alpha = st.slider("α — coesão",        0.01, 0.20, 0.055, 0.005)
    beta  = st.slider("β — espaço",        0.01, 0.15, 0.025, 0.005)
    sigma = st.slider("σ — vida (OU)",     0.001, 0.03, 0.009, 0.001)
    n_steps = st.slider("passos",          10, 100, 50, 5)

    st.markdown('<div class="pipe-header">pares antipáticos</div>',
                unsafe_allow_html=True)
    anti_mode = st.radio("detectar por", ['both','auto','opposites'],
        format_func=lambda x: {
            'both':      'opostos + corr negativa',
            'auto':      'só corr negativa',
            'opposites': 'só pares Y/N opostos'
        }[x])
    anti_thr = st.slider("threshold |corr|", 0.0, 0.8, 0.10, 0.05)

    st.markdown('<div class="pipe-header">visualização</div>',
                unsafe_allow_html=True)
    show_mode = st.radio("painel", ['both','before','after'],
        format_func=lambda x: {
            'both':   'before + after',
            'before': 'só antes',
            'after':  'só depois'
        }[x])

    st.markdown("---")
    st.markdown("""
<div style='font-size:10px;color:#4a3a5a;line-height:2.0;'>
<span style='color:#7a4060;'>antiempatia</span><br>
y<sub>t+1</sub> = y<sub>t</sub> − λ·|cov(y<sub>t</sub>,o<sub>t</sub>)|<br><br>
<span style='color:#7a4060;'>movimento A-PINN</span><br>
ẋ = <span style='color:#ff6070;'>y<sub>t</sub>·F<sub>rep</sub></span>
  + <span style='color:#a070ff;'>α(v̄−v)</span>
  + <span style='color:#40c090;'>β·sep</span>
  + <span style='color:#c0b0ff;'>η(t)</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# A-PINN")
st.markdown("*Antipathy Physics-Informed Neural Network*")

st.markdown("""
<div class='formula-box'>
<b style='color:#ff80a0;'>estado interno &nbsp;&nbsp;
y<sub>t+1</sub> = y<sub>t</sub> − λ · |cov(y<sub>t</sub>, o<sub>t</sub>)|</b><br>
<b style='color:#d0a0c0;'>movimento &nbsp;&nbsp;
ẋ &nbsp;=&nbsp;</b>
<span class='formula-term'>
  <span class='name' style='color:#ff6070;'>PROTEÇÃO</span>
  <b style='color:#ff6070;'>y<sub>t</sub> · F<sub>rep</sub></b>
</span>
+
<span class='formula-term'>
  <span class='name' style='color:#a070ff;'>COESÃO</span>
  <b style='color:#a070ff;'>α(v̄ − v)</b>
</span>
+
<span class='formula-term'>
  <span class='name' style='color:#40c090;'>ESPAÇO</span>
  <b style='color:#40c090;'>β · sep</b>
</span>
+
<span class='formula-term'>
  <span class='name' style='color:#c0b0ff;'>VIDA</span>
  <b style='color:#c0b0ff;'>η(t)</b>
</span>
</div>
""", unsafe_allow_html=True)

# carrega
if uploaded:
    df_raw = pd.read_csv(uploaded)
elif use_default:
    df_raw = pd.read_csv(io.StringIO(DEFAULT_CSV))
else:
    st.info("faça upload de um CSV ou ative o exemplo.")
    st.stop()

# PIPE 1
st.markdown('<div class="pipe-header">pipe 1 — normalização universal</div>',
            unsafe_allow_html=True)
df_norm, meta = pipe1_normalize(df_raw)

c1,c2,c3,c4 = st.columns(4)
for col_ui, label, val, clr in [
    (c1, 'LINHAS',          len(df_raw),              '#c080ff'),
    (c2, 'FEATS NUMÉRICAS', len(meta['numeric']),     '#c080ff'),
    (c3, 'FEATS BINÁRIAS',  len(meta['binary']),      '#ff80a0'),
    (c4, 'FEATS TOTAL',     len(meta['all_feats']),   '#40c0a0'),
]:
    col_ui.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>{label}</div>
    <div class='metric-value' style='font-size:22px;color:{clr};'>{val}</div>
    </div>""", unsafe_allow_html=True)

with st.expander("ver dataframe normalizado"):
    st.dataframe(df_norm.round(3), width='stretch', height=200)

# PIPE 2
st.markdown('<div class="pipe-header">pipe 2 — big bang: projeção das features</div>',
            unsafe_allow_html=True)
feats, coords_init, vel_init, corr_matrix = pipe2_bigbang(df_norm, projection)
var_sizes = df_norm.var().values
if (var_sizes.max() - var_sizes.min()) > 0:
    var_sizes = (var_sizes - var_sizes.min()) / (var_sizes.max() - var_sizes.min())

st.caption(f"{len(feats)} features projetadas  ·  big bang σ=0.015  ·  projeção: {projection.upper()}")

# PIPE 3
st.markdown('<div class="pipe-header">pipe 3 — simulação A-PINN</div>',
            unsafe_allow_html=True)
with st.spinner("simulando A-PINN..."):
    traj, y_hist, anti_pairs, sym_pairs, y_final = pipe3_apinn(
        feats, coords_init, vel_init, corr_matrix,
        lam=lam, alpha=alpha, beta=beta,
        sigma_ou=sigma, n_steps=n_steps,
        anti_mode=anti_mode, anti_threshold=anti_thr,
        var_sizes=var_sizes
    )

m1,m2,m3,m4 = st.columns(4)
collapsed = sum(1 for y in y_final if y<0.12)
for col_ui, label, val, clr in [
    (m1, 'PARES ANTIPÁTICOS',       len(anti_pairs),        '#ff80a0'),
    (m2, 'PARES SIMPÁTICOS',        len(sym_pairs),         '#40c0a0'),
    (m3, 'FEATURES COLAPSADAS',     collapsed,              '#ff5040'),
    (m4, 'y_t MÉDIO FINAL',         f"{np.mean(y_final):.3f}", '#c080ff'),
]:
    col_ui.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>{label}</div>
    <div class='metric-value' style='font-size:20px;color:{clr};'>{val}</div>
    </div>""", unsafe_allow_html=True)

# PIPE 4 — VISUALIZAÇÃO
st.markdown('<div class="pipe-header">pipe 4 — visualização antiempática (matplotlib)</div>',
            unsafe_allow_html=True)
fig = make_figure(feats, traj, y_hist, anti_pairs, sym_pairs,
                  show_mode=show_mode, var_sizes=var_sizes)
st.pyplot(fig, width='stretch')

# PIPE 5 — EVOLUÇÃO y_t
st.markdown('<div class="pipe-header">pipe 5 — drenagem de y_t por feature</div>',
            unsafe_allow_html=True)
evo = pd.DataFrame(np.array(y_hist), columns=feats)
st.line_chart(evo, height=210, width='stretch')

# PIPE 6 — INSIGHTS
st.markdown('<div class="pipe-header">pipe 6 — insights heurísticos</div>',
            unsafe_allow_html=True)
for kind, msg, detail in make_insights(feats, anti_pairs, sym_pairs, y_final):
    cls = {'anti':'insight-card','sym':'insight-card attract',
           'collapse':'insight-card collapse'}[kind]
    icon = {'anti':'↔ ','sym':'↑ ','collapse':'⊘ '}[kind]
    st.markdown(f"""<div class='{cls}'>
    <b>{icon}{msg}</b><br>
    <span style='color:#5a3a5a;font-size:10px;'>{detail}</span>
    </div>""", unsafe_allow_html=True)

# PIPE 7 — TABELA FINAL
st.markdown('<div class="pipe-header">pipe 7 — estado final das features</div>',
            unsafe_allow_html=True)
final_df = pd.DataFrame({
    'feature':   feats,
    'y_inicial': np.round(y_hist[0], 3),
    'y_final':   np.round(y_final,   3),
    'Δy':        np.round(y_final - y_hist[0], 3),
    'variância': np.round(var_sizes, 3),
    'status':    ['colapsou' if y<0.12 else ('estável' if y>0.5 else 'drenando')
                  for y in y_final]
})
st.dataframe(final_df, width='stretch', height=320)
