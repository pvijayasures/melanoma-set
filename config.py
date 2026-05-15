import os
import torch

class CFG:
    # ── Reproduzierbarkeit ─────────────────────────────────────────
    seed         = 42

    # ── Pfade ──────────────────────────────────────────────────────
    base_dir     = os.path.expanduser('~')
    data_dir     = os.path.expanduser('~/data')
    img_dir      = os.path.join(data_dir, 'images/train')
    test_img_dir = os.path.join(data_dir, 'images/test')
    train_csv    = os.path.join(data_dir, 'train.csv')
    test_csv     = os.path.join(data_dir, 'test.csv')
    output_dir   = os.path.join(base_dir, 'outputs')

    # ── Modell ─────────────────────────────────────────────────────
    backbone     = 'efficientnet_b5'   # für finales Run: 'efficientnet_b5'
    pretrained   = True
    dropout      = 0.5                 # ← 0.4 → 0.5 (gegen Overfitting)

    # ── Bild ───────────────────────────────────────────────────────
    img_size     = 384                 # B4: 384, B5: 456
    img_ext      = '.png'              # euer Datensatz nutzt PNG

    # ── Training ───────────────────────────────────────────────────
    n_folds      = 5
    epochs       = 15
    batch_size   = 32                  # B5: auf 16 reduzieren
    num_workers  = 8

    # ── Optimierer ─────────────────────────────────────────────────
    lr           = 1e-4
    weight_decay = 1e-4                # ← 1e-5 → 1e-4 (stärkere L2-Reg.)
    # ABLATION-ERGEBNIS: Uniform LR war bester Ansatz (AUC 0.8793)
    # Discriminative LR war zu konservativ für den Backbone.
    uniform_lr   = True                # ← NEU: Flag für Optimizer

    # ── Scheduler ──────────────────────────────────────────────────
    scheduler    = 'cosine'            # 'cosine' oder 'step'
    t_max        = 15                  # = epochs für CosineAnnealingLR
    warmup_epochs = 1

    # ── Loss ───────────────────────────────────────────────────────
    # ABLATION-ERGEBNIS: Focal Loss klar besser als BCE + pos_weight
    loss         = 'focal'
    focal_alpha  = 0.25
    focal_gamma  = 2.0
    label_smoothing = 0.05

    # ── Metadaten ──────────────────────────────────────────────────
    # ABLATION-ERGEBNIS: Metadaten verbessern PR-AUC (0.1183 vs 0.1001)
    meta_cols    = None                # wird automatisch in train.py gesetzt

    # ── Augmentierung ──────────────────────────────────────────────
    aug_p_flip   = 0.5
    aug_p_color  = 0.7                 # ← 0.6 → 0.7 (etwas stärker)
    aug_p_cutout = 0.4                 # ← 0.3 → 0.4 (gegen Overfitting)

    # ── Inferenz / TTA ─────────────────────────────────────────────
    tta_steps    = 11                  # Anzahl TTA-Durchläufe bei Inferenz
    threshold    = 0.3                 # Für Metriken, nicht für AUC

    # ── Hardware ───────────────────────────────────────────────────
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = torch.cuda.is_available()

    # ── Logging ────────────────────────────────────────────────────
    log_interval = 50                  # Alle N Batches ausgeben
    save_best_only = True

# Beim Import sofort outputs-Ordner anlegen
os.makedirs(CFG.output_dir, exist_ok=True)