# Diagrams for FER Thesis
**Mermaid diagrams — render in any Markdown viewer with Mermaid support**

---

## 1. CNN Architecture Diagram
```mermaid
graph TD
    A[Input: 48×48×1] --> B[Conv2D: 64, 3×3, valid, ReLU]
    B --> C[Conv2D: 64, 3×3, same, ReLU]
    C --> D[BatchNorm]
    D --> E[MaxPool: 2×2]
    E --> F[Dropout: 0.5]
    
    F --> G[Conv2D: 128, 3×3, same, ReLU]
    G --> H[BatchNorm]
    H --> I[Conv2D: 128, 3×3, same, ReLU]
    I --> J[BatchNorm]
    J --> K[MaxPool: 2×2]
    K --> L[Dropout: 0.5]
    
    L --> M[Conv2D: 256, 3×3, same, ReLU]
    M --> N[BatchNorm]
    N --> O[Conv2D: 256, 3×3, same, ReLU]
    O --> P[BatchNorm]
    P --> Q[MaxPool: 2×2]
    Q --> R[Dropout: 0.5]
    
    R --> S[Flatten: 5×5×256 = 6,400]
    S --> T[Dense: 256, ReLU]
    T --> U[Dropout: 0.4]
    U --> V[Dense: 128, ReLU]
    V --> W[Dropout: 0.5]
    W --> X[Dense: 7, Softmax]
    
    style A fill:#e1f5fe
    style X fill:#c8e6c9
    style S fill:#fff3e0
    style T fill:#fff3e0
    style V fill:#fff3e0
```

**Layer Summary Table:**

| Block | Layers | Output Shape | Parameters |
|-------|--------|--------------|------------|
| **Input** | — | 48×48×1 | 0 |
| **Block 1** | Conv(64,3×3,valid) → Conv(64,3×3,same) → BN → MaxPool → Dropout(0.5) | 23×23×64 | 37,696 |
| **Block 2** | Conv(128,3×3,same) → BN → Conv(128,3×3,same) → BN → MaxPool → Dropout(0.5) | 11×11×128 | 221,952 |
| **Block 3** | Conv(256,3×3,same) → BN → Conv(256,3×3,same) → BN → MaxPool → Dropout(0.5) | 5×5×256 | 886,336 |
| **Classifier** | Flatten → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.5) → Dense(7) | 7 | 1,672,391 |
| **Total** | 23 layers | — | **2,818,375** |

---

## 2. End-to-End Pipeline Flowchart
```mermaid
flowchart TD
    subgraph DATA["Data Preparation"]
        D1[fer2013.csv\n35,887 rows]
        D2[Parse pixel strings\n2,304 → 48×48×1]
        D3[One-hot encode labels\n7 classes]
        D4[Split by Usage column]
    end
    
    subgraph SPLITS["Leakage-Free Splits"]
        S1[Training: 28,709\nUsage=Training]
        S2[Validation: 2,871\n10% stratified from Training]
        S3[Test: 7,178\nPublicTest + PrivateTest\nOPENED ONCE ONLY]
    end
    
    subgraph TRAIN["Training Pipeline"]
        T1[Data Generator\nAugmentation: rot±20°, flip, zoom±20%, shear]
        T2[Validation Generator\nRescale only]
        T3[CNN Model\n3 blocks + classifier head]
        T4[Adam, CE loss, batch=64]
        T5[Callbacks:\n- ReduceLROnPlateau (factor=0.8, patience=3)\n- EarlyStopping (patience=6)\n- ModelCheckpoint (best val_loss)]
    end
    
    subgraph EVAL["Evaluation Protocol"]
        E1[Load best checkpoint]
        E2[Predict on test set\nonce, no retraining]
        E3[Metrics: Accuracy, Macro-F1,\nPer-class P/R/F1, Confusion Matrix]
        E4[Bootstrap 95% CI\nMcNemar paired tests]
        E5[Compare to baselines:\nRandom 14.3%, Majority 24.7%]
    end
    
    subgraph DEMO["Live Demo"]
        M1[Webcam capture]
        M2[Haar cascade detection]
        M3[Crop → resize 48×48 → normalize]
        M4[Model prediction]
        M5[Overlay label on frame]
    end
    
    D1 --> D2 --> D3 --> D4
    D4 --> S1
    D4 -.->|MERGED in\noriginal code| S3
    S1 -->|90/10 stratified| S2
    S1 --> T1
    S2 --> T2
    T1 & T2 --> T3
    T3 --> T4 --> T5
    T5 -->|Best checkpoint| E1
    S3 -.->|NEVER used\nduring training| E1
    E1 --> E2 --> E3 --> E4 --> E5
    T5 -->|Saved .keras| M1
    M1 --> M2 --> M3 --> M4 --> M5
    
    style S3 fill:#ffcdd2
    style E1 fill:#c8e6c9
    style T5 fill:#fff3e0
```

---

## 3. Ablation Study Design
```mermaid
flowchart LR
    subgraph FULL["Configuration A: Full Model"]
        FA[Full Architecture]
        FB[Aug: rot, flip, zoom, shear]
        FC[Dropout: 0.5/0.4/0.5]
        FD[BatchNorm: all 5 layers]
    end
    
    subgraph NOAUG["Configuration B: No Augmentation"]
        BA[Same Architecture]
        BB[Aug: rescale only]
        BC[Dropout: 0.5/0.4/0.5]
        BD[BatchNorm: all 5 layers]
    end
    
    subgraph NODROP["Configuration C: No Dropout"]
        CA[Same Architecture]
        CB[Aug: rot, flip, zoom, shear]
        CC[Dropout: 0.0 everywhere]
        CD[BatchNorm: all 5 layers]
    end
    
    subgraph NOBN["Configuration D: No BatchNorm"]
        DA[Architecture minus 5 BN layers]
        DB[Aug: rot, flip, zoom, shear]
        DC[Dropout: 0.5/0.4/0.5]
        DD[No BatchNorm]
    end
    
    SEEDS[3 Seeds: 42, 123, 2026]
    
    FULL -->|12 runs total| RUNS[(Run Matrix)]
    NOAUG --> RUNS
    NODROP --> RUNS
    NOBN --> RUNS
    SEEDS --> RUNS
    
    RUNS --> METRICS[Per Run:\n- Test Accuracy + 95% CI\n- Macro-F1 + 95% CI\n- Best val_loss & epoch\n- Epochs trained]
    METRICS --> STATS[Statistical Tests:\n- McNemar A vs B/C/D\n- Bootstrap Δ accuracy CI\n- Paired across seeds]
    
    style FULL fill:#e8f5e9
    style NOAUG fill:#fff3e0
    style NODROP fill:#fff3e0
    style NOBN fill:#fff3e0
    style RUNS fill:#e3f2fd
    style STATS fill:#fce4ec
```

**Run Matrix (12 runs):**

| Config | Description | Seeds | Priority |
|--------|-------------|-------|----------|
| A | Full model | 42, 123, 2026 | 1 (baseline) |
| B | No augmentation | 42, 123, 2026 | 2 (biggest expected effect) |
| C | No dropout | 42, 123, 2026 | 3 |
| D | No batch norm | 42, 123, 2026 | 4 |

---

## 4. Deployment Gap: Benchmark vs. Webcam
```mermaid
flowchart LR
    subgraph BENCH["Benchmark Setting (FER2013)"]
        B1[48×48 tight crop\nFace centered, ~80% of frame]
        B2[Grayscale, normalized [0,1]]
        B3[Fixed pose: near-frontal]
        B4[Controlled lighting\n(in dataset collection)]
        B5[No temporal context\nsingle frame classification]
    end
    
    subgraph WEBCAM["Webcam Demo Setting"]
        W1[Detector rectangle\nLoose: hair, bg, shoulders]
        W2[Resize detector crop → 48×48\nFace SMALLER than training]
        W3[Color → grayscale conversion]
        W4[Variable lighting:\nbacklit, dim, color cast]
        W5[Unconstrained pose:\nprofile, tilt, occlusion]
        W6[Temporal flicker:\nframe-by-frame independent]
    end
    
    subgraph SHIFTS["Four Distribution Shifts"]
        SF1[Framing Shift\nDetector includes non-face regions]
        SF2[Illumination Shift\nWebcam ≠ dataset lighting]
        SF3[Geometry Shift\nNo alignment, rotation, profile]
        SF4[Temporal Shift\nNo smoothing, single-frame]
    end
    
    subgraph FIXES["Proposed Mitigations"]
        FX1[Landmark detection\nMTCNN / RetinaFace / MediaPipe]
        FX2[Histogram equalization\nor CLAHE pre-processing]
        FX3[Landmark-based alignment\nSimilarity transform]
        FX4[Temporal smoothing\nEMA or majority vote window]
    end
    
    BENCH -->|Ideal input| MODEL[CNN Classifier]
    WEBCAM -->|Actual input| MODEL
    BENCH -.->|Training distribution| MODEL
    WEBCAM -.->|Test distribution| MODEL
    
    BENCH --> SHIFTS
    WEBCAM --> SHIFTS
    SHIFTS --> FIXES
    
    style BENCH fill:#e8f5e9
    style WEBCAM fill:#ffcdd2
    style SHIFTS fill:#fff3e0
    style FIXES fill:#e3f2fd
```

**Quantifying the Framing Shift:**

```
FER2013 Training Crop:          Webcam Detector Crop:
┌─────────────────────┐         ┌─────────────────────┐
│  👤  FACE 80%      │         │  👤  FACE 40%       │
│   (centered)       │         │  👔  SHOULDER 20%   │
│                   │         │  🌲  BACKGROUND 40% │
└─────────────────────┘         └─────────────────────┘
       ↓ resize 48×48                ↓ resize 48×48
┌─────────────────────┐         ┌─────────────────────┐
│  FACE fills frame  │         │  FACE is SMALL      │
│  Features clear    │         │  Features BLURRED   │
└─────────────────────┘         └─────────────────────┘
```

---

## 5. Receptive Field Growth (Visual)
```mermaid
graph LR
    subgraph RF["Receptive Field Progression"]
        direction TB
        L1[Input: 48×48\nRF: 1×1]
        L2[Conv 3×3, valid\nRF: 3×3]
        L3[Conv 3×3, same\nRF: 5×5]
        L4[Pool 2×2\nRF: 6×6, stride 2]
        L5[Conv 3×3, same\nRF: 10×10]
        L6[Conv 3×3, same\nRF: 14×14]
        L7[Pool 2×2\nRF: 16×16, stride 4]
        L8[Conv 3×3, same\nRF: 24×24]
        L9[Conv 3×3, same\nRF: 32×32]
        L10[Pool 2×2\nRF: 34×34, stride 8]
        L11[Final: ~40×40\ncovers 83% of face]
    end
    
    L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7 --> L8 --> L9 --> L10 --> L11
    
    style L1 fill:#e1f5fe
    style L11 fill:#c8e6c9
```

---

## 6. Class Imbalance Visualization
```mermaid
pie title FER2013 Training Distribution (28,709 images)
    "Happy (7,215)" : 7215
    "Neutral (4,965)" : 4965
    "Sad (4,830)" : 4830
    "Fear (4,097)" : 4097
    "Angry (3,995)" : 3995
    "Surprise (3,171)" : 3171
    "Disgust (436)" : 436
```

**Imbalance Ratio:** Happy : Disgust = **16.5 : 1**

---

## How to Embed in Thesis

### Option 1: Mermaid (renders in GitHub, GitLab, VS Code, Obsidian, Notion)
```markdown
```mermaid
graph TD
    A[Input] --> B[Conv]
    B --> C[Output]
```
```

### Option 2: Static Images (for PDF)
Export from Mermaid Live Editor (https://mermaid.live) → Save as SVG/PNG → Include in LaTeX/Word.

### Option 3: ASCII Fallback (already in chapters)
The current chapters use ASCII diagrams that work everywhere.

---

## Diagram Checklist for Thesis

- [x] Architecture diagram (Ch 3)
- [x] Pipeline flowchart (Ch 3)
- [x] Ablation design (Ch 5)
- [x] Deployment gap (Ch 3/6)
- [x] Receptive field growth (Ch 3)
- [x] Class imbalance (Ch 2/3)
- [ ] Training curves (Ch 5 - needs actual runs)
- [ ] Confusion matrix (Ch 5 - needs actual runs)
- [ ] Ablation box plots (Ch 5 - needs actual runs)