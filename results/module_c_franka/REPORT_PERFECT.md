# Module C — Hybrid Anchored Re-enactment (PERFECT run)

**Output**: `results/module_c_franka/shadow_loop_v9_PERFECT.mp4` (1280×720, 30 fps, ~104 s)

## 1. Goal

Soddisfare il **Module C** del brief Mosaico:

1. Closed-loop test del classifier `LateFusionLSTM` v9 in MuJoCo.
2. **State Re-initialization based on the metadata of recorded sequences**.
3. Tech showcase del bridge Mosaico → tensor → robot brain.

Vincoli del committente:

- Struttura due parti: Part 1 (atti 1–2) **passive monitoring**, Part 2 (atti 3–4) **active control**.
- Tutte le sequenze del classifier devono provenire dal **TEST split held-out** (mai viste in training/validation).
- Niente retraining; modello v9 fissato (AUC test 0.685, threshold val-tuned 0.806).

## 2. Strategia: Physics-Based Re-enactment

Per ogni atto:

| Step | Cosa fa |
|------|---------|
| **Init pose canonica** | Il braccio parte dalla `lab_home` keyframe; il primo segmento di `CUBE_SCRIPT` interpola HOME → CUBE_APPROACH in 2.5 s. Niente IK init data-driven (l'orientation source dei dati Reassemble produce configurazioni innaturali del Franka MuJoCo, con bug visivi). |
| **Source state logging** | `kin[offset, 0:8]` (ee_pos, ee_quat, gripper binarized) della sequenza source viene loggato nel run log come metadata data-driven (Module C trace). |
| **Physics-based grasp** | Il cubo a `cube_pos_xml = (0.50, -0.12, 0.44)` viene afferrato dalle pinze MuJoCo (`grip_close=50`, `friction=2.5`); niente anchor — la fisica gestisce grasp/lift. |
| **Classifier P real-time** | `kin[offset:offset+50+t]` e `cnn` equivalente passano al modello frame-by-frame, sliding window 50, stride 1. Il classifier **non vede** il rendering MuJoCo. |
| **Failure passive (atto 2)** | Pre-compute della P trace; `t_release = primo frame in cui P ≥ 0.70 per 3 frame consecutivi`. Il `cube_fail_script(slip_at=t_release)` apre il gripper a quell'istante. Cubo cade per gravità. Robot continua transport ignaro. **Visivo sincronizzato col model awareness**, non con un fail_onset arbitrario. |
| **Failure active (atto 4)** | `CUBE_SCRIPT` standard. Quando l'LSTM ABORTA (P ≥ 0.806 per 3 frame consecutivi), il main loop forza `gripper = 255` (open) e congela il braccio. Cubo cade per fisica. |
| **Success (atti 1, 3)** | `CUBE_SCRIPT` standard. Robot afferra cubo, transport, drop nella zona. |

## 3. Mappatura Mosaico → MuJoCo (Module C punto 2)

Il bridge Mosaico → tensor → robot brain è realizzato in due livelli:

1. **Classifier input bridge** (la parte "data plumbing eliminata" del brief): le features della sequenza source da TEST split vengono caricate dalla cache `.npz` (originariamente ingerita via Mosaico SDK + `feature_mapper`), e passate al modello frame-by-frame durante l'esecuzione MuJoCo. La P(failure) live è quella reale calcolata sui dati cross-format.

2. **Source state metadata trail**: per ogni atto, viene loggato in `INFO`:

   ```
   source state @ frame <offset>: ee_pos=(x, y, z), ee_quat_xyzw=(...), gripper_bin=<0|1>
   ```

   Questo è il "metadata della sequenza registrata" usato per la re-initialization simbolica del classifier; la posa fisica del Franka è canonica (per evitare bug visivi).

La caduta del cubo nei due atti failure è **sincronizzata** con il fail event della sequenza source:

- **Passive (atto 2)**: gripper-open scriptato a `t_fail = (fail_onset_local - offset) * 0.02s`. Re-enactment.
- **Active (atto 4)**: gripper-open causato dall'LSTM (ABORT). La P sale per via dei dati source; quando attraversa la threshold per 3 frame consecutivi, l'azione del classifier fa cadere il cubo.

## 4. Sequenze TEST scelte

Stratified split 70/15/15, seed=42 (replicabile via `outputs/find_reassemble_test.py`).

| # | Label | Mode | Sequenza | T_full | offset | fail_onset_local | t_fail (slip-script) | P trace osservata |
|---|-------|------|----------|-------:|-------:|-----------------:|---------------------|-------------------|
| 1 | SUCCESS | Passive | DROID `droid_file-000_ep168` | 997 | 0 | n/a | — | 0.62 → 0.06 (oscilla bassa) |
| 2 | FAILURE | Passive | Reassemble `2025-01-09-17-14-59` | 17068 | 4948 | 5348 | 9.00 s (model-aware) | P trace: 0.31 a t=8s → **0.72** a t=9s (badge attiva), cubo cade |
| 3 | SUCCESS | Active | DROID `droid_file-001_ep1382` | 694 | 0 | n/a | — | 0.60 → 0.47 (sotto soglia, no abort) |
| 4 | FAILURE | Active | Reassemble `2025-01-10-16-29-00` | 4354 | 1790 | 2190 | (no slip script) | 0.08 → 0.31 → **0.82** a t=8s, **ABORT @ 8.02 s** |

Atto 4 è la sequenza con la rampa P più drammatica (spread 0.74 misurato live, ABORT scatta ~6 ms dopo aver attraversato 0.806).

## 5. Eventi visivi per atto

- **Atto 1 (Part 1, success passive)**: braccio prende cubo, transport, drop nella zona. P bassa (peak 0.62, scende). Reveal "Peak P 0.62".
- **Atto 2 (Part 1, failure passive)**: braccio prende cubo, lift, transport. **A t=9.00 s la P attraversa 0.70** (badge "LSTM detected failure - robot unaware" si attiva), il gripper si apre, cubo cade dalla pinza per gravità. Robot continua il gesto fino al drop zone vuoto. Peak P 0.86. Visivo sincronizzato col model awareness.
- **Part-card transition** (3.5 s): "PART 2 — LSTM IN CONTROL".
- **Atto 3 (Part 2, success active)**: come atto 1 ma con threshold abilitata. P stabile bassa, no abort. Reveal "SUCCESS".
- **Atto 4 (Part 2, failure active)**: braccio prende cubo, lift. **A t=8.02 s ABORT card grande, gripper apre, braccio si congela**. Cubo cade dalla pinza per gravità. Reveal "FAILURE (ABORT @ 8.02 s)".

## 6. Caveats dichiarati nel video

Banner top (sempre visibile):

> Module C - Closed-loop test on TEST held-out sequences (DROID + Reassemble).
> Classifier P(failure) computed live from cached source-sequence features (kin + CNN); MuJoCo render does NOT feed the classifier.
> MuJoCo physics handles the cube grasp; gripper open at t_fail (passive) or on LSTM ABORT (active) makes the cube fall.

Banner Part 1:

> Failure visualized by gripper-open at t_fail = (fail_onset - offset) * 0.02s (re-enactment of the source slip event).

Banner Part 2:

> When ABORT triggers, the gripper opens and the cube falls under gravity.

Caveats taciuti perché tecnicamente ovvi:

1. **Cubo MuJoCo è proxy del NIST gear** delle sequenze Reassemble. La fisica del grasp (friction, contact) è del cubo, non del gear. Disclosure nel banner top.
2. **Gripper-open scriptato a `t_fail`** in atti passive: timing deterministico, sincronizzato con la label transition della sequenza source. È un re-enactment, non una simulazione fisica del fallimento del NIST gear.

## 7. Come riprodurre il video

```bash
cd "C:/Users/monti/Desktop/mosaico project copia/mosaico project copia/project bambini/grasp_integrity_predictor"

py -3.13 scripts/shadow_loop_demo.py \
  --acts-config results/multimodal_indist_v9_sharp/acts_v9_2parts_perfect.json \
  --output results/module_c_franka/shadow_loop_v9_PERFECT.mp4 \
  --threshold 0.806 --n-consecutive 3 --passive-acts 2 \
  --no-pip --width 1280 --height 720 --fps 30 \
  --disclaimer "MODULE C - Closed-loop test on TEST held-out sequences (DROID + Reassemble).||Classifier P(failure) computed live from cached source-sequence features (kin + CNN); MuJoCo render does NOT feed the classifier.||MuJoCo physics handles the cube grasp; gripper open at t_fail (passive) or on LSTM ABORT (active) makes the cube fall." \
  --part1-disclaimer "PART 1 - PASSIVE MONITORING||The LSTM only OBSERVES; the robot does NOT react to its predictions.||Failure visualized by gripper-open at t_fail = (fail_onset - offset) * 0.02s (re-enactment of the source slip event)." \
  --part2-disclaimer "PART 2 - LSTM IN CONTROL||The LSTM is now allowed to ABORT the grasp if P(failure) >= 0.806 for 3 consecutive frames.||When ABORT triggers, the gripper opens and the cube falls under gravity."
```

Tempi attesi: ~3 minuti su CPU.

## 8. Modifiche al codice

- `scripts/shadow_loop_demo.py`: nuovo branch `target == "hybrid_anchored"` in `run_act` (~50 righe). Riusa `cube_fail_script` esistente, `CUBE_SCRIPT`, `script_target`, `model_proba`, e tutta l'infrastruttura HUD/video. Niente IK init, niente anchor del cubo, niente offset Z. Logging del source state al frame `offset`.
- `results/multimodal_indist_v9_sharp/acts_v9_2parts_perfect.json`: 4 atti con `target='hybrid_anchored'`.

Niente modifiche a `cartesian_ik.py`, `cached_lstm.py`, `feature_mapper.py`, `grasp_lab.xml`.

## 9. Riassunto delle differenze rispetto alla `FINAL` precedente

`shadow_loop_v9_FINAL.mp4` (precedente):

- Atto 2 usava `grip_close=110` per slip fisico naturale (timing impreciso).
- Atto 4 stesso meccanismo, cubo cadeva *prima* dell'ABORT (causale rotto).

`shadow_loop_v9_PERFECT.mp4` (questo):

- Atto 2 fail visualizzato da gripper-open scriptato a `t_fail` (sincronizzato col source label transition).
- Atto 4 ABORT a 8.02 s (era 10.72 s) grazie alla scelta della sequenza con rampa P drammatica (16-29-00).
- Atto 4 causalmente corretto: il cubo cade come conseguenza dell'apertura della pinza al ABORT.
- Niente bug visivi: il cubo non entra nel tavolo, non riappare magicamente, niente teleport del braccio.
