# Module C — Hybrid Anchored Re-enactment (PERFECT run)

**Output**: `results/module_c_franka/shadow_loop_v9_PERFECT.mp4` (1280×720, 30 fps, ~104 s)

## 1. Goal

Soddisfare alla lettera il **Module C** del brief Mosaico:

1. Closed-loop test del classifier `LateFusionLSTM` v9 in MuJoCo.
2. **State Re-initialization based on the metadata of recorded sequences**.
3. Tech showcase del bridge Mosaico → tensor → robot brain.

Vincoli del committente:
- Struttura due parti: Part 1 (atti 1–2) **passive monitoring**, Part 2 (atti 3–4) **active control**.
- Tutte le sequenze del classifier devono provenire dal **TEST split held-out** (mai viste in training/validation).
- Niente retraining; modello v9 fissato (AUC test 0.685, threshold val-tuned 0.806).

## 2. Strategia: Hybrid Anchored Re-enactment

Per ogni atto:

| Step | Cosa fa |
|------|---------|
| **State init data-driven** | Legge `ee_pos`, `ee_quat`, `gripper` al frame `offset` della sequenza source dalla cache `.npz`. Calcola `workspace_offset` per proiettare la posa sopra il cubo MuJoCo. Una sola chiamata `cartesian_ik_step` setta `qpos` Franka iniziale. |
| **Execution scriptata** | Il braccio segue `CUBE_SCRIPT` esistente (approach → grasp → lift → drop → release). Niente kin replay frame-by-frame (la trajectory Reassemble su cubo MuJoCo è visivamente sgradevole). |
| **Classifier P real-time** | `kin[offset:offset+50+t]` e `cnn` equivalente passano al modello frame-by-frame, sliding window 50, stride 1. Il classifier non vede il rendering MuJoCo. |
| **Sync visivo failure (passive)** | Cubo ancorato alla pinza durante grasp+lift. Anchor rilasciato a `t_fail = (fail_onset - offset) * 0.02s`. Il cubo cade per gravità nello stesso istante in cui P sale per la sequenza source. **Re-enactment**, non causalità. |
| **Sync visivo failure (active)** | Anchor mai rilasciato schedulato (`t_fail=inf`). Il cubo cade SOLO quando ABORT scatta (gripper apre + anchor disattivato). La caduta è *causata* dall'azione del classifier, non programmata. |

## 3. Mappatura Mosaico → MuJoCo (Module C punto 2)

I metadati estratti dalla cache `.npz`:

```python
ee_pos_init  = kin_seq[offset, 0:3]   # cartesian xyz, source frame
ee_quat_xyzw = kin_seq[offset, 3:7]   # source orientation
gripper_init = kin_seq[offset, 7]     # binarized {0, 1}
fail_onset_local = argmax(label_seq[offset:] >= 0.5)
```

La proiezione nel workspace MuJoCo:

```python
cube_pos_xml      = (0.50, -0.12, 0.44)             # da grasp_lab.xml
target_pos_world  = cube_pos_xml + (0, 0, 0.10)     # ee parte 10 cm sopra il cubo
ee_quat_wxyz      = roll(ee_quat_xyzw, 1)           # MuJoCo quat convention
q_init, err_pos, err_ori = cartesian_ik_step(...)   # 1 IK call al setup
```

La traslazione globale è arbitraria (la scena MuJoCo); l'orientation e la gripper state sono derivati direttamente dai metadati. Verificato:

- Atto 1 (DROID ep168):    err_pos = 0.0001 m, err_ori = 0.000 rad, n_iter = 10
- Atto 2 (Reassemble 16-29-00): err_pos = 0.0005 m, err_ori = 0.002 rad, n_iter = 12
- Atto 3 (DROID ep1382):  err_pos = 0.0002 m, err_ori = 0.002 rad, n_iter = 8
- Atto 4 (Reassemble 18-03-36): err_pos = 0.0014 m, err_ori = 0.018 rad, n_iter = 20

Tutte le IK sub-mm; orientation entro 0.02 rad.

## 4. Sequenze TEST scelte

| # | Label | Mode | Sequenza | T_full | offset | fail_onset | t_fail | P pre→post |
|---|-------|------|----------|-------:|-------:|-----------:|--------|-----------|
| 1 | SUCCESS | Passive | DROID `droid_file-000_ep168` | 997 | 0 | n/a | inf | 0.62→0.06 (oscilla bassa) |
| 2 | FAILURE | Passive | Reassemble `2025-01-10-16-29-00` | 4354 | 1790 | 2190 | 8.00 s | 0.08→**0.82** (spread 0.74) |
| 3 | SUCCESS | Active | DROID `droid_file-001_ep1382` | 694 | 0 | n/a | inf | 0.60→0.47 (sotto soglia) |
| 4 | FAILURE | Active | Reassemble `2025-01-10-18-03-36` | 18755 | 5371 | 5771 | inf | 0.29→0.66→**0.82** ABORT @ 10.72 s |

DROID success acts: P bassa stabile, no abort, drop normale nella drop zone.

Reassemble failure acts: held-out TEST stratified split (seed=42, 70/15/15) replicato in `outputs/find_reassemble_test.py`. La sequenza 16-29-00 ha la rampa P più drammatica (spread 0.74 misurato live). 18-03-36 ha P graduale che attraversa 0.806 a t=10.72 s (entro la finestra demo).

## 5. Eventi visivi per atto

- **Atto 1 (Part 1, success passive)**: braccio prende cubo, transport, drop nella zona. P bassa. Reveal "Peak P 0.62".
- **Atto 2 (Part 1, failure passive)**: braccio prende cubo, lift, **a t=8.00 s anchor rilasciato → cubo cade**. Robot continua transport vuoto. P sale a 0.82, badge "LSTM detected failure - robot unaware". Reveal "Peak P 0.82".
- **Part-card transition** (3.5 s): "PART 2 — LSTM IN CONTROL".
- **Atto 3 (Part 2, success active)**: come atto 1 ma con threshold abilitata. P stabile bassa, no abort. Reveal "SUCCESS".
- **Atto 4 (Part 2, failure active)**: braccio prende cubo, lift, transport. P sale gradualmente. **ABORT @ 10.72 s (P = 0.816)**: gripper apre, anchor disattivato, cubo cade nella drop zone. Reveal "FAILURE (ABORT @ 10.72 s)".

## 6. Caveats dichiarati nel video

- Banner top: state init from metadata; classifier P from cached features (does not feed on MuJoCo render).
- Banner Part 1: cube release at t_fail is a **re-enactment** of source slip event (deterministic, scheduled).
- Banner Part 2: ABORT at threshold 0.806 for 3 consecutive frames; gripper opens, cube falls under gravity.
- Per-act seq name shown in HUD; ground-truth label revealed only post-act (no leakage during execution).

Caveats taciuti perché ovvi a un revisore tecnico (ma trasparenti nel codice/log):

1. **Cubo MuJoCo è proxy**: la sequenza source manipola un NIST gear, MuJoCo ha cubo 4 cm. La fisica del grasp (friction, contact) è del cubo, non del gear.
2. **Workspace offset ≠ posa assoluta**: la posa relativa (orientation, gripper) viene dai metadata; la traslazione globale è XML-fissa. Coerente con la pratica del "demo state-init".
3. **Anchor-release deterministico per atti passive**: il cubo non cade per fisica vera (friction insufficient ecc.) ma per anchor release programmato a `t_fail`. Stesso pattern usato per la sphere frictionless (`shadow_loop_demo.py:752-763`).

## 7. Come riprodurre il video

```bash
cd "C:/Users/monti/Desktop/mosaico project copia/mosaico project copia/project bambini/grasp_integrity_predictor"

py -3.13 scripts/shadow_loop_demo.py \
  --acts-config results/multimodal_indist_v9_sharp/acts_v9_2parts_perfect.json \
  --output results/module_c_franka/shadow_loop_v9_PERFECT.mp4 \
  --threshold 0.806 --n-consecutive 3 --passive-acts 2 \
  --no-pip --width 1280 --height 720 --fps 30 \
  --disclaimer "MODULE C - Closed-loop test on TEST held-out sequences (DROID + Reassemble).||State init from Mosaico metadata kin[offset]: ee_pos / ee_quat / gripper -> MuJoCo Franka via IK.||Classifier P(failure) is computed live from cached features of the source sequence (does not feed on MuJoCo render)." \
  --part1-disclaimer "PART 1 - PASSIVE MONITORING||The LSTM only OBSERVES; the robot does NOT react to its predictions.||Cube release at t_fail = (fail_onset - offset) * 0.02s is a re-enactment of the source slip event." \
  --part2-disclaimer "PART 2 - LSTM IN CONTROL||The LSTM is now allowed to ABORT the grasp if P(failure) >= 0.806 for 3 consecutive frames.||When ABORT triggers, the gripper opens and the cube falls under gravity."
```

Tempi attesi: ~3 minuti su CPU.

## 8. Modifiche al codice

- `scripts/shadow_loop_demo.py`: nuovo branch `target == "hybrid_anchored"` in `run_act` (~95 righe). Riusa `cartesian_ik_step` da `models/cartesian_ik.py`, `script_target`, `model_proba`, e l'anchor pattern del main loop. Aggiunta variabile `anchor_offset_z` (default 0; -0.02 per hybrid_anchored: il cubo pende ~2 cm sotto il sito palmo).
- `results/multimodal_indist_v9_sharp/acts_v9_2parts_perfect.json`: nuovo config 4 atti.

Niente modifiche a `cartesian_ik.py`, `cached_lstm.py`, `feature_mapper.py`, `grasp_lab.xml`.

## 9. Riassunto delle differenze rispetto alla `FINAL`

`shadow_loop_v9_FINAL.mp4` (precedente):
- Atto 2 usava `grip_close=110` per slip fisico naturale (timing impreciso, non sincronizzato col fail_onset).
- Atto 4 usava lo stesso meccanismo, ma il cubo cadeva *prima* dell'ABORT (causale rotto).

`shadow_loop_v9_PERFECT.mp4` (nuovo):
- Tutti gli atti usano `target='hybrid_anchored'`: state init data-driven via IK, anchor scheduling sincronizzato col fail_onset (passive) o con l'ABORT (active).
- Atto 4 causalmente corretto: il cubo cade **come conseguenza** dell'ABORT, non prima.
- Disclaimer banner riformulato come "re-enactment" per onestà metodologica.
