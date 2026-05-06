# Module C — Kinematic Replay (REPLAY run)

**Output**: `results/module_c_franka/shadow_loop_v9_REPLAY.mp4` (1280×720, 30 fps, ~104 s)

Versione **non scriptata**: il Franka MuJoCo segue la cinematica reale della sequenza source frame-by-frame via Cartesian IK. Da confrontare con `shadow_loop_v9_PERFECT.mp4` (versione scriptata).

## 1. Differenze chiave rispetto a `PERFECT`

| Aspetto | PERFECT (scriptato) | REPLAY (non scriptato) |
|---------|---------------------|-------------------------|
| Movimento del braccio | CUBE_SCRIPT (waypoint hardcoded HOME → CUBE_APPROACH → ...) | **Cinematica reale**: `kin[offset:offset+600, 0:7]` (pos+quat) della sequenza source, IK Cartesian frame-by-frame |
| Init pose | `lab_home` keyframe canonica | Warm-up 1.0 s che blenda HOME → primo frame replay (smooth) |
| Workspace alignment | Posizione XML standard | `replay_offset_xyz` traslata così che il source ee al frame di grasp si allinei al cubo XML |
| Smoothing | n/a (waypoint discreti) | **Savitzky-Golay** (window 15, order 3) su position e quaternion (riduce micro-vibrazioni del Franka loggato a 50 Hz) |
| Anchor del cubo | No (fisica gestisce grasp) | Sì, durante grasp source (gripper closed AND label<0.5 AND ee_z above table) |
| Trigger del slip in atto 2 passive | `cube_fail_script` (gripper open programmato) | Anchor release automatico quando `label_seq` transita a 1 (vero fail event) |
| Trigger del fail in atto 4 active | LSTM ABORT (P ≥ 0.806) → gripper open | LSTM ABORT (P ≥ 0.806) → gripper open + anchor release |
| Movimento atti success (1, 3) | CUBE_SCRIPT in drop zone canonica | Traiettoria reale DROID (potrebbe non finire nella drop zone) |

## 2. Miglioramenti rispetto al precedente replay (`fa schifo`)

Il branch `target='replay'` esisteva ma con problemi noti. Le modifiche fatte:

1. **Cubo a posizione XML canonica** invece di `seq_window[z_min, :2]` (random spot). L'allineamento avviene tramite `replay_offset_xyz` che mappa il source grasp moment a `cube_pos_xml`.
2. **Workspace alignment XY+Z** (prima solo Z): l'intera traiettoria source è traslata in `(dx, dy, dz)` per match il cubo MuJoCo.
3. **Savitzky-Golay smoothing** sul kin source: cuts del jitter del Franka logger.
4. **Warm-up 1.0 s**: il braccio non snap-a alla prima posa Reassemble; interpola da HOME smoothly.
5. **Anchor Z offset 0** (era −0.02): il cubo non entra più nel tavolo durante il descent.
6. **Anchor condizionato a `ee_z ≥ table_top + cube_half`**: niente anchor quando il braccio è più basso del cubo (evita teleport del cubo dentro al tavolo).
7. **Anchor disattivato durante warm-up**: il cubo rimane sul tavolo finché il braccio non raggiunge la posa replay.

## 3. Sequenze e timing

Stesse 4 sequenze TEST held-out di `PERFECT`. Stesso pre-compute classifier (P live frame-by-frame). Stesso threshold (0.806) e n_consecutive (3) per l'ABORT.

| # | Label | Mode | Sequenza | offset | Source fail_onset_local (frame relativo offset) | Risultato osservato |
|---|-------|------|----------|--------|---------------------------------|---------------------|
| 1 | SUCCESS | Passive | DROID `droid_file-000_ep168` | 0 | n/a | Movimento DROID reale; cubo afferrato e depositato in posizione DROID end. P bassa (peak 0.62). |
| 2 | FAILURE | Passive | Reassemble `2025-01-09-17-14-59` | 4948 | 400 (t=8.0 s) | A t≈8 s anchor release (label transition); cubo cade dalla pinza. Badge "LSTM detected failure" si attiva a t≈9.2 s (P=0.70). |
| 3 | SUCCESS | Active | DROID `droid_file-001_ep1382` | 0 | n/a | Movimento DROID; P stabile bassa, no ABORT. |
| 4 | FAILURE | Active | Reassemble `2025-01-10-16-29-00` | 1790 | 400 (t=8.0 s) | **ABORT @ 8.02 s** (P=0.815). Gripper open, cubo cade. |

## 4. Caveats specifici della versione REPLAY

1. **Atti success DROID non finiscono nel drop zone canonica**: la traiettoria DROID reale termina dove il robot reale ha terminato il task (posizione arbitraria sul tavolo), non nella zona blu/verde della scena MuJoCo. Coerente con la cinematica reale.
2. **Atto 2 passive — slip non sincronizzato col model awareness**: il cubo cade alla `label transition` (t=8 s) della sequenza source, ma la P del classifier sale dopo (P=0.70 a t=9.2 s). Il visivo precede leggermente la consapevolezza del modello. Onesto: è il vero fail event della sequenza Reassemble.
3. **Atto 4 active — pre-grasp basso**: la sequenza 16-29-00 è un manipolo di gear NIST in posizione quasi piatta sul board; il "lift" è minimo. Il cubo a posizione MuJoCo standard dà l'illusione che il robot afferri il cubo low e lo manipoli low — coerente col gesto reale del Franka source.
4. **Cubo è proxy del NIST gear**: stesso caveat di PERFECT.

## 5. Come riprodurre

```bash
cd "C:/Users/monti/Desktop/mosaico project copia/mosaico project copia/project bambini/grasp_integrity_predictor"

py -3.13 scripts/shadow_loop_demo.py \
  --acts-config results/multimodal_indist_v9_sharp/acts_v9_replay.json \
  --output results/module_c_franka/shadow_loop_v9_REPLAY.mp4 \
  --threshold 0.806 --n-consecutive 3 --passive-acts 2 \
  --no-pip --width 1280 --height 720 --fps 30 \
  --disclaimer "MODULE C - REPLAY: kinematic playback of TEST held-out sequences (DROID + Reassemble).||Franka in MuJoCo follows source ee_pose via Cartesian IK frame-by-frame (Savitzky-Golay smoothed); workspace alignment to canonical cube position.||Classifier P(failure) computed live from cached source-sequence features." \
  --part1-disclaimer "PART 1 - PASSIVE MONITORING (kin replay)||The arm replays the SOURCE Reassemble trajectory; LSTM only OBSERVES.||Cube release on source label transition (visualization of the real slip event)." \
  --part2-disclaimer "PART 2 - LSTM IN CONTROL (kin replay)||Source trajectory replayed; LSTM may ABORT if P(failure) >= 0.806 for 3 consecutive frames.||When ABORT triggers, gripper opens, cube falls under gravity."
```

Tempi attesi: ~3 minuti su CPU (più lento di PERFECT per via dell'IK frame-by-frame).

## 6. Modifiche al codice (`scripts/shadow_loop_demo.py`)

Tutte nel branch `target == "replay"` di `run_act` e nel main loop:

- **Setup** (~50 righe nuove): Savitzky-Golay smoothing, grasp-frame detection (gripper closure o z_min fallback), workspace alignment XY+Z, warm-up state, cubo a XML position.
- **Main loop** (~25 righe nuove): warm-up blend HOME → first replay pose nei primi 1.0 s, gripper open durante warm-up, anchor con offset Z=0 e gate `ee_z ≥ table_top + cube_half`.

Niente modifiche a `cartesian_ik.py`, `cached_lstm.py`, `feature_mapper.py`, `grasp_lab.xml`.

## 7. Coesistenza con PERFECT

I due deliverable coesistono:
- `acts_v9_2parts_perfect.json` + `target='hybrid_anchored'` → `shadow_loop_v9_PERFECT.mp4` (scriptato)
- `acts_v9_replay.json` + `target='replay'` → `shadow_loop_v9_REPLAY.mp4` (kin reale)

L'utente può confrontarli affiancati e scegliere quale presentare al committente Mosaico.
