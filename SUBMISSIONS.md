# 🔥 WiDS 2026 — Submissions Log

**Current Best:** `0.97284` (submission_v20_MEGA.csv)  
**Goal:** Beat 0.97284

---

### 🥇 SLOT 1 — `submission_v21_C.csv`
**Submit at:** ~1:20 AM IST, Apr 6  
**Description:** 3-zone gate (far=0.001, active=0.999, static=40% ML + 60% h_blend)  
**Why:** Preserves proven h_blend ranking + adds ML signal. Tighter floor than v18_BLEND. Pushes 4 active events from ~0.87→0.999 (Brier gain ~0.014/event). Highest probability of beating 0.97216.

---

### 🥈 SLOT 2 — `submission_v20_MEGA.csv`
**Submit at:** After SLOT 1 result  
**Description:** 3-zone gate with independent calibration pipeline  
**Why:** Completely different blending approach from v21_C. If the v21 pipeline has any flaw, v20_MEGA avoids it. Diversity hedge — catches errors the others miss.

---

### 🥉 SLOT 3 — `submission_v21_A.csv`
**Submit at:** After SLOT 2 result  
**Description:** 3-zone gate + 100% pure ML for static zone (zero h_blend)  
**Why:** Most different from everything submitted so far. If h_blend has a hidden systematic error that ALL blended files inherit, v21_A is the ONLY file that escapes it. High risk, high reward.
