# EDA Report — WiDS 2026 Wildfire Survival Analysis

## Dataset Overview
- **Train:** (221, 39)
- **Test:** (95, 35)
- **Events:** 69 (31.2%)
- **Censored:** 152 (68.8%)
- **Time range:** 0.001 - 66.994 hours

## Event Distribution by Horizon
| Horizon | Events | Rate |
|---------|--------|------|
| 12h | 49 | 22.2% |
| 24h | 63 | 28.5% |
| 48h | 66 | 29.9% |
| 72h | 69 | 31.2% |

## Column Profiling
| Column | Type | Nulls | Unique | Skew | Corr w/ Event |
|--------|------|-------|--------|------|---------------|
| event_id | int64 | 0 | 221 | 0.015 | -0.0547 |
| num_perimeters_0_5h | int64 | 0 | 12 | 3.174 | 0.3705 |
| dt_first_last_0_5h | float64 | 0 | 62 | 1.409 | 0.353 |
| low_temporal_resolution_0_5h | int64 | 0 | 2 | -1.035 | -0.3791 |
| area_first_ha | float64 | 0 | 221 | 4.721 | -0.1813 |
| area_growth_abs_0_5h | float64 | 0 | 26 | 11.231 | 0.1583 |
| area_growth_rel_0_5h | float64 | 0 | 26 | 11.952 | 0.166 |
| area_growth_rate_ha_per_h | float64 | 0 | 26 | 10.282 | 0.1724 |
| log1p_area_first | float64 | 0 | 221 | -0.112 | -0.1679 |
| log1p_growth | float64 | 0 | 25 | 3.635 | 0.2927 |
| log_area_ratio_0_5h | float64 | 0 | 26 | 6.338 | 0.2293 |
| relative_growth_0_5h | float64 | 0 | 26 | 11.952 | 0.166 |
| radial_growth_m | float64 | 0 | 26 | 6.429 | 0.2093 |
| radial_growth_rate_m_per_h | float64 | 0 | 26 | 6.47 | 0.215 |
| centroid_displacement_m | float64 | 0 | 26 | 6.753 | 0.208 |
| centroid_speed_m_per_h | float64 | 0 | 26 | 6.959 | 0.2093 |
| spread_bearing_deg | float64 | 0 | 26 | 4.142 | 0.281 |
| spread_bearing_sin | float64 | 0 | 26 | 1.317 | 0.1883 |
| spread_bearing_cos | float64 | 0 | 26 | -3.487 | -0.3232 |
| dist_min_ci_0_5h | float64 | 0 | 221 | 1.635 | -0.4814 |
| dist_std_ci_0_5h | float64 | 0 | 20 | 11.589 | 0.142 |
| dist_change_ci_0_5h | float64 | 0 | 19 | -11.187 | -0.1064 |
| dist_slope_ci_0_5h | float64 | 0 | 53 | -11.981 | -0.1153 |
| closing_speed_m_per_h | float64 | 0 | 19 | 10.994 | 0.1074 |
| closing_speed_abs_m_per_h | float64 | 0 | 19 | 11.152 | 0.1387 |
| projected_advance_m | float64 | 0 | 19 | 11.187 | 0.1064 |
| dist_accel_m_per_h2 | float64 | 0 | 36 | -8.269 | -0.0726 |
| dist_fit_r2_0_5h | float64 | 0 | 20 | 3.885 | 0.1431 |
| alignment_cos | float64 | 0 | 62 | -0.031 | 0.1199 |
| alignment_abs | float64 | 0 | 62 | 1.614 | 0.3491 |
| cross_track_component | float64 | 0 | 26 | 2.425 | -0.0583 |
| along_track_speed | float64 | 0 | 26 | -3.77 | 0.0081 |
| event_start_hour | int64 | 0 | 22 | -0.987 | 0.0474 |
| event_start_dayofweek | int64 | 0 | 7 | 0.169 | -0.1193 |
| event_start_month | int64 | 0 | 9 | -1.188 | 0.0933 |
| time_to_hit_hours | float64 | 0 | 221 | -0.226 | -0.7195 |
| event | int64 | 0 | 2 | 0.816 | 1.0 |

## Key Findings

1. **Distance is king:** `dist_min_ci_0_5h` has the strongest correlation with event (r=-0.48). Close fires hit.
2. **Alignment matters:** `alignment_abs` (r=+0.35) — fires heading directly toward the zone are more dangerous.
3. **Data richness proxy:** `num_perimeters_0_5h` (r=+0.37) — more observations = closer/more active fire.
4. **Tight time window:** 49/69 events happen within 12h. Only 3 events between 48h and 72h.
5. **Zero train-test drift:** No feature shows meaningful distribution shift between train and test.
6. **EPV concern:** 69 events / 25 features = EPV of 2.76. Need ≤7 features for EPV≥10.

## Survival Plots
- See `plots/km_overall.png`, `plots/km_by_distance.png`, `plots/km_by_alignment.png`

## Leakage Risk Assessment
- No post-event features detected in raw data
- Censoring indicator properly preserved
- Time-to-hit only used for label creation, never as a feature
