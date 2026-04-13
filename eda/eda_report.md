# EDA Report
## Dataset Profiles
- **Train Shape:** (221, 37)
- **Test Shape:** (95, 35)

### Train DataFrame Profile
,dtype,nulls,cardinality,skew
event_id,int64,0,221,0.015166738432879861
num_perimeters_0_5h,int64,0,12,3.1740664276743904
dt_first_last_0_5h,float64,0,62,1.408767783499892
low_temporal_resolution_0_5h,int64,0,2,-1.0346564583890905
area_first_ha,float64,0,221,4.721166786066688
area_growth_abs_0_5h,float64,0,26,11.231249631617034
area_growth_rel_0_5h,float64,0,26,11.951707967984877
area_growth_rate_ha_per_h,float64,0,26,10.281603532297073
log1p_area_first,float64,0,221,-0.11151885519241445
log1p_growth,float64,0,25,3.6351478748590456
log_area_ratio_0_5h,float64,0,26,6.33771516132606
relative_growth_0_5h,float64,0,26,11.951707967984877
radial_growth_m,float64,0,26,6.4287923300554475
radial_growth_rate_m_per_h,float64,0,26,6.469868362456623
centroid_displacement_m,float64,0,26,6.75309096455621
centroid_speed_m_per_h,float64,0,26,6.959181489260942
spread_bearing_deg,float64,0,26,4.141749256249927
spread_bearing_sin,float64,0,26,1.3174015859715698
spread_bearing_cos,float64,0,26,-3.486536895994804
dist_min_ci_0_5h,float64,0,221,1.6345525473387754
dist_std_ci_0_5h,float64,0,20,11.589398487444974
dist_change_ci_0_5h,float64,0,19,-11.187420090231448
dist_slope_ci_0_5h,float64,0,53,-11.980701807867629
closing_speed_m_per_h,float64,0,19,10.994164926814271
closing_speed_abs_m_per_h,float64,0,19,11.15223195509505
projected_advance_m,float64,0,19,11.187420090231448
dist_accel_m_per_h2,float64,0,36,-8.269463799299203
dist_fit_r2_0_5h,float64,0,20,3.884857086247491
alignment_cos,float64,0,62,-0.03094130035801714
alignment_abs,float64,0,62,1.6144776893170962
cross_track_component,float64,0,26,2.4254140219361164
along_track_speed,float64,0,26,-3.770070567325029
event_start_hour,int64,0,22,-0.9873532637397644
event_start_dayofweek,int64,0,7,0.16886390784342994
event_start_month,int64,0,9,-1.1876129285628645
time_to_hit_hours,float64,0,221,-0.22591642746112803
event,int64,0,2,0.8160093132336624


## Survival Analysis (Kaplan-Meier)
Plotted survival curves for different cohorts (e.g. Month) and saved to `/eda/plots/km_curves.png`.

## Watch Duty Correlations & Leakage Risks
Data dictionary mentions Watch Duty, but no explicit columns are directly mapped in raw CSVs.
Checking highly clustered physical features for colinearity:
### Top Features Correlated with Target (Event)
- **time_to_hit_hours**: 0.719
- **dist_min_ci_0_5h**: 0.481
- **low_temporal_resolution_0_5h**: 0.379
- **num_perimeters_0_5h**: 0.371
- **dt_first_last_0_5h**: 0.353
Correlation matrix saved to `/eda/plots/correlation_matrix.png`.