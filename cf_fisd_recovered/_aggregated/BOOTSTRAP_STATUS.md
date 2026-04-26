# Teacher-seed bootstrap status

- **Last update**: 2026-04-26T06:31:57.620628+00:00 UTC = 2026-04-26 14:31 SGT
- **Hard cutoff**: 2026-04-26T10:30:00+00:00 UTC = 2026-04-26 18:30 SGT
- **Cells complete**: 12/20 = **60%**

## Per-cell completion (5 datasets × 4 seeds = 20 cells)

|  | seed=1 | seed=2 | seed=3 | seed=4 |
|---|---|---|---|---|
| **sberbank-housing** | ✓ | ✓ | ✓ | ✓ |
| **ecom-offers** | ✓ | ✓ | ✓ | ✓ |
| **homesite-insurance** | · | · | · | · |
| **cooking-time** | ✓ | ✓ | ✓ | ✓ |
| **delivery-eta** | · | · | · | · |

## Bootstrap rho summary per dataset

| Dataset | n seeds done | ρ(XGB,LGBM) range | ρ(LGBM,CAT) range | Rule prediction | Bootstrap-robust? | Actual outcome | Correct? |
|---|---|---|---|---|---|---|---|
| sberbank-housing | 4 | [+0.525, +0.566] | [+0.671, +0.705] | match | NO | match | YES |
| ecom-offers | 4 | [+0.260, +0.500] | [+0.049, +0.213] | match | YES | match | YES |
| homesite-insurance | 0 | - | - | - | - | WIN | - |
| cooking-time | 4 | [-0.078, -0.016] | [+0.820, +0.838] | match | YES | match | YES |
| delivery-eta | 0 | - | - | - | - | match | - |

## Recommendation

In progress. 60% complete, +4.0 hours until hard cutoff. Re-evaluating every 30 min.

## Raw JSON

```json
{
  "completion_pct": 60.0,
  "cells_done": 12,
  "cells_total": 20,
  "hard_cutoff_utc": "2026-04-26T10:30:00+00:00",
  "last_update_utc": "2026-04-26T06:31:57.620628+00:00",
  "rule_robust_overall": false,
  "all_classified_correct_at_mean": true,
  "full_coverage": false,
  "branch_a_ready": false,
  "per_dataset": {
    "sberbank-housing": {
      "n_seeds": 4,
      "seeds": [
        1,
        2,
        3,
        4
      ],
      "xgb_lgbm": {
        "mean": 0.5432454147996252,
        "lo": 0.5247571511062563,
        "hi": 0.5660552445854423,
        "std": 0.018108628420502385,
        "values": [
          0.5335904648945605,
          0.5485787986122417,
          0.5660552445854423,
          0.5247571511062563
        ]
      },
      "lgbm_cat": {
        "mean": 0.6877463646463517,
        "lo": 0.6706499161747216,
        "hi": 0.7052946758352096,
        "std": 0.017717292531331308,
        "values": [
          0.6706499161747216,
          0.674455719398016,
          0.7005851471774602,
          0.7052946758352096
        ]
      },
      "xgb_cat": {
        "mean": 0.5553436287785072,
        "lo": 0.5296236108278876,
        "hi": 0.6100575232812733,
        "std": 0.03730582581028558,
        "values": [
          0.5337854617013844,
          0.5479079193034835,
          0.6100575232812733,
          0.5296236108278876
        ]
      },
      "rule_predictions_under_bootstrap": [
        "WIN",
        "match"
      ],
      "rule_robust": false,
      "rule_predicted": "match",
      "actual_outcome": "match",
      "classified_correctly": true
    },
    "ecom-offers": {
      "n_seeds": 4,
      "seeds": [
        1,
        2,
        3,
        4
      ],
      "xgb_lgbm": {
        "mean": 0.411053912719886,
        "lo": 0.2603027874035927,
        "hi": 0.4999662069286935,
        "std": 0.10425220912349412,
        "values": [
          0.2603027874035927,
          0.44755234966693164,
          0.4999662069286935,
          0.4363943068803263
        ]
      },
      "lgbm_cat": {
        "mean": 0.1417394385771362,
        "lo": 0.04906723296921267,
        "hi": 0.2127380081104389,
        "std": 0.07042645884115507,
        "values": [
          0.2127380081104389,
          0.17511345360469702,
          0.13003905962419618,
          0.04906723296921267
        ]
      },
      "xgb_cat": {
        "mean": 0.11585771329742314,
        "lo": 0.04398365655681021,
        "hi": 0.15855413077373343,
        "std": 0.051245536503885435,
        "values": [
          0.15855413077373343,
          0.04398365655681021,
          0.14567452722511542,
          0.11521853863403354
        ]
      },
      "rule_predictions_under_bootstrap": [
        "match"
      ],
      "rule_robust": true,
      "rule_predicted": "match",
      "actual_outcome": "match",
      "classified_correctly": true
    },
    "homesite-insurance": {
      "n_seeds": 0
    },
    "cooking-time": {
      "n_seeds": 4,
      "seeds": [
        1,
        2,
        3,
        4
      ],
      "xgb_lgbm": {
        "mean": -0.05272055583134955,
        "lo": -0.07785414914684102,
        "hi": -0.016101850093589783,
        "std": 0.02837057908846168,
        "values": [
          -0.04475019391921853,
          -0.07785414914684102,
          -0.016101850093589783,
          -0.07217603016574886
        ]
      },
      "lgbm_cat": {
        "mean": 0.8265339792486137,
        "lo": 0.8201584244364268,
        "hi": 0.8381202018283918,
        "std": 0.008081493610237111,
        "values": [
          0.8258609800658266,
          0.8219963106638094,
          0.8381202018283918,
          0.8201584244364268
        ]
      },
      "xgb_cat": {
        "mean": 0.3253732570599246,
        "lo": 0.30844444293736273,
        "hi": 0.34189431136912357,
        "std": 0.014343034633631685,
        "values": [
          0.33094159455280364,
          0.30844444293736273,
          0.34189431136912357,
          0.32021267938040854
        ]
      },
      "rule_predictions_under_bootstrap": [
        "match"
      ],
      "rule_robust": true,
      "rule_predicted": "match",
      "actual_outcome": "match",
      "classified_correctly": true
    },
    "delivery-eta": {
      "n_seeds": 0
    }
  },
  "qstat": "\npbs101: \n                                                                 Req'd  Req'd   Elap\nJob ID               Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n-------------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n13910555.pbs101      simerjit gdev     cf_fisd_t* 28457*   1  16  110gb 02:00 R 01:00\n13910557.pbs101      simerjit gdev     cf_fisd_t* 18240*   1  16  110gb 02:00 R 01:01\n13910558.pbs101      simerjit gdev     cf_fisd_t* 18257*   1  16  110gb 02:00 R 00:56\n13910559.pbs101      simerjit gdev     cf_fisd_t* 20685*   1  16  110gb 02:00 R 00:55\n13910566.pbs101      simerjit gdev     cf_fisd_t* 28512*   1  16  110gb 02:00 R 00:48\n13910567.pbs101      simerjit gdev     cf_fisd_t* 31663*   1  16  110gb 02:00 R 00:47\n13910568.pbs101      simerjit gdev     cf_fisd_t* 20786*   1  16  110gb 02:00 R 00:25\n13910569.pbs101      simerjit gdev     cf_fisd_t* 37317*   1  16  110gb 02:00 R 00:24\n13911135.pbs101      simerjit gdev     cf_fisd_r*    --    1  16  110gb 01:00 Q   -- \n13911136.pbs101      simerjit gdev     cf_fisd_r*    --    1  16  110gb 01:00 Q   -- \n13911137.pbs101      simerjit gdev     cf_fisd_r*    --    1  16  110gb 01:00 Q   -- \n13911138.pbs101      simerjit gdev     cf_fisd_r*    --    1  16  110gb 01:00 Q   -- \n13911139.pbs101      simerjit gdev     cf_fisd_r*    --    1  16  110gb 01:00 Q   -- \n"
}
```