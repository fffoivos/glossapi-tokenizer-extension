# Goal-eval — whole-doc windows: LINE-level total bibliography recall/precision

Recall = fraction of Opus-marked bibliography lines the detector flagged. Body windows reveal end-of-chapter / mid-doc bibliographies the end-matter detector misses.

| source | window | bib-recall | precision | opus-bib-lines |
|---|---|---:|---:|---:|
| greek_phd | ALL | 0.80 | 0.54 | 8097 |
| greek_phd | body | 0.57 | 0.45 | 1089 |
| greek_phd | tail | 0.84 | 0.55 | 7008 |
| openarchives | ALL | 0.58 | 0.26 | 3149 |
| openarchives | body | 0.31 | 0.13 | 813 |
| openarchives | tail | 0.67 | 0.31 | 2336 |

## By window type (both sources)
| window | bib-recall | precision |
|---|---:|---:|
| tail | 0.80 | 0.48 |
| body | 0.46 | 0.26 |

**Read:** high tail-recall = end-matter lists caught; low body-recall = end-of-chapter / mid-doc bibliography the detector structurally misses (the goal-vs-component gap).