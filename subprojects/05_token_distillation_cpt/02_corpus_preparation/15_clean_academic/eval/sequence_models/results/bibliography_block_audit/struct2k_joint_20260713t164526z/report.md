# Silver bibliography block audit

A document's score is the number of continuous emitted `BIB`-label runs.
The audit does not run a classifier and does not reinterpret bibliography
headers; it measures the existing silver section labels exactly.

## Distribution

| Metric | Value |
|---|---:|
| Documents | 1,392 |
| Mean BIB blocks/document | 1.435 |
| Median | 1.0 |
| 90th percentile | 3 |
| 95th percentile | 5 |
| 99th percentile | 9 |
| Maximum | 19 |
| No BIB block | 160 |
| Exactly one BIB block | 973 |
| Multiple BIB blocks | 259 |

## Highest scores

| Rank | Document | Source | Split | Coverage | Blocks | BIB lines |
|---:|---|---|---|---|---:|---:|
| 1 | `9ea3e3138b74a83ec131052029fc045b567cb1ef0ab867471e700a400b571ccd` | greek_phd | train | annotated_windows | 19 | 207 |
| 2 | `bdae35f0737e73bf5d7f64f36fcfeb636101d20814935b06d3cd98660d63eede` | kallipos | train | annotated_windows | 18 | 60 |
| 3 | `edbb74f5988af45550f57a331e61381df85c78b90e5759dffdaeb742422a64b3` | kallipos | train | full_document | 16 | 21 |
| 4 | `415f8322a0ba6105a16ece2e6562c7ca256adf757ed61a8eb222e9f8345a4a8e` | greek_phd | train | annotated_windows | 15 | 228 |
| 5 | `5dc840b35610f3980c30742d57b9f53506d11f0f27e8c32ea8f18b865dd1388d` | openarchives | train | full_document | 13 | 231 |
| 6 | `7dc44e348122b1efe8f07550505ac9ac0f53819a6ca5f10f4051c58fcd4c7ead` | kallipos | train | annotated_windows | 13 | 192 |
| 7 | `cfefc9244f76f2e5da54638dd53679e0ae4ee0f383790c7f5011148e95deff22` | openarchives | train | full_document | 13 | 121 |
| 8 | `d60ed9b41196bee554978e6728d9113126d616a0e503410811e52584f5da37e6` | greek_phd | train | full_document | 12 | 157 |
| 9 | `fcfbd070651734312374b09b63aec1190450422179f1206d81d2e5741679464c` | kallipos | train | annotated_windows | 12 | 70 |
| 10 | `3ceab08822d91194a58959c11baf18a6d688ee625ff2e70fe00452db7188aedd` | kallipos | train | annotated_windows | 10 | 245 |
| 11 | `af693b43b50c273b5ef1296eb3c27808aa78a7d2e4e75647a455bcc674acf4c7` | kallipos | train | annotated_windows | 10 | 136 |
| 12 | `d01239a2d4847c0692f0806652be012c8654c62acac319a722d6ab424a9d4e9f` | kallipos | validation | annotated_windows | 10 | 106 |
| 13 | `0587a44e1a2008bd4dd92baabfe43c7bcde57244b3df06aa8b39d18a9fd0559c` | kallipos | validation | annotated_windows | 10 | 19 |
| 14 | `336cbeddc1c1eb2bd08bc443b1266424205666c9fcbf52729070ac2417fac9b6` | openarchives | train | annotated_windows | 9 | 282 |
| 15 | `cca838e7d044aa54790d1d5e6b5c1ed7b31ff7d8c9cc32af9b97a78e7d0fde82` | greek_phd | train | annotated_windows | 9 | 196 |
| 16 | `590ea1074b09c9bbf7718f0ffb6bf596458591f2842215049265476a2337d873` | kallipos | train | annotated_windows | 9 | 115 |
| 17 | `dafa8d1ef11c1c45fd1fbc1a30ff82773f2ac21b2c52dbeb0f514cc72e126796` | openarchives | train | full_document | 9 | 85 |
| 18 | `2aa2b9bbb63fae43318b976f38e0f4230250295e49c4f62eb71937b71d664c0e` | kallipos | train | full_document | 9 | 67 |
| 19 | `e4dca959e7eb0ce223314581a1b83c0d812cf1ee2c32db6fc73d28bfdbcb21bc` | kallipos | train | annotated_windows | 8 | 375 |
| 20 | `09a90c16ac477b6712d1a313a3b1e247efe2371df3676ade382fe8e6939c4a3c` | greek_phd | train | annotated_windows | 8 | 344 |
| 21 | `9852f819e8d17f446807284656f7d9eee604dcbb9a55a9457ea2c1c2a62ecee0` | greek_phd | train | annotated_windows | 8 | 307 |
| 22 | `9fad184dbb08a4ef1f25ddfcdba19eca76976c6cbc311486dfaa336eda97acf3` | kallipos | train | annotated_windows | 8 | 219 |
| 23 | `e0dc07c6febae59e4d34c27a5200b67c00307892e5788ee4ce68ef11a4de2a2e` | greek_phd | validation | annotated_windows | 8 | 183 |
| 24 | `17ccb7cebb1b931de1b3a0ba2877e29a0ce0f64142cbabae3faf8222f8a1219b` | greek_phd | train | full_document | 8 | 152 |
| 25 | `6186c2201b45e2d21d754f90c1e99c41e447236de98a2880844f8eaa2ab4577f` | greek_phd | train | annotated_windows | 8 | 116 |

## Preserved outputs

- `documents.jsonl`: every document, exact block count, and block spans
- `documents.csv`: flattened per-document review table
- `summary.json`: distribution, group summaries, and ranked tail
- `distribution.svg`: histogram and highest-scoring documents
- `receipt.json`: hashes and run provenance
