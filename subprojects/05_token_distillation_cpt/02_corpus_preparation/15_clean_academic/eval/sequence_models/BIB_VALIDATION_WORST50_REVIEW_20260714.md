# Bibliography validation worst-50 extraction review — 2026-07-14

This is an outcome-directed diagnostic review, not a prediction-blind quality
sample.  Documents were ranked by missed LLM-silver BIB tokens from the frozen
validation prediction.  Extraction usability was then judged from text only.
For each document, the beginning, 25%, middle, 75%, end, and the
first/middle/last missed-BIB regions were inspected.  A document was excluded
only when the extracted text itself was unusable; poor classifier recall was
not an exclusion reason.

Result: 44 keep and 6 exclude among these 50.  Five exclusions confirm prior
prediction-blind decisions.  Rank 3 (`eaf30b21c052…`) is the one new exclusion.
The seventh exclusion in the complete validation decision file (`485835e73336…`)
was found by the earlier all-document blind screen and is outside this top 50.

| Rank | Missed BIB tokens | Document | Decision |
|---:|---:|---|---|
| 1 | 37,385 | `6a3b5b03d5974ba5a5247de89eec9bc819a3fc10c8a71ecd1a76e209f3ab6107` | exclude — one-word fragmentation |
| 2 | 31,365 | `806df80cc19a9c94574bdcbdd80a4709d7e6eed4f297c8ed8d602c1f773bdf54` | keep |
| 3 | 25,638 | `eaf30b21c0521c0ee56830a20d4eecb5ff08c1a5a2d5626c5455e33549b571d5` | exclude — central/BIB regions garbled |
| 4 | 22,236 | `1a8c752862ba58560883cae3fd1718da677b3217c930dd3c6698fd443c6cf49e` | keep |
| 5 | 20,239 | `182b139ccd8e9c9cb12672b86b464402174ae2a06b95664edc496a74d7252e8f` | exclude — character-corrupted OCR |
| 6 | 17,804 | `5644b4ed18fb424453f34c12d072abb098ccf2b5cce6b333cf7e4144116a4d59` | keep |
| 7 | 15,404 | `d2b13ed8f8e5e659717bd8b40f9775bcb4bd93c695d33bb9b6056007fc32f071` | keep |
| 8 | 13,236 | `d2beb0b78e45436082adba863e5cee8538d9d2b44e546ba59779777844c2698b` | exclude — unresolved GLYPH placeholders |
| 9 | 9,702 | `411066ba264123854cb297d9508db20730a655148fed5e38cdcbbbb33cb691dd` | keep |
| 10 | 9,088 | `861a90305f7aca08cea8e9fbc25f3bb99bbc456416c5026bc5a6804f5a4552fb` | keep |
| 11 | 8,266 | `74181e81c22e28cb628e95c91b1ffac456a3f4481de7ca5a12dd803dfc1fdb22` | keep |
| 12 | 7,646 | `0b4aa3cb1467beb18fc42e84acb260dd0fb553b6af5a3b770aade3f8207c6e16` | keep |
| 13 | 6,861 | `c94f241613ecbb42eb312a8548d7b4b82c24771b4e990dafe00190c3698a3410` | keep |
| 14 | 5,551 | `91762d9e472bcf399f7e83154b23508527df7260a0580690b54b0e3a85c5b138` | keep |
| 15 | 4,729 | `659d2485ffbe356c713b80372acd0eab68c44dcd68bb062aa4ba2874b033663b` | keep |
| 16 | 4,442 | `c75bd63217b14fa34e27065a0ae2e6307cb0d5f4047a71bae86b40bb542f13e1` | keep |
| 17 | 4,220 | `99b8f79ce5fb1afe56eb10954650c27f32f60300055242b3560c273d635aef89` | keep |
| 18 | 3,940 | `7e85e1e5a9f0907bea174746a5bffae739d6f75c2fce38273c85eba12ec8b976` | keep |
| 19 | 3,766 | `7b74f3b9918386c340a726651a170ce5b1b80c25b2ca15a1de97bb2c95d97900` | keep |
| 20 | 3,605 | `a8717fc4df54caf762fd659d4e8be1ec17128cce833d737998360781b57df155` | keep |
| 21 | 3,438 | `b398ec04d232b5aed7422e7d4d88b80a4ef698c2c550533b3081c8294f5fd385` | keep |
| 22 | 3,410 | `accb609600359d5f958d88c3eadbd5cc4da41779a9378df238b1a93d67b73443` | keep |
| 23 | 3,318 | `268a8325ff7aae53821d7f34fd3c3ec4e3c69a06a743fe9fa46052dbe3ddec9b` | keep |
| 24 | 3,313 | `b62b94ad5405868aff370bc342d7cc1c885e113174e76495ae269795e853360e` | keep |
| 25 | 3,233 | `88514d131d564f7f3a086bcf7f22300ff9541bc2d5ec37ab7766bcf4bced928b` | keep |
| 26 | 3,190 | `4b9e9eef02870ad6a309d84a8b0d3f4490dbe78389a9a5dfeff250dec985bd0f` | keep |
| 27 | 3,157 | `cd843744bd7b4e2d9346dfeab5905ffaa59aa2c86892d764f9037b8c3519c046` | keep |
| 28 | 3,067 | `5da3c49f530c188c01e64873624e8c5ccb00d9a6174a5fbeab4039d7b7aa78bf` | keep |
| 29 | 3,016 | `5f18b6a35963d0bad5b3b3c0b568d0d835ea8619ad2a5ebe5bf992b7b9fbb62a` | keep |
| 30 | 2,892 | `2ce6f4062432a423eaa65b624ae1b46010263332868546c17306ee444b4aee1a` | exclude — one-word fragmentation |
| 31 | 2,818 | `b5b217f0807bf5a7d25c22f7b8884943d6254fa2594616fb62c2d3a4c548f940` | keep |
| 32 | 2,749 | `7837c2127a124d18733364c3964d779c2816a9a80aad3f4ccd2cb259ad903b94` | keep |
| 33 | 2,670 | `1328064816dfebbdfcb89cdc07fd99b8c9505849c5844cf87bed48f7ef3d699e` | keep |
| 34 | 2,624 | `69c5ecfcbaa90b20dba038c7bcb37e5992884e1226d02a592420d6deb819a23b` | keep |
| 35 | 2,560 | `1a788fb354f7171e00ba9834476de0f4f6d3475f2802a1a17a5699b8b0be2ae6` | keep |
| 36 | 2,326 | `5b658d86a1d90e86c637f6c34227f3a6c5c3f34050406e9ee8fc4476c79804f4` | keep |
| 37 | 2,255 | `8855481b3f26ca9d11d263b1f77fbdae9fe2d42cfdd1ef12c819eda78bd73fa5` | keep |
| 38 | 2,152 | `48b5ecb03617cb0de9dbe4a46cf3e6c61cc750e379a012f4c16bfac8c1f2a318` | keep |
| 39 | 2,029 | `da9a0de787059308f5dd0a38870435e33405320c11f6223bc17af04bedea9021` | keep |
| 40 | 1,869 | `4a588842e2037c1b2f645f9d711a29f908b884fe54d9ebd8837d663edd2f6e04` | exclude — one-word fragmentation |
| 41 | 1,799 | `5f3c38e1881e7ce3016409cf03edd5711a5b12e216e1d05e847f334008ea6621` | keep |
| 42 | 1,731 | `ce489c2a18e4fb5d81df16c17dab771fb94ef014ae6928f9435b2a596c33908b` | keep |
| 43 | 1,729 | `6d7a95490da6309e1fa93b39dd6ab32ceb5b94ce213090a48cb230589f7ae46d` | keep |
| 44 | 1,697 | `03f2443274108105e3483d6a5871896e7177f51fb213f4282c43963387e63f63` | keep |
| 45 | 1,661 | `22aa47b2590586084e49664c9cdf53b874e0d63ada426e8371e9b0299e90aad4` | keep |
| 46 | 1,637 | `c87ec0e4eb3f04eaa25487875a675b15c636454e6cb18296b470f9ac17285552` | keep |
| 47 | 1,563 | `9201577a711d91aed1cbf1dcfe1168d7005f8d73cf3f9327ec48f03fdb93eaf2` | keep |
| 48 | 1,539 | `6c333f852a553d9cb58e3acb0997ea54d41233cea13cdb71fbd236ceb585ed63` | keep |
| 49 | 1,486 | `1feca59114f4b2f51498237e0805413cf8ad3d57c13d22f4e5b990e9d7fedad7` | keep |
| 50 | 1,324 | `e0dc07c6febae59e4d34c27a5200b67c00307892e5788ee4ce68ef11a4de2a2e` | keep |

The follow-up-qualified 267-document metrics are documented as a diagnostic
sensitivity analysis in `BIB_ENTRY_LADDER_RUN_20260713.md`.  They must not
replace the frozen 274-document result or the independently qualified
268-document report when making leakage-sensitive claims.
