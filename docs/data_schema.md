# Data schema

This repository uses normalized English-only tables.

## Ocular surface table

Required columns:

- `image_path`: image file path recorded in the spreadsheet
- `source`: site or cohort name in ASCII form
- `quality_label`: one of `poor`, `medium`, `good`
- `stage1_label`: one of `na`, `abnormal`, `normal`
- `stage3_label`: one of `na`, `referral`, `non_referral`

The ocular surface pipeline filters rows to:
- `quality_label in {medium, good}`
- `stage1_label in {abnormal, normal}`

Then it maps:
- `stage3_label == referral` -> positive class `1`
- all other stage-3 values -> negative class `0`

## Cataract table

Required columns:

- `image_name`: image file name relative to the image root directory
- `source`: site or cohort name in ASCII form
- `label`: binary class label (`0` or `1`)

## Recommended site-name mapping

If your original spreadsheet uses local-language site names, normalize them before use. Recommended examples:

- `hangzhou`
- `ningbo`
- `wenzhou`
- `wushi`
- `aksu`
- `baicheng`
- `kuche`
- `shaya`
- `xinhe`
