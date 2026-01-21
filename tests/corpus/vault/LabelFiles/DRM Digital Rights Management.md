---
tags:
  - document 📑
---
# DRM Digital Rights Management

## Looking for DRM

### Check for FairPlay

`mdls -name kMDItemFairPlayProtected "your_audiobook.m4b"`

```shell
# a value of 1 is protected
kMDItemFairPlayProtected = (null)

```

`ffprobe -v quiet -print_format json -show_format -show_streams "your_audiobook.m4b"`



- `encryption_scheme`: If present, indicates encryption

- `codec_name`: Look for “aac_latm” which often indicates protected content

- `format_name`: “mov,mp4,m4a,3gp,3g2,mj2” is common for protected files

## Apps that remove DRM

- OpenAudible (audiobook)

- Libation (audiobook)

- Calibre (ebook)