---
tags:
  - document 📑
---
# Automatically Sync OpenAudible books

Download and de-DRM your Audible library automatically.

### 1\. Open OpenAudible at startup

Add OpenAudible to the `Settings → Login Items & Extensions → Open at Login`

### 2\. Keep OpenAudible running

I just minimize it and ignore it

### 3\. Set Auto-Download

A hidden option n `settings.json` (per [this thread](https://github.com/openaudible/openaudible/discussions/1081?t)):

```json
{
  ...

  "autoRefreshMinutes": 120,
  ...
}
```

The current settings path can be found in the app (`Settings Gear → Preferences Directory`), on my Mac it defaults to `/Users/[me]/Library/OpenAudible/settings.json`.

1. Close the app (if you don’t, it will overwrite it)

2. Adjust the setting from 0 (the default) and save

3. Open the app

Now, with OpenAudible running 24/7 it will download per the cadence (every two hours in the example).

> NOTE: DO NOT use 120 minutes. I am getting blocked/limited