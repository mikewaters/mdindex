---
tags:
  - document 📑
---
# Extracting Apple Health data

### Using an exported zip file

Example: [self-tracking](https://github.com/davidxmoody/self-tracking/tree/b16f8e76539fe5573fe758d684531f6ee4da1391) (github)

[source](https://github.com/davidxmoody/self-tracking/blob/b16f8e76539fe5573fe758d684531f6ee4da1391/self_tracking/importers/applehealth.py)

```json
def getroot() -> ET.Element:
    fp = sorted(glob(expandvars("$HOME/Downloads/????-??-??-apple-health.zip")))[-1]
    with ZipFile(fp) as zf:
        return ET.parse(zf.open("apple_health_export/export.xml")).getroot()
```