---
tags:
  - document 📑
---
# FileVault Recovery Key

To regenerate: `sudo fdesetup changerecovery -personal`

Mine is stored in Enpass

Alternatively, I could disable and then enable to generate a new one, but it will have to decrypt (I think) which should take time.

```json
sudo fdesetup disable
sudo fdesetup enable
```