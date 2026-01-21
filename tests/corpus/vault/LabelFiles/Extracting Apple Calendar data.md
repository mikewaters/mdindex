---
tags:
  - document 📑
---
# Extracting Apple Calendar data

**Reference:** [Calendar Data Import](https://app.raindrop.io/my/50100332) (raindrop)

### iCalBuddy

Has a cli, its kinda complicated though.

`brew install ical-buddy`

### iCalBuddy proxy

#### Using subprocess

[self-tracker](https://github.com/davidxmoody/self-tracking/blob/b16f8e76539fe5573fe758d684531f6ee4da1391/self_tracking/importers/atracker.py#L24) (github)

Exposes `get_events()` which loads iCal data into a pandas dataframe

#### Python module with Local+iCal integration

[Open Conference URL](https://github.com/caleb531/open-conference-url) Alfred workflow exposes a python module that could be easily reused, which uses iCalBuddy (if present, otherwise local calendar) to expose rich objects.


