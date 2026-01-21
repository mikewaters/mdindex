---
tags:
  - experiment
---
# Proper Zooming with a Macropad

## Problem

For both Heptabase and Miro, I am unable to properly zoom using the mousewheel on my Macropad.

- [x] Configure my Macbook to be able to scroll in Heptabase

- [x] Configure my Macbook to be able to scroll in Miro

Further, the zoom I am able to figure out/hack together is not as good as the trackpad zoom.

- [ ] Configure my Macbook to replicate the touchpad scroll using a Macropad

## Solutions

### Miro basic scroll

I was able to make this work by defining the macropad behavior at a system level using Via. When scrolling, the macropad emits the keypad-minus and keypad-plus key codes.

<https://app.tana.inc?nodeid=_kETh3yViji->

### Heptabase basic scroll

Heptabase does not use the same key codes for scrolling as Miro, and so I had to override the behavior in BetterTouchTool. For the app Heptabase, translating `keypad-{-,+}` to `cmd-{up,down}`

### Touchpad experience

Hammerspoon can probably do this. Here is[ a hammerspoon lua script](https://gist.github.com/ebai101/cc5ff3ef39e00ff34e1bbb02b531c65d) that does **the opposite**.