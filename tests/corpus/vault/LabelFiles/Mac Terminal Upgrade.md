---
tags:
  - experiment
Status:
  - completed
---
# ~~Mac Terminal Upgrade~~

Is a:: Experiment

Child of:: [Computer Projects List.md](./Computer%20Projects%20List.md)

Status:: Complete

Tried the Warp terminal, but I do not like it. iTerm is showing its age, and it is annoying to configure.

## Plan

I am going to leave Warp and Oh-My-Zsh installs alone for now, and remove them from dotfiles. I’ve stuck some shims in places I wasnt sure of.

## Constraints

- GPU support

- Open source

### todo

- [x] Cancel/check for a Warp Terminal subscription

## Options

+ ### Warp Terminal

   Using this, but it requires a purchase and is closed source. I actualy am paying for it, I think.

+ [WezTerm.md](./WezTerm.md)

   WezTerm is a powerful cross-platform terminal emulator and multiplexer written by @wez and implemented in Rust

   [WezTerm configuration documentation](https://wezfurlong.org/wezterm/config/files.html)

   <https://wezfurlong.org/wezterm/index.html>

   <https://github.com/wez/wezterm>

   14\.9k github stars

+ Contour

   > contour is a modern and actually fast, modal, virtual terminal emulator, for everyday use. It is aiming for power users with a modern feature mindset.

   <http://contour-terminal.org>

   <https://github.com/contour-terminal/contour>

   2\.3k github stars

   \~150-00 commits/month in 2024

+ ### kitty

   The fast, feature-rich, GPU based terminal emulator

   <https://sw.kovidgoyal.net/kitty/>

   <https://github.com/kovidgoyal/kitty>

   22\.7k github stars

   Been around for a while.

+ ### Rio

   > tl;dr: Rio is a terminal built to run everywhere, as a native desktop applications by Rust or even in the browser powered by WebAssembly.

   <https://raphamorim.io/rio/>

   <https://github.com/raphamorim/rio>

   3\.2k github stars

## Resources

### Emoji Notes

> Mode 2027 is **++[a proposal for grapheme support in terminals](https://github.com/contour-terminal/terminal-unicode-core)++**. 

### Terminal Comparisons

[Terminal latency](https://danluu.com/term-latency/)

[Measuring terminal latency | Luke's Wild Website](https://www.lkhrs.com/blog/2022/07/terminal-latency/)

[A look at terminal emulators, part 2 \[LWN.net\]](https://lwn.net/Articles/751763/)

[Measured: Typing latency of Zutty (compared to others)](https://tomscii.sig7.se/2021/01/Typing-latency-of-Zutty)