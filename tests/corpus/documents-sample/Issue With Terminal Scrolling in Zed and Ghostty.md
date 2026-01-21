# Issue With Terminal Scrolling in Zed and Ghostty

Oct 6, 2025

## THE FIXXXXXXXX

`tput rmcup`

## The problem

After using Mcfly, or pudb (not sure which), my whole damn Mac starts scrolling through history instead of scrolling through text.

## Alternate Scrolling setting in Terminal.app

<https://github.com/bpython/bpython/issues/517>

I’ve tried to change the setting in Terminal.app (it *was* enabled, but it didnt fix anything

## Cmd-R toggles this in macOS

> At least in El Capitan and probably also in Yosemite you can move from scrolling the history to scrolling the window with cmd R

Issue is, Zed overrides this key combo, and I probably did some shit in the system config as well (in one of them, at least)

## This is related to some DECSET command

### Per the Zed docs

<https://zed.dev/docs/configuring-zed#terminal-alternate-scroll>

> Terminal: Alternate Scroll
>
> Description: Set whether Alternate Scroll mode (DECSET code: ?1007) is active by default. Alternate Scroll mode converts mouse scroll events into up / down key presses when in the alternate screen (e.g. when running applications like vim or less). The terminal can still set and unset this mode with ANSI escape codes.
>
> Setting: alternate_scroll
>
> Default: off

### Nothing in Ghostty

They have a ton of docs about terminal control codes, but nothing about DECSET thats obvious. They also do not have a setting for Alternate Scroll like Zed does.

<https://ghostty.org/docs/vt/csi/su>

### Similar thing in Superuser.com

`printf "\e[?1004l"` was recommended for this other thing. When I do this with `1007`, shit goes haywire. I tried another few commands, but scrolling started creating jumbled crap.

<https://superuser.com/questions/931873/o-and-i-appearing-on-iterm2-when-focus-lost>

### Relevant docs for iTerm2

> DECSET(1007): Alternate scroll
>
> When enabled and when the terminal is in alternate screen mode, the scroll wheel causes the terminal emulator to send cursor up and cursor down keys.
>
> Precedence
>
> DECSET(1000), DECSET(1002), and DECSET(1003) take precedence over DECSET(1007) if both 1007 and one of the other modes is enabled at the same time.
>
> If more than one of 1000, 1002, and 1003 are on, the most recently enabled mode takes precedence and automatically disables the previous mode. For example, this sequence disables mouse reporting: DECSET(1000), DECSET(1003), DECRST(1003).

<https://iterm2.com/feature-reporting/>

## A fix

“Disable mouse scrolling through terminal command history on Mac terminal”

<https://unix.stackexchange.com/questions/511740/disable-mouse-scrolling-through-terminal-command-history-on-mac-terminal>

> Run this command:
>
> ```
> $ tput rmcup
> 
> ```
>
> What happened most likely is that you were, either locally or remotely, running a command (like `vim`, or `top`, or many programs that use libraries similar to `ncurses`) that uses the terminal's "alternate screen" mode. When this is active, many terminal programs helpfully remap the scrolling action on the mouse to arrow keys, because generally scrolling the local display is less than helpful. If this application terminated ungracefully, your terminal may still think it's in that mode.
>
> This command resets this, and should re-enable your ability to scroll.