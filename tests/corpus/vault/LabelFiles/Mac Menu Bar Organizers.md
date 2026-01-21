# Mac Menu Bar Organizers

Bartender5 replacements (due to [this](https://www.macbartender.com/b5blog/Lets-Try-This-Again/)).

## System configs

You can adjust menu bar item spacing:

```bash
# Set spacing to 12 and padding to 8, someone else said this is a good tradeoff
defaults -currentHost write -globalDomain NSStatusItemSpacing -int 12
defaults -currentHost write -globalDomain NSStatusItemSelectionPadding -int 8
killall SystemUIServer
# Reset to defaults if it sucks
defaults -currentHost delete -globalDomain NSStatusItemSpacing
defaults -currentHost delete -globalDomain NSStatusItemSelectionPadding
killall SystemUIServer
```

There’s also **++[Menu Bar Spacing](https://sindresorhus.com/menu-bar-spacing)++**.

## Interesting Apps that replicate Bartender

### Ice

Basically a reimplementation of Bartender. Installed and using as of Aug 2025.

Open source

### SwiftBar

Another “BitBar” successor, supporting BitBar plugins. Use this if you want to mess around with config files.

<https://github.com/swiftbar/SwiftBar>

> Powerful macOS menu bar customization tool

> Add custom menu bar programs on macOS in three easy steps:
>
> - Write a shell script
>
> - Add it to SwiftBar
>
> - ... there is no 3rd step!

Open source, scriptable, plugins supported, lots of development as of 2025

`brew install swiftbar`

### 

## Interesting Apps that do something adjacent

### Badgeify ($)

[Badgeify menu bar companion](https://docs.badgeify.app/introduction)

One-time purchase

> - Add custom application icons
>
> - Create groups for related items
>
> - Set up contextual showing/hiding based on conditions

### Hidden Bar

Can hide menu bar icons.

<https://github.com/dwarvesf/hidden>

Free

### Vanilla

Like Hidden Bar but costs money

<https://matthewpalmer.net/vanilla/>

## Other Apps

There are a number of other options presented in this [matrix of Bartender5 replacements.](https://procrastopossum.com/bartender-alternatives/)

### Bartender5 ($)

One-time purchase per major version; they are releasing [Bartender6](https://macbartender.com/Bartender6/alpha.html) for Tahoe which will require a new license.

### Xbar

A “BitBar” successor

<https://github.com/matryer/xbar>

Open source, last release was 2021

## Discussions

- [Bartender Replacement — Six Bartender Alternatives to Manage Your Mac's Menu Bar - MacRumors](https://www.macrumors.com/2024/06/06/alternatives-bartender-mac-menu-bar/) June 2024

- [Bartender Replacement — Managing Your Mac Menu Bar: A Roundup of My Favorite Bartender Alternatives - MacStories](https://www.macstories.net/roundups/managing-your-mac-menu-bar-a-roundup-of-my-favorite-bartender-alternatives/) June 2024

- 
