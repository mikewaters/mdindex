---
tags:
  - document 📑
Document Type:
  - Runbook
---
# Mike’s Computer Runbook

Operations cheat-sheet for my computers, describing the.

[Terminal File Managers.md](./Terminal%20File%20Managers.md)/Installed

## TODO

- [Use the Globe Key to Launch Raycast’s Emoji Picker](https://www.macstories.net/tutorials/three-tips-to-combine-bettertouchtool-and-raycast-for-simpler-keyboard-shortcuts/)

- [ ] \[HEPTA\] Centralize my runbooks into Heptabase

   - ~~Drafts~~

   - Roam

   - Local Markdown files on `beaker`

- `Opt-{<-,->]` EMPTY What is this??

## CLI Experiments

[Terminal File Managers.md](./Terminal%20File%20Managers.md)

Trying:

- `nnn` (n^3)

- `spf` (Superfile)

## Creating a runbook dashboard

- What services are installed? How are they run?

   - `brew services list`

      - jupyter kernels

   - `uv tool list `[uv (python).md](./uv%20\(python\).md)

      - mflux

- What custom scripts or commands am I using?

   - Raycast

   - BTT

- What keyboard shortcuts am I learning?

- What have I installed recently? Which apps am I testing?

- Which projects and project directories are active?

- 

## Structure of this|a Runbook

### Human Interfaces

How does one navigate the system, what customizations are applied etc

- input devices and their configurations

   - custom keyboard or mouse setup

- keyboard shortcuts, global and per-app 

- Built-in tools

   - Finder

- automation tools

   - BTT

   - Karabiner

   - KeyboardMaestro

   - Raycast

   - Hookmark

   - Apple Shortcuts



# User→Machine Interfaces (`beaker`)

Description of the manner in which my systems have been set up; what assistive systems have I provisioned to make computing easier. For example: assistive apps or tools, input device configurations

*command legend:* 

| `app/system default` | `reconfigured/customized, in-app` | `customized or created by me (via script etc)` | 
|---|---|---|

## Keyboard shortcuts

[Keyboard Shortcuts.md](./Keyboard%20Shortcuts.md)

## Custom scripts and commands

### *++Raycast commands++*

[Raycast Script Commands.md](./Raycast%20Script%20Commands.md)

#### Copy web page Title and URL (Raycast script)

I capture Safari page URLs with a few scripts:

- **Copy Markdown URL and Title**

- **Copy Plaintext URL and Title**

   - results in “`Some Title https://something`”

- **Copy Blockquote URL and Title**

   - Embeds a child-blockquote “under” some quote, to serve as a reference/source

   - A Markdown link is embedded

#### Raycast extension TBD

- ~~Hookmark~~

- Clickup

### Apple Shortcuts

### Shell commands

~~`hook `Hookmark’s cli~~

`alias` show aliases (builtin)

`whichif` show the iface thats been prioritized for egress traffic

`flushdns` flushes MacOS DNS cache

`nnetstat` lsof alias to replicate netstat on linux

`take` An implementation of the Oh-My-Zsh command that wraps `mkdir && cd`

#### Shell aliases

```bash
ee="exit"
lc='colorls -lA --sd --gs'
ll=
la=
tree='colorls -l --tree'
atree='colorls -lA --tree'
dev="cd ~/Develop"
zv="nvim ~/.zshrc". # edit zshell config
zr=". ~/.zshrc" # reload zshell config
```

## My Filesystem Layout

Quest: [My Directory Structures.md](./My%20Directory%20Structures.md)

# Systems Reference

## Keyboard Customization

I use VIA/QMK on keyboards that support it; for others I use Karabiner Elements.

For example, the Leopold FC750R doesnt support firmware editing, and also has fucky keys. 

## Keyboard Shortcuts

I use Better Touch Tool to manage my keyboard shortcuts. However I am considering using [Raycast Hotkeys](https://manual.raycast.com/command-aliases-and-hotkeys); it will depend on which one better supports migrating via dotfiles.

## Emoji

Raycast

## Clipboard Management

Raycast [clipboard history](https://www.raycast.com/core-features/clipboard-history) is mapped to the [Raycast alias](https://manual.raycast.com/command-aliases-and-hotkeys) `clip`

Other options are `pb{copy|paste}`, `osascript -e 'the clipboard as record' | xargs -n 2`, and the Finder clipboard editor.

Move to iphone runbook: on IOS, we have the [Pasteboard Manager](https://apps.apple.com/us/app/pasteboard-viewer/id1499215709)

- [ ] inshall pasteboard on my mac

article: <https://alexharri.com/blog/undefined/blog/clipboard> ([discussion](https://alexharri.com/blog/undefined/blog/clipboard))

## Terminal Configuration (`beaker`)

### My `.zshrc`

+ **Export secure credentials into the environment**

   `oai`

   ```shell
   ### OpenAI function to export my current API key from Keychain into
   # a probably-appropriate environment variable.
   function oai() {
     # Interestingly, Keychain on MacOS asks for my password **twice** here; the issue is - I think -
     # that the first is the Application accessing the keychain, and then the second is reading the key.
     export OPENAI_API_KEY=$(security find-generic-password -a "$USER" -s "OPENAI_API_KEY" -w)
   }
   #
   # ## end OAI
   ```

   Usage

   ```shell
   # inject openai creds into ComfyUI
   oai && comfy launch
   ```

### WezTerm

Experiment: [WezTerm.md](./WezTerm.md)

### McFly

<https://github.com/cantino/mcfly>

> Fly through your shell history. Great Scott!

Configured and initialized in `~/.zshrc`

Exposes cli tool `mcfly`

### Starship

> The cross-shell prompt for astronauts. ☄🌌️

Configured and initialized in `~/.zshrc`

Exposes cli tool `starship`

Config file is `/Users/mike/.config/starship.toml`

### Eza

> A modern alternative to ls\
> <https://eza.rocks>

Changed my `ll` replacement to use eza, and will alias ls in a way that only fucks me up in wezterm


