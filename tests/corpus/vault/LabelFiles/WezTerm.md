---
tags:
  - experiment
  - software
---
# WezTerm

Alias: Using Wezterm (Experiment)

Related: [Computer Projects List.md](./Computer%20Projects%20List.md), [Vim!Nvim Learning.md](./Vim!Nvim%20Learning.md)

> WezTerm is a powerful cross-platform terminal emulator and multiplexer written by @wez and implemented in Rust

## Useful commands

*[Wezterm has a cli.](https://wezfurlong.org/wezterm/cli/cli/)*

Change the title of the current (or any) tab: `$> wezterm cli set-tab-title TITLE`

## Docs

[WezTerm configuration documentation](https://wezfurlong.org/wezterm/config/files.html)

<https://wezfurlong.org/wezterm/index.html>

<https://github.com/wez/wezterm>

## Todo

- I need to get copy-on-select working

   - Look into “[copy mode](https://wezfurlong.org/wezterm/copymode.html)” and “[quick select](https://wezfurlong.org/wezterm/quickselect.html)”

- Get a good font

   - <https://github.com/bbkane/dotfiles/blob/master/wezterm/dot-config/wezterm/font.lua>

   - <https://github.com/bbkane/dotfiles/blob/master/wezterm/README.md#install>

      ```
      brew tap homebrew/cask-fonts
      # Fonts I'm liking these days
      brew install font-hack
      brew install font-fira-code
      brew install font-ia-writer-mono
      ```

- zsh plugins

   - syntax highlighting 

   - autocomplete

- window resize

- change leader to caps lock? <https://youtu.be/V1X4WQTaxrc?si=ZPK98qAXWOg4-DxR>

## Usage

### Configuration

My config: `~/.config/wezterm/wezterm.lua`

Leader key

ctrl-b in tmux?

### Mouse

Default mouse behavior: <https://wezfurlong.org/wezterm/config/mouse.html#default-mouse-assignments>

| **Open new tab** | Click the `+` button in tab bar | `Cmd` + `t` | 
|---|---|---|
| **Open the [launcher menu](https://wezfurlong.org/wezterm/config/launch.html#the-launcher-menu)** | Right-click the `+` button in the tab bar |  | 

### Keyboard shortcuts

Default keyboard shortcuts: <https://wezfurlong.org/wezterm/config/default-keys.html>

| **Open the config file**  | `Cmd` + `,`  | Opens in new pane | 
|---|---|---|
|  | `wezv` | In `~/.zshrc` | 
| **Command palette** | `Ctrl` + `Shift` + `P` | *default* | 
| **[Debug panel](https://wezfurlong.org/wezterm/troubleshooting.html)** Lua REPL | `Ctrl` + `Shift` + `L` | *default* | 
| **Leader key** | `Ctrl` + `a` |  | 
| **Move forward word** | `Option` + `Right-Arrow` | *vim std* | 
| **Move backward word** | `Option` + `Left-Arrow` | *vim std* | 
| **Backspace word** | `CTRL` + `w`  | *unix std* | 
| **Quick Select mode** | `Ctrl` + `Shift` + `Space` | *default* | 
| **Insert emoji 🤖** | `CTRL` + `SHIFT` + `u` | *default* | 

### Split Panes

| **Horizontal split** | `Leader` + `”` | *tmux std* | 
|---|---|---|
| **Vertical split** | `Leader` + `%` | *tmux std* | 
| **Split pane navigation** | `Leader` + `{}-Arrow` |  | 

### Workspaces

Add new workspaces in `~/.config/wezterm/projects.lua` ; otherwise it will pick up anything in `~/Develop/`.

| **Workspace list** | `Leader` + `p` |  | 
|---|---|---|
| **Switch workspaces** | `Leader` + `f` | This is a filtered “launcher menu” | 
| **Abandon a workspace** | `Ctrl` + `d` | *default* | 

## Features

+ ### Launcher Menu

   > [The launcher menu](https://wezfurlong.org/wezterm/config/launch.html#the-launcher-menu) by default lists the various multiplexer domains and offers the option of connecting and spawning tabs/windows in those domains.

   My lua config binds `Leader + f` to a filtered launcher, showing the currently active multiplexer workspaces, a fuzzy filter on Workspaces:

   ```lua
   action = wezterm.action.ShowLauncherArgs { flags = 'FUZZY|WORKSPACES' },
   ```

   #### Modifying the launcher menu

   You can [activate a custom/filtered launcher menu programmatically](https://wezfurlong.org/wezterm/config/lua/keyassignment/ShowLauncherArgs.html):

   ```lua
   -- in the key mapping table:
   action = wezterm.action.ShowLauncherArgs { flags = 'FUZZY|TABS' },
   ```

   You can add items to the Launcher Menu programmatically:

   ```lua
   table.insert(launch_menu, {
     label = 'PowerShell',
     args = { 'powershell.exe', '-NoLogo' },
   })
   ```

   My launcher menu, no filters:

   ![ScreenShot 2024-08-26 at 15.06.19@2x.png](./WezTerm-assets/ScreenShot%202024-08-26%20at%2015.06.19@2x.png)

+ ### Workspaces

+ ### Quick Select

   <https://wezfurlong.org/wezterm/quickselect.html>

   `Shift + Ctrl + Space`

   > … the terminal is searched for items that match the patterns defined by the [quick_select_patterns](https://wezfurlong.org/wezterm/config/lua/config/quick_select_patterns.html) configuration combined with a default set of patterns that match things such as URL and path fragments, git hashes, ip addresses and numbers.

   > The bottom of the screen shows your input text along with a hint as to what to do next; typing in a highlighted prefix will cause that text to be selected and copied to the clipboard, and quick select mode will be cancelled.
   >
   > Typing in the uppercase form of the prefix will copy AND paste the highlighted text, and cancel quick select mod.

   

## Notes

### Set up

`brew install --cask wezterm`

> Unlike terminals where your settings are adjusted via the UI (iTerm 2), your WezTerm config lives in your dotfiles and is portable across all your machines.

> watches your config file, and when it changed it auto-reloaded instantly.

### Terminal multiplexer

> If you make use of a multiplexer (i.e. tmux) then you may consider using WezTerm’s builtin capabilities instead. They’ll generally give you a more integrated experience, with individual scrollback buffers per pane, better mouse control, easier selection functionality, and generally faster performance.

> each project to maintain its own multiplexer instance with its own windows, panes, and tabs. In tmux you might achieve this with different sessions. In WezTerm we’ll do it with [workspaces](https://wezfurlong.org/wezterm/recipes/workspaces.html).

### Sample config

> lua config: <https://gist.github.com/alexpls/83d7af23426c8928402d6d79e72f9401>

> <https://github.com/bbkane/dotfiles/tree/master/wezterm>

> <https://alexplescan.com/posts/2024/08/10/wezterm/>

## Discussions

<https://news.ycombinator.com/item?id=41223934>