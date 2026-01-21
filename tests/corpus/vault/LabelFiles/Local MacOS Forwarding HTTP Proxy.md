---
tags:
  - experiment
---
# Local MacOS Forwarding HTTP Proxy

## AKA Hacking HTTP URLs on the Mac for Heptabase, Tana, etc

## Problem

I have a number of apps installed on my Macbook that register URL Handlers for in-app links, having non-standard protocol specifiers. However, the knowledge-management apps that I use only support the standard protocols in embedded links.  Hence, I cannot refer to an in-app link on my system from within the KM apps (like Heptabase).

- [ ] Set up a Hammerspoon config to perform URL rewriting for Drafts

## Approach

I’d like to run an http listener that can rewrite HTTP URLs on the fly, using an heuristic to detect those URLs that represent in-app links, rewrite their protocol specifiers, and redirect them. Once redirected, MacOS will take over and route the protocols to the registered apps.

Pseudocode:

```lua
fakeurl = "http://drafts.redirect/<$paths-and-params-for-drafts>"
if fakeurl.tld == ".redirect":
    proto = fakeurl.host
    path = fakeurl.path
    RedirectTo("{proto}://{path}")
else
    PassThrough(fakeurl)
end if
```

## Solutions

### Hammerspoon

Hammerspoon can intercept URLs like this, using [hs.urlevent](https://www.hammerspoon.org/docs/hs.urlevent.html), and there is [a published “Spoon”](https://www.hammerspoon.org/Spoons/URLDispatcher.html) designed to do almost exactly what I want.

Example code ([source](https://zzamboni.org/post/my-hammerspoon-configuration-with-commentary/)):

```lua
function appID(app)
  return hs.application.infoForBundlePath(app)['CFBundleIdentifier']
end
chromeBrowser = appID('/Applications/Google Chrome.app')
edgeBrowser = appID('/Applications/Microsoft Edge.app')
braveBrowser = appID('/Applications/Brave Browser Dev.app')

DefaultBrowser = braveBrowser
WorkBrowser = edgeBrowser

JiraApp = appID('~/Applications/Epichrome SSBs/Jira.app')
WikiApp = appID('~/Applications/Epichrome SSBs/Wiki.app')
OpsGenieApp = WorkBrowser

Install:andUse("URLDispatcher",
               {
                 config = {
                   url_patterns = {
                     { "https?://jira%.work%.com",      JiraApp },
                     { "https?://wiki%.work%.com",      WikiApp },
                     { "https?://app.*%.opsgenie%.com", OpsGenieApp },
                     { "msteams:",                      "com.microsoft.teams" },
                     { "https?://.*%.work%.com",        WorkBrowser }
                   },
                   url_redir_decoders = {
                     -- Send MS Teams URLs directly to the app
                     { "MS Teams URLs",
                       "(https://teams.microsoft.com.*)", "msteams:%1", true },
                     -- Preview incorrectly encodes the anchor
                     -- character in URLs as %23, we fix it
                     { "Fix broken Preview anchor URLs",
                       "%%23", "#", false, "Preview" },
                   },
                   default_handler = DefaultBrowser
                 },
                 start = true,
                 -- Enable debug logging if you get unexpected behavior
                 -- loglevel = 'debug'
               }
)
```

In order to use Hammerspoon in this way, I will need to configure an `init.lua` in `~/.hammerspoon/`; here is an example in github: <https://github.com/zzamboni/dot-hammerspoon/blob/master/init.lua> (from the author of the above code).

#### References:

[Hammerspoon “Getting Started”](https://www.hammerspoon.org/go/)

[hammerspoon dotfiles](https://github.com/zzamboni/dot-hammerspoon/tree/master)

[My Hammerspoon Configuration, With Commentary](https://zzamboni.org/post/my-hammerspoon-configuration-with-commentary/)