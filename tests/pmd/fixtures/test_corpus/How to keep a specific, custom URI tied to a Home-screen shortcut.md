# How to keep a **specific, custom URI** tied to a Home-screen shortcut

For example, having a Home Screen shortcut (like a Safari saved site) that points to some custom URI.

| Method | How it works | What stays editable later? | Pros | Limits | 
|---|---|---|---|---|
| **1\. Shortcuts app – “Open URL” action** | \-  Open *Shortcuts* → **\+** → add **URL ➜ Open URLs**. |  |  |  | 

- Paste any HTTP/HTTPS link, deep-link or x-callback.  

- Share-sheet › **Add to Home Screen**. | You can reopen the shortcut and change the URL or its icon at any time. | -  Survives iOS updates.  

- Works with multiple links (ask-fornput, menus, etc.). | Launch briefly shows “Shortcuts” splash unless *Reduce Motion* is on. |
   | **2\. Shortcuts URL-scheme launcher (`shortcuts://run-shortcut?name=`)** | Put only a *Shortcuts* URL in **Add to Home Screen**—this icon triggers a named shortcut that in turn opens your real link. | Edit the underlying shortcut. The web-clip URL itself never changes. | -  Lets you hide complex logic; the on-screen icon holds a fixed, harmless URL. | Two-step hop is slightly slower than Method 1. |
   | **3\. Airplane-mode trick (stock Safari “Add to Home Screen”)** | -  Copy the exact address.  

- Enable Airplane Mode.  

- Paste URL in Safari and tap **Go**. Page fails to load; immediately pick **Share › Add to Home Screen**. | Not editable—must recreate to change. | -  Achieves a *pure* web-clip with the unaltered string (no canonical rewrite). | Relies on manual timing; still cannot be modified later. |
   | **4\. Progressive Web App manifest control** | If you own the site, set `start_url` in your `manifest.json`. Safari stores that value when visitors add the page to Home Screen. | Update the manifest → users who **re-add** get the new URL. | -  No user work-around needed; feels like a native app. | Only for sites you control; existing clips don’t update. |
   | **5\. MDM / configuration profile “Web Clip”** | An admin pushes a *Web Clip* payload specifying `URL`, title and icon. | Admin can resend a profile with a new URL; clip updates silently. | -  Central control over thousands of devices. | Requires Apple Business/School Manager or a Mac + Configurator; not for personal use. |

#### Choosing the right approach

- **Personal device, single link:** use **Shortcuts → Open URL**. It is one-tap to run and totally editable later.  

- **You maintain the website:** set `start_url` so every visitor who installs the clip gets the exact path or query string you need.  

- **Enterprise fleet:** deploy a Web Clip payload; it is the only Apple-supported way to update a shortcut remotely.  

- **Need a one-off clip without Shortcuts:** the airplane-mode trick forces Safari to record the literal URL instead of its canonical form.

None of these techniques lets you edit an already-saved standard web-clip; you must either rely on Shortcuts or recreate the icon.

Sources
\[1\] Custom web app icons : r/shortcuts - Reddit <https://www.reddit.com/r/shortcuts/comments/161qdsh/custom_web_app_icons/>
\[2\] Run a shortcut using a URL scheme on iPhone or iPad <https://support.apple.com/guide/shortcuts/run-a-shortcut-from-a-url-apd624386f42/ios>
\[3\] Shortcuts Automation URL Scheme - Reddit <https://www.reddit.com/r/shortcuts/comments/1fdkqaj/shortcuts_automation_url_scheme/>
\[4\] Search Native Apps from Apple Shortcuts with Custom URL Schemes <https://www.youtube.com/watch?v=OeRzBLzipbo>
\[5\] How to add url shortcut ln iOS homescreen for youtube or any website <https://community.brave.com/t/how-to-add-url-shortcut-ln-ios-homescreen-for-youtube-or-any-website/529217>
\[6\] URL Scheme shortcuts://create-shortcut?name ... - Stack Overflow <https://stackoverflow.com/questions/77196447/url-scheme-shortcuts-create-shortcutname-yourshortcutnameurl-your-url-encode>
\[7\] How to Quickly Open a Settings Page Using Shortcuts on iPhone and iPad <https://www.howtogeek.com/657555/how-to-quickly-open-a-settings-page-using-shortcuts-on-iphone-and-ipad/>
\[8\] Open and create a shortcut using a URL scheme on iPhone or iPad <https://support.apple.com/guide/shortcuts/open-create-and-run-a-shortcut-apda283236d7/ios>
\[9\] Create Website Shortcut | Support & Resources - Brandeis University <https://www.brandeis.edu/its/support/website-shortcut.html>
\[10\] Shortcut to open links in specific app? - Reddit <https://www.reddit.com/r/shortcuts/comments/1fvbiy4/shortcut_to_open_links_in_specific_app/>
\[11\] iOS Home Screen Shortcut URL <https://apple.stackexchange.com/questions/340709/ios-home-screen-shortcut-url>
\[12\] How do I configure a Shortcut to send URLs for Safari web pages? <https://www.reddit.com/r/shortcuts/comments/17qy22m/how_do_i_configure_a_shortcut_to_send_urls_for/>
\[13\] iOS shortcuts Open URL with deep link - Ask Different <https://apple.stackexchange.com/questions/435013/ios-shortcuts-open-url-with-deep-link>
\[14\] Adding specific links as home screen icon… - Apple Communities <https://discussions.apple.com/thread/252606725>
\[15\] Create a shortcut that will create url - Automators Talk <https://talk.automators.fm/t/create-a-shortcut-that-will-create-url/3821>
\[16\] Building 7 Shortcuts for App Icons, Menus, QR Codes, & HomeKit ... <https://www.youtube.com/watch?v=CNLo26EVUu8>
\[17\] Editing the four url shortcuts on chrome homepage IOS - Google Help <https://support.google.com/chrome/thread/68973352/editing-the-four-url-shortcuts-on-chrome-homepage-ios>
\[18\] How to add a shortcut icon to your iPhone that opens a web link <https://www.youtube.com/watch?v=q5a-YZYmV8c>
\[19\] iOS Custom URL Launcher - SharpTools Community <https://community.sharptools.io/t/ios-custom-url-launcher/11234>
\[20\] How to add a Custom URL to your iPhone Home Screen as an Icon <https://www.youtube.com/watch?v=-HpuqT-Hw7U>