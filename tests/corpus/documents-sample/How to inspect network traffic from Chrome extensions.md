# How to inspect network traffic from Chrome extensions

I have a third party chrome extension which sends some requests to a website and gets some data. I want to analyse network traffic for those requests. I tried using Chrome debugger, but that did not work. Using tools like wireshark might work but that is a lengthy process. Is there any chrome extension/other way to accomplish this?

asked Jun 4, 2018 at 4:23



1

You can monitor an extension in Google Chrome by:

1. Going to Settings -> Selecting the *Extensions* Section. OR go to `chrome://extensions/` from the Address bar.

2. Checking the **Developer Mode** tick box in the **top right corner** of the page, which will change the display to look like this:

3. Click on the link next to the extension’s `Inspect Views` text

4. A Developer Tools window will open up where you can monitor the extension by selecting the Network Tab at the top

answered Feb 23, 2019 at 23:57



[LeOn - Han Li](https://stackoverflow.com/users/1486742/leon-han-li)LeOn - Han Li

9,9382 gold badges67 silver badges62 bronze badges

For extensions that do not have `Inspect views` in `chrome://extensions/`:

- Right click on the button for the extension (next to the browser's address bar) and select `Inspect popup`

answered Jul 18, 2023 at 4:10



[u17](https://stackoverflow.com/users/385513/u17)u17

2,8045 gold badges31 silver badges43 bronze badges

You cannot see requests most probably because they are sent through a background service. Go to the Extensions Settings page and next to the extension you want to inspect, there should be something like `background.html`. Click on it, this is what you are looking for

answered Nov 6, 2023 at 14:38



[ofarukcaki](https://stackoverflow.com/users/7500203/ofarukcaki)ofarukcaki

1113 silver badges13 bronze badges

## 