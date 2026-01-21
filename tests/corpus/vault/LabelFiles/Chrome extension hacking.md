---
tags:
  - document 📑
---
# Chrome extension hacking

## Message passing

Because content scripts run in the context of a web page, not the extension that runs them, they often need ways to communicate with the rest of the extension. For example, an RSS reader extension might use content scripts to detect the presence of an RSS feed on a page, then notify the service worker to display an action icon for that page.

This communication uses message passing, which allows both extensions and content scripts to listen for each other's messages and respond on the same channel. A message can contain any valid JSON object (null, boolean, number, string, array, or object). There are two message passing APIs: one for [one-time requests](#simple), and a more complex one for [long-lived connections](#connect) that allow multiple messages to be sent. For information about sending messages between extensions, see the [cross-extension messages](#external) section.

## One-time requests

To send a single message to another part of your extension, and optionally get a response, call `[runtime.sendMessage()](https://developer.chrome.com/docs/extensions/reference/api/runtime#method-sendMessage)` or `[tabs.sendMessage()](https://developer.chrome.com/docs/extensions/reference/api/tabs#method-sendMessage)`. These methods let you send a one-time JSON-serializable message from a content script to the extension, or from the extension to a content script. To handle the response, use the returned promise. For backward compatibility with older extensions, you can instead pass a callback as the last argument. You can't use a promise and a callback in the same call.

For information on converting callbacks to promises and for using them in extensions, see [the Manifest V3 migration guide](https://developer.chrome.com/docs/extensions/develop/migrate/api-calls#replace-callbacks).

Sending a request from a content script looks like this:

content-script.js:

```
(async () => {
  const response = await chrome.runtime.sendMessage({greeting: "hello"});
  // do something with response here, not outside the function
  console.log(response);
})();

```

If you want to respond synchronously to a message, just call `sendResponse` once you have the response, and return `false` to indicate it's done. To respond asynchronously, return `true` to keep the `sendResponse` callback active until you are ready to use it. Async functions are not supported because they return a Promise, which is not supported.

To send a request to a content script, specify which tab the request applies to as shown in the following. This example works in service workers, popups, and chrome-extension:// pages opened as a tab.

```
(async () => {
  const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
  const response = await chrome.tabs.sendMessage(tab.id, {greeting: "hello"});
  // do something with response here, not outside the function
  console.log(response);
})();

```

To receive the message, set up a `[runtime.onMessage](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onMessage)` event listener. These use the same code in both extensions and content scripts:

content-script.js or service-worker.js:

```
chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    console.log(sender.tab ?
                "from a content script:" + sender.tab.url :
                "from the extension");
    if (request.greeting === "hello")
      sendResponse({farewell: "goodbye"});
  }
);

```

In the previous example, `sendResponse()` was called synchronously. To use `sendResponse()` asynchronously, add `return true;` to the `onMessage` event handler.

If multiple pages are listening for `onMessage` events, only the first to call `sendResponse()` for a particular event will succeed in sending the response. All other responses to that event will be ignored.

## Long-lived connections

To create a reusable long-lived message passing channel, call `[runtime.connect()](https://developer.chrome.com/docs/extensions/reference/api/runtime#method-connect)` to pass messages from a content script to an extension page, or `[tabs.connect()](https://developer.chrome.com/docs/extensions/reference/api/tabs#method-connect)` to pass messages from an extension page to a content script. You can name your channel to distinguish between different types of connections.

One potential use case for a long-lived connection is an automatic form-filling extension. The content script might open a channel to the extension page for a specific login, and send a message to the extension for each input element on the page to request the form data to fill in. The shared connection allows the extension to share state between extension components.

When establishing a connection, each end is assigned a `[runtime.Port](https://developer.chrome.com/docs/extensions/reference/api/runtime#type-Port)` object for sending and receiving messages through that connection.

Use the following code to open a channel from a content script, and send and listen for messages:

content-script.js:

```
var port = chrome.runtime.connect({name: "knockknock"});
port.postMessage({joke: "Knock knock"});
port.onMessage.addListener(function(msg) {
  if (msg.question === "Who's there?")
    port.postMessage({answer: "Madame"});
  else if (msg.question === "Madame who?")
    port.postMessage({answer: "Madame... Bovary"});
});

```

To send a request from the extension to a content script, replace the call to `runtime.connect()` in the previous example with `[tabs.connect()](https://developer.chrome.com/docs/extensions/reference/api/tabs#method-connect)`.

To handle incoming connections for either a content script or an extension page, set up a `[runtime.onConnect](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onConnect)` event listener. When another part of your extension calls `connect()`, it activates this event and the `[runtime.Port](https://developer.chrome.com/docs/extensions/reference/api/runtime#type-Port)` object. The code for responding to incoming connections looks like this:

service-worker.js:

```
chrome.runtime.onConnect.addListener(function(port) {
  console.assert(port.name === "knockknock");
  port.onMessage.addListener(function(msg) {
    if (msg.joke === "Knock knock")
      port.postMessage({question: "Who's there?"});
    else if (msg.answer === "Madame")
      port.postMessage({question: "Madame who?"});
    else if (msg.answer === "Madame... Bovary")
      port.postMessage({question: "I don't get it."});
  });
});

```

### Port lifetime

Ports are designed as a two-way communication method between different parts of the extension. A top-level frame is the smallest part of an extension that can use a port. When part of an extension calls `[tabs.connect()](https://developer.chrome.com/docs/extensions/reference/api/tabs#method-connect)`, `[runtime.connect()](https://developer.chrome.com/docs/extensions/reference/api/runtime#method-connect)` or `[runtime.connectNative()](https://developer.chrome.com/docs/extensions/reference/api/runtime#method-connectNative)`, it creates a [Port](https://developer.chrome.com/docs/extensions/reference/api/runtime#type-Port) that can immediately send messages using `[postMessage()](https://developer.chrome.com/docs/extensions/reference/api/runtime#property-Port-postMessage)`.

If there are multiple frames in a tab, calling `[tabs.connect()](https://developer.chrome.com/docs/extensions/reference/api/tabs#method-connect)` invokes the `[runtime.onConnect](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onConnect)` event once for each frame in the tab. Similarly, if `[runtime.connect()](https://developer.chrome.com/docs/extensions/reference/api/runtime#method-connect)` is called, then the `onConnect` event can fire once for every frame in the extension process.

You might want to find out when a connection is closed, for example if you're maintaining separate states for each open port. To do this, listen to the `[runtime.Port.onDisconnect](https://developer.chrome.com/docs/extensions/api/reference/runtime#property-Port-onDisconnect)` event. This event fires when there are no valid ports at the other end of the channel, which can have any of the following causes:

- There are no listeners for `[runtime.onConnect](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onConnect)` at the other end.

- The tab containing the port is unloaded (for example, if the tab is navigated).

- The frame where `connect()` was called has unloaded.

- All frames that received the port (via `[runtime.onConnect](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onConnect)`) have unloaded.

- `[runtime.Port.disconnect()](https://developer.chrome.com/docs/extensions/reference/api/runtime#property-Port-disconnect)` is called by *the other end*. If a `connect()` call results in multiple ports at the receiver's end, and `disconnect()` is called on any of these ports, then the `onDisconnect` event only fires at the sending port, not at the other ports.

## Cross-extension messaging

In addition to sending messages between different components in your extension, you can use the messaging API to communicate with other extensions. This lets you expose a public API for other extensions to use.

To listen for incoming requests and connections from other extensions, use the `[runtime.onMessageExternal](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onMessageExternal)` or `[runtime.onConnectExternal](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onConnectExternal)` methods. Here's an example of each:

service-worker.js

```
// For a single request:
chrome.runtime.onMessageExternal.addListener(
  function(request, sender, sendResponse) {
    if (sender.id === blocklistedExtension)
      return;  // don't allow this extension access
    else if (request.getTargetData)
      sendResponse({targetData: targetData});
    else if (request.activateLasers) {
      var success = activateLasers();
      sendResponse({activateLasers: success});
    }
  });

// For long-lived connections:
chrome.runtime.onConnectExternal.addListener(function(port) {
  port.onMessage.addListener(function(msg) {
    // See other examples for sample onMessage handlers.
  });
});

```

To send a message to another extension, pass the ID of the extension you want to communicate with as follows:

service-worker.js

```
// The ID of the extension we want to talk to.
var laserExtensionId = "abcdefghijklmnoabcdefhijklmnoabc";

// For a simple request:
chrome.runtime.sendMessage(laserExtensionId, {getTargetData: true},
  function(response) {
    if (targetInRange(response.targetData))
      chrome.runtime.sendMessage(laserExtensionId, {activateLasers: true});
  }
);

// For a long-lived connection:
var port = chrome.runtime.connect(laserExtensionId);
port.postMessage(...);

```

## Send messages from web pages

Extensions can also receive and respond to messages from other web pages, but can't send messages to web pages. To send messages from a web page to an extension, specify in your `manifest.json` which websites you want to communicate with using the `["externally_connectable"](https://developer.chrome.com/docs/extensions/reference/manifest/externally-connectable)` manifest key. For example:

manifest.json

```
"externally_connectable": {
  "matches": ["https://*.example.com/*"]
}

```

This exposes the messaging API to any page that matches the URL patterns you specify. The URL pattern must contain at least a [second-level domain](https://wikipedia.org/wiki/Second-level_domain); that is, hostname patterns such as "\*", "\*.com", "\*.co.uk", and "\*.appspot.com" are not supported. Starting in Chrome 107, you can use `<all_urls>` to access all domains. Note that because it affects all hosts, Chrome web store reviews for extensions that use it [may take longer](https://developer.chrome.com/docs/webstore/review-process#review-time-factors).

Use the `[runtime.sendMessage()](https://developer.chrome.com/docs/extensions/reference/api/runtime#method-sendMessage)` or `[runtime.connect()](https://developer.chrome.com/docs/extensions/reference/api/runtime#method-connect)` APIs to send a message to a specific app or extension. For example:

webpage.js

```
// The ID of the extension we want to talk to.
var editorExtensionId = "abcdefghijklmnoabcdefhijklmnoabc";

// Make a simple request:
chrome.runtime.sendMessage(editorExtensionId, {openUrlInEditor: url},
  function(response) {
    if (!response.success)
      handleError(url);
  });

```

From your extension, listen to messages from web pages using the `[runtime.onMessageExternal](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onMessageExternal)` or `[runtime.onConnectExternal](https://developer.chrome.com/docs/extensions/reference/api/runtime#event-onConnectExternal)` APIs as in [cross-extension messaging](#external). Here's an example:

service-worker.js

```
chrome.runtime.onMessageExternal.addListener(
  function(request, sender, sendResponse) {
    if (sender.url === blocklistedWebsite)
      return;  // don't allow this web page access
    if (request.openUrlInEditor)
      openUrl(request.openUrlInEditor);
  });

```

## Native messaging

Extensions [can exchange messages](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging#native-messaging-client) with native applications that are registered as a [native messaging host](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging#native-messaging-host). To learn more about this feature, see [Native messaging](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging).

## Security considerations

Here are a few security considerations related to messaging.

### Content scripts are less trustworthy

[Content scripts are less trustworthy](https://developer.chrome.com/docs/extensions/develop/security-privacy/stay-secure#content_scripts) than the extension service worker. For example, a malicious web page might be able to compromise the rendering process that runs the content scripts. Assume that messages from a content script might have been crafted by an attacker and make sure to [validate and sanitize all input](https://developer.chrome.com/docs/extensions/develop/security-privacy/stay-secure#sanitize). Assume any data sent to the content script might leak to the web page. Limit the scope of privileged actions that can be triggered by messages received from content scripts.

### Cross-site scripting

Make sure to protect your scripts against [cross-site scripting](https://wikipedia.org/wiki/Cross-site_scripting). When receiving data from an untrusted source such as user input, other websites through a content script, or an API, take care to avoid interpreting this as HTML or using it in a way which could allow unexpected code to run.

Safer methods

Use APIs that don't run scripts whenever possible:

service-worker.js

```
chrome.tabs.sendMessage(tab.id, {greeting: "hello"}, function(response) {
  // JSON.parse doesn't evaluate the attacker's scripts.
  var resp = JSON.parse(response.farewell);
});
```

service-worker.js

```
chrome.tabs.sendMessage(tab.id, {greeting: "hello"}, function(response) {
  // innerText does not let the attacker inject HTML elements.
  document.getElementById("resp").innerText = response.farewell;
});
```

Unsafe methods

Avoid using the following methods that make your extension vulnerable:

service-worker.js

```
chrome.tabs.sendMessage(tab.id, {greeting: "hello"}, function(response) {
  // WARNING! Might be evaluating a malicious script!
  var resp = eval(`(${response.farewell})`);
});
```

service-worker.js

```
chrome.tabs.sendMessage(tab.id, {greeting: "hello"}, function(response) {
  // WARNING! Might be injecting a malicious script!
  document.getElementById("resp").innerHTML = response.farewell;
});
```



Source: <https://developer.chrome.com/docs/extensions/develop/concepts/messaging>