---
tags:
  - experiment
---
# Tana Hacking

Notes on the Tana Input API

### Old clipper code from verveguy

<https://github.com/verveguy/clip2tana>

### Raycast extension to check out

<https://www.raycast.com/cheslip/tana>

### Creating a supertag using the input api

```bash
const schemaNodeId = "SCHEMA";

const endpoint =
	"https://europe-west1-tagr-prod.cloudfunctions.net/addToNodeV2";

export async function makeRequest(payload, token) {
	const response = await fetch(endpoint, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(payload),
	});

	if (response.status === 200 || response.status === 201) {
		const json = await response.json();
		if ("setName" in payload) {
			return [json];
		}
		return json.children;
	}
	console.log(await response.text());
	throw new Error(`${response.status} ${response.statusText}`);
}

export async function createTagDefinition(node, token) {
	if (!node.supertags) {
		node.supertags = [];
	}
	node.supertags.push({ id: coreTemplateId });
	const payload = {
		targetNodeId: schemaNodeId,
		nodes: [node],
	};

	const createdTag = await makeRequest(payload, token);
	return createdTag[0].nodeId;
}
```

> <https://github.com/tanainc/tana-input-api-samples#creating-fields--tags>
>
> You just have to use the tag id SYS_T01 (edited)  

### Tana raycast script command to review

```bash
#!/usr/bin/env node

// Required parameters:
// @raycast.schemaVersion 1
// @raycast.title Save Page to Tana
// @raycast.mode silent
// @raycast.packageName Save to Tana

// Optional parameters:
// @raycast.icon ⚡️

import { runAppleScript } from "run-applescript";
import { token } from "./token.js";
import { TidyURL } from "tidy-url";
import { removeTrackingParams } from "@nbbaier/remove-tracking-parameters";

const content = await runAppleScript(
  `tell application "Arc"
set currentURL to URL of active tab of window 1
set currentTitle to title of active tab of window 1
end tell
return [currentURL, currentTitle]
`,
  { humanReadableOutput: false }
);

const baseLink = TidyURL.clean(
  JSON.parse(content.replace("{", "[").replace("}", "]"))[0]
).url;

const apiUrl = new URL(
  "https://europe-west1-tagr-prod.cloudfunctions.net/addToNode"
);

apiUrl.searchParams.append("note", removeTrackingParams(baseLink));

(async () => {
  try {
    const response = await fetch(apiUrl, {
      method: "GET",
      headers: {
        Authorization: "Bearer " + token,
        "Content-Type": "application/json",
      },
    });

    console.log("Page saved!!");
  } catch (error) {
    console.error(error);
  }
})();
```