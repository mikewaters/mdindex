# UCA Leads Architecture

## Notes

Jul 24, 2024

> A quick look at the UCA code shows that DDC leads are routed through Accelerate infrastructure - the AIO proxy. Here is a short descriptive doc. However, there is the concept of leadRoutingOverrideSettings (dealer config level), which does some mangling of the request payload going to Finance Services in the submission lambda, which makes me believe that there is some lead routing happening in FS itself (Lead Driver?). I do not know if this "override" prevents the normal lead routing path though, or if they are sent both ways.
>
> There's more: The "container" for UCA in the Accelerate flow is called "Creditapp Activity" - this is what we emit events to. There are two events in the Creditapp Activity that, when received, result in a lead being submitted. One of those events is the Personal Info page being filled. Another is the confirmation page. (edited) 
>
> NEW
>
> 11:14
>
> Personal Info page event handler (aka leadSubmissionHandler: <https://ghe.coxautoinc.com/DigitalRetailing/dr-activities-credit-app/blob/e1ac4344cba0[…]ceea472d400db79/src/components/accelerate/messageHandler.ts> (edited) 
>
> 11:15
>
> Confirmation page handler: <https://ghe.coxautoinc.com/DigitalRetailing/dr-activities-credit-app/blob/e1ac4344cba0[…]ceea472d400db79/src/components/accelerate/messageHandler.ts>
>
> 11:15
>
> So the implication here is that UCA events information out to Accelerate, which may cause leads to be submiitted by Accelerate.

> UCA sends DDC leads via Accelerate AIO
>
> UCA has a lead override that seems to do something with Lead Driver, but its just a hint until we can prove it in DTA systems
>
> UCA sends events up to Accelerate (when embedded there) that appear to have some CRM lead integration, but I cannot figure out how (accelerate uses a lot of magic aka Redux)
>
> 

## Diagrams

[SCA Lead Routing Flow](https://lucid.app/lucidchart/ab2bc82f-f4c0-48b7-8506-4134b73127a9/edit?invitationId=inv_43d80041-b232-4a40-ad0b-d27159ccd84e&page=PXeXsdmkU68K#)

![SCA Lead Routing Flow.jpeg](./UCA%20Leads%20Architecture-assets/SCA%20Lead%20Routing%20Flow.jpeg)

![ScreenShot 2024-07-24 at 12.43.24@2x.png](./UCA%20Leads%20Architecture-assets/ScreenShot%202024-07-24%20at%2012.43.24@2x.png)

[What is AIO?](https://lucid.app/lucidchart/d1fd0405-bb99-4a8c-ac43-af58afac9b70/edit?invitationId=inv_5d3606be-5930-4c31-8b38-3c58355a5908&page=AF.HoC\~0LLyG#)

## Splunk queries (Digital Retailing app)

> index=aws aio-ddc-intake-processor sourcetype="aws:cloudwatchlogs:lambda" EVENT_RECEIVED

> index=aws "DDC " sourcetype="aws:cloudwatchlogs:lambda"