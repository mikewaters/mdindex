# Flexible LED Panel CoolLED1248

Requires the [CoolLED1248](https://apps.apple.com/us/app/coolled1248/id1435796557) app, pairing appears to be Bluetooth.

Markings:

- `JT-2121-1632-16S-V5.3`

The CoolLED1248 app works fine on the Mac, and discovers the device via Bluetooth.

## Info

` core-bluetooth-tool scan 48D172C8-A082-ECBA-0FCC-6283DC107E0F`

Its BLE UUID is `48D172C8-A082-ECBA-0FCC-6283DC107E0F`

Its BLE name is `CoolLEDX`

## Ways to connect to it?

Somebody has done this!!!

### **CoolLEDX Driver**

<https://pypi.org/project/coolledx/> 

<https://github.com/UpDryTwist/coolledx-driver>

examples in code search: <https://github.com/search?q=coolledx&type=code>

### JT-Edit

> Use the browser to create JT files for the cool led panel
>
> <https://github.com/auc0le/JT-Edit>

### wtf **[CoolLed-VoiceControl-App](https://github.com/NotBlue-Dev/CoolLed-VoiceControl-App)**

> This project is a React Native application that integrates voice recognition, Bluetooth Low Energy (BLE) communication, and text-to-speech (TTS) functionalities. Allows users to interact with CoolLED devices through voice commands, providing a seamless experience for controlling various features such as brightness, color, text, images and animations.

Code: <https://github.com/NotBlue-Dev/CoolLed-VoiceControl-App>

Companion API: <https://github.com/NotBlue-Dev/CoolLed-VoiceControl-Api>

### My BT dev notes

[Bluetooth Programming (MacOS).md](./Bluetooth%20Programming%20\(MacOS\).md)

## Device Info

### [Thread](https://forum.arduino.cc/t/cant-find-schematics-or-programs-for-specific-led-matrix/1058399) on an Arduino forum:

> The schematics/programs for[ this led matrix ](https://www.alibaba.com/product-detail/iledshow-Customized-flexible-led-display-led_62447860966.html)don't seem to exist

![Pasted 2024-12-19-19-18-40.jpeg](./Flexible%20LED%20Panel%20CoolLED1248-assets/Pasted%202024-12-19-19-18-40.jpeg)

> … what I've learned so far is that the PCB glued in with the battery seems to mainly handle the charging and WiFi connectivity. The actual memory and control of the matrix seem to happen on the LED matrix side with a HC89F0541 MCU and a bunch of other chips.\
> While trying to send commands to it I connected it to 5V and it displayed a "Battery Charging" symbol. I'm going to try and dig further to see if it's possible to control this with an Arduino. So far I've also found this page dedicated to a similar matrix: [Asian Flexible LED matrix – Anders K Nelson](https://www.andersknelson.com/blog/?p=964)

### Links

[Asian Flexible LED matrix](https://www.andersknelson.com/blog/?p=964) analysis

[LED Face shields](https://git.team23.org/CrimsonClyde/led-faceshields) (git)



Similar [item](https://www.amazon.com/Bluetooth-Programmable-Flexible-Display-Application/dp/B0B3XY6G1N?th=1): 

![ScreenShot 2024-12-19 at 19.17.21@2x.png](./Flexible%20LED%20Panel%20CoolLED1248-assets/ScreenShot%202024-12-19%20at%2019.17.21@2x.png)