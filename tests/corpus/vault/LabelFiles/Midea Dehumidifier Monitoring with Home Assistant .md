# Midea Dehumidifier Monitoring with Home Assistant 

This device requires a cloud account in order to monitor locally. This also requires an app install called SmartHome. There are some options to bypass this but they require physical hacking.

If I am going to use this, I must VLAN this shit.

### Cloud Interface

There are a few libraries that integrate with Home Assistant, the best looking on appears to be this one, which is built as a custom component and so either needs to be manually built into HASS or installed via HACS:

> V3 and V2 protocols allow local network access. V3 protocol requires one connection to Midea cloud to get token and key needed for local network access. Some old models use V1 XML based protocol which is not supported. Some newer models use Tuya protocol.

> <https://github.com/nbogojevic/homeassistant-midea-air-appliances-lan>

### Local Hacking

There are a few projects to build an ESP32 device to replace the built in WiFi adapter:

> from [https://www.reddit.com/r/homeassistant/comments/1ayctay/comment/krtxl24/](https://www.reddit.com/r/homeassistant/comments/1ayctay/comment/krtxl24/?utm_source=share&utm_medium=mweb3x&utm_name=mweb3xcss&utm_term=1&utm_content=share_button)

> <https://github.com/Hypfer/esp8266-midea-dehumidifier>
>
> <https://github.com/reneklootwijk/mideahvac-dongle>

> 
>
> The WiFi interface \[on these machines\] is provided by a dongle, called WiFi SmartKey, either connected to an USB type-A connector or a JST-HX type of connector. This dongle wraps the UART protocol used to communicate with the unit with a layer for authentication and encryption for communication with a mobile app via the Midea cloud or directly via a local LAN connection.  However, it turned out the dongle is just connected to a serial interface (TTL level) of the unit. This means an alternative is to hook up directly to this serial interface and bypass the cloud, authentication and encryption stuff, \[which is exactly what this project does\].
>
> <https://github.com/Hypfer/esp8266-midea-dehumidifier>

Research:

<https://kagi.com/search?q=homeassistant+midea+dehumidifier>

<https://www.splitbrain.org/blog/2024-07/27-comfee_midea_dehumidifier_homeassistant_setup>