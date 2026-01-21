# Govee Bluetooth Temperature and Humidity Monitor

Model GVH5105

## Integrating with [Home Assistant.md](./Home%20Assistant.md)

[This](https://community.home-assistant.io/t/add-support-for-govee-h5105-ble/663862) post suggests that these are discoverable on stock ESPHome BtProxy:

> I’ve just powered up three of these devices and all were automatically discovered by Home assistant (I’m using an ESPHome BTProxy if that makes any difference). The only thing that doesn’t seem to be working is the Signal Strength; that just says “Entity is Unavailable”.

# ESPHome Bluetooth Proxy moveme 

[ESPHome](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/97b13360-fb08-40a3-9804-e03efb457535) is a addon to [Home Assistant.md](./Home%20Assistant.md), and it published a [Bluetooth Proxy component](https://esphome.io/components/bluetooth_proxy.html):

> Home Assistant can expand its Bluetooth reach by communicating through the Bluetooth proxy component in ESPHome. The individual device integrations in Home Assistant (such as BTHome) will receive the data from the Bluetooth Integration in Home Assistant which automatically aggregates all ESPHome Bluetooth proxies with any USB Bluetooth Adapters you might have.

The ESPHome website published some “ready made” device lists, and [the one for BTProxy](https://esphome.io/projects/index.html) suggests the **Olimex ESP32 Power-over-Ethernet ISO.** A [Reddit thread](https://www.reddit.com/r/Esphome/comments/1gyr4lo/what_is_the_best_esp32_for_a_bluetooth_proxy/) also noted that this was the one with the least problems per the dev team.

Research suggests that you shouldn’t use wifi, unless your device has two radios and antennae - you want one radio dedicated to Bluetooth traffic, and so it’s recommended to use Ethernet for connecting to the Home Assistant and using the radio for Bluetooth only.

[BTProxy device recommendations](https://www.reddit.com/r/Esphome/comments/1gyr4lo/what_is_the_best_esp32_for_a_bluetooth_proxy/)

### Olimex ESP32-POE-ISO-EA

> ESP32 board with wired Ethernet connection that can also be powered using Power over Ethernet 802.3af. Note that when installed via this website, Wi-Fi is disabled and it needs to be connected via Ethernet. The *ESP32-POE-ISO-EA* variant may provide better Bluetooth range since it has an external antennae. [Case on Thingiverse.](https://www.thingiverse.com/thing:3857281)

### GL-Inet GL-S10

> I should go with this one.

However there’s a more hands off option with the gl-inet GL-S10: <https://store.gl-inet.com/products/gl-s10-ble-iot-gateway?variant=39514685145182> and it’s only 30 bucks.

Theres a website to flash it with ESPhome: <https://blakadder.com/gl-s10/>