---
tags:
  - document 📑
---
# Bluetooth Programming (MacOS)

### Command line

`core-bluetooth-tool (brew) `[github](https://github.com/mickeyl/core-bluetooth-tool)



### Python libraries

All are wrappers around Core Bluetooth on MacOS

+ `[pybluez](https://github.com/karulis/pybluez)` (fork of [pybluez/pybluez](https://github.com/pybluez/pybluez)) package can discover BLE devices, but isnt being developed anymore

   ```json
   # simple inquiry example
   import bluetooth
   
   nearby_devices = bluetooth.discover_devices(lookup_names=True)
   print("found %d devices" % len(nearby_devices))
   
   for addr, name in nearby_devices:
       print("  %s - %s" % (addr, name))
   ```

+ [Bleak](https://github.com/hbldh/bleak) (Bluetooth Low Energy platform Agnostic Klient) - active project

   ```python
   import asyncio
   from bleak import BleakScanner
   
   async def main():
       devices = await BleakScanner.discover()
       for d in devices:
           print(d)
   
   asyncio.run(main())
   ```

+ [Bless](https://github.com/kevincar/bless) (Bluetooth Low Energy (BLE) Server Supplement) - no updates recently