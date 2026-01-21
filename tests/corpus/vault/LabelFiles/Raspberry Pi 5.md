---
tags:
  - document 📑
---
# Raspberry Pi 5

<https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-mechanical-drawing.pdf?utm_source=chatgpt.com>



**1\. USB Connectors**

• **USB 3.0 Ports**:

• **Type**: Standard Type-A.

• **Purpose**: High-speed data transfer (up to 5 Gbps) and power for peripherals like external drives, keyboards, and mice.

• **Cable**: USB Type-A to Type-A or Type-A to Type-B (depending on the peripheral).

• **USB-C Power Port**:

• **Purpose**: Powers the Raspberry Pi 5 (requires a 5V, 5A USB-C power adapter).

• **Cable**: USB-C power cable.



**2\. HDMI Connectors**

• **Micro HDMI Ports**:

• **Number**: 2 (supports dual monitors).

• **Purpose**: Transmit video and audio to displays.

• **Standards**: Supports up to 4K resolution at 60 fps.

• **Cable**: Micro HDMI to HDMI cable.



**3\. GPIO (General Purpose Input/Output)**

• **40-Pin GPIO Header**:

• **Purpose**: Connects to external hardware like sensors, LEDs, and motor drivers.

• **Protocols Supported**: I2C, SPI, UART, and more.

• **Cable**: GPIO ribbon cable or individual jumper wires.



**4\. MIPI Interfaces**

• **MIPI DSI (Display Serial Interface)**:

• **Purpose**: Connects Raspberry Pi-specific displays, like the official touchscreen.

• **Cable**: Flat Flexible Cable (FFC).

• **MIPI CSI (Camera Serial Interface)**:

• **Purpose**: Connects cameras like the Raspberry Pi Camera Module.

• **Cable**: Flat Flexible Cable (FFC).



**5\. Audio/Video**

• **3\.5mm Audio/Composite Jack**:

• **Purpose**: Outputs analog audio and composite video.

• **Cable**: 3.5mm TRRS cable.



**6\. PCIe Connector**

• **Purpose**: High-speed expansion for peripherals like NVMe SSDs, networking cards, or other PCIe devices.

• **Connector**: PCIe Gen 2 x1 interface.

• **Cable/Adapter**: Requires a PCIe adapter or breakout board.



**7\. MicroSD Card Slot**

• **Purpose**: Holds the microSD card for operating system and storage.

• **Cable**: Not applicable, but microSD cards are essential.



**8\. Ethernet Port**

• **Type**: Gigabit Ethernet (RJ45).

• **Purpose**: High-speed wired network connection.

• **Cable**: Ethernet cable (Cat 5e or higher recommended).



**9\. Fan Connector**

• **Type**: 3-pin fan connector.

• **Purpose**: Connects a cooling fan for active cooling.

• **Cable**: 3-pin fan cable.



**10\. USB Debug Connector**

• **Type**: Micro-USB.

• **Purpose**: For low-level debugging and diagnostics.

• **Cable**: Micro-USB cable.



**11\. Power Management Connector (PoE HAT)**

• **Type**: PoE (Power over Ethernet) Header.

• **Purpose**: Allows power delivery through a compatible PoE HAT.

• **Cable**: Ethernet cable (if using PoE).



**12\. Other Specialized Interfaces**

• **RTC (Real-Time Clock) Connector**:

• **Purpose**: For attaching an external RTC module.

• **Cable**: Varies depending on the RTC module.

• **Debug GPIO Header**:

• **Purpose**: Advanced debugging.

• **Cable**: Jumper wires or custom debug cables.



**Typical Cables and Accessories**

• **Flat Flexible Cables (FFC)** for CSI/DSI interfaces.

• **Micro HDMI to HDMI cables** for video output.

• **USB-C power cables** for powering the board.

• **GPIO ribbon cables** for external hardware connections.

## MIPI

MIPI (Mobile Industry Processor Interface) is a standard for high-speed, low-power interconnects primarily used in mobile and embedded systems. It is managed by the **MIPI Alliance**, a global consortium that defines specifications to ensure compatibility across devices. Two commonly used MIPI interfaces in the Raspberry Pi and other devices are **CSI (Camera Serial Interface)** and **DSI (Display Serial Interface)**.



**1\. MIPI CSI (Camera Serial Interface)**

• **Purpose**: Connects cameras to a host processor, such as the Raspberry Pi.

• **Use Case**: Designed for transmitting high-definition video and image data from a camera module to a processor.

• **Data Transfer**: Transfers data from camera sensors to the processor for processing.

• **Structure**:

• Includes data lanes and a clock lane.

• Uses **MIPI CSI-2**, the most common version, which supports multiple data lanes for increased bandwidth.

• Data is transmitted serially over **Differential Pair** lanes.

• **Key Features**:

• High frame rate support.

• Low power consumption.

• Often used with camera modules like the Raspberry Pi Camera Module.

• **Cable**: Uses a Flat Flexible Cable (FFC) or custom ribbon cable.



**2\. MIPI DSI (Display Serial Interface)**

• **Purpose**: Connects displays to a host processor, such as the Raspberry Pi.

• **Use Case**: Designed for driving LCD, OLED, or other display panels.

• **Data Transfer**: Sends processed video or graphical data from the processor to the display panel.

• **Structure**:

• Similar to CSI, with data lanes and a clock lane.

• Uses **MIPI DSI-1** and **DSI-2**, supporting high-resolution displays and reduced power consumption.

• Can support touch-sensitive displays (when integrated with touch controllers).

• **Key Features**:

• High-speed serial communication for graphics.

• Reduced EMI (Electromagnetic Interference).

• Supports commands for display initialization and control.

• **Cable**: Typically uses an FFC ribbon cable.



**Comparison Between MIPI CSI and DSI**



**Feature** **MIPI CSI** **MIPI DSI**

**Primary Purpose** Connect cameras to processors Connect displays to processors

**Data Direction** Sensor (camera) → Processor Processor → Display

**Data Type** Raw image/video data Processed video/graphics

**Bandwidth** High (for high-res video capture) High (for high-res displays)

**Power Consumption** Low Low

**Applications** Cameras, video capture, drones Smartphones, LCD/OLED displays

**Cable Type** FFC ribbon or custom cable FFC ribbon

**Common Protocol** CSI-2 DSI-1 or DSI-2



**Key Differences**

1\. **Direction of Data**:

• CSI: Data flows *to* the processor from a camera.

• DSI: Data flows *from* the processor to a display.

2\. **Data Type**:

• CSI: Handles raw data from image sensors.

• DSI: Handles processed display signals.

3\. **Primary Use**:

• CSI: Image/video acquisition.

• DSI: Driving visual output.






