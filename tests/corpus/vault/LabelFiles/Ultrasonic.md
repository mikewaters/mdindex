# Ultrasonic

[http://electronics-diy.com/electronic_schematic.php?id=](http://electronics-diy.com/electronic_schematic.php?id=108)

<https://www.electronics-diy.com/schematics/108/6.gif>

<https://www.ti.com/lit/ds/symlink/lm386.pdf>



Sonar transducer <https://a.co/d/2EotcCE> higher freq



<https://a.co/d/ejpnJLu>

These work great driven from Arduino as pest or animal repeller

Reviewed in the United States on March 24, 2022

These are loud at 5 volts. 4000hz made my ears hurt. They draw about 10ma so 2 can be hooked up to the tone pin on Arduino. Pest, animal repeller that you know that generates ultrasonic waves. Although I've seen these spec'd at 10v that much is not needed. Many repellers sold on line do not work or have insufficient volume.\
\
There is some sympathetic frequency generated at 31khz, but not bad to my old ears, but may bother others. \*Edit 3/27/22 I added a 1k resistor in series to the + side of the speaker and it is now completely silent.\
\
Code : for variable frequency on Arduino 22khz to 42khz\
/\* Varible Frequency Output From Godzilla \*/\
int speaker = 9; /\* Pin 9 \*/\
unsigned int sfrequency = 22000;\
unsigned int efrequency =42000;\
void setup(){\
pinMode(speaker, OUTPUT);\
}\
void loop(){\
for (unsigned int i = sfrequency; i <= efrequency; i=i+500) {\
tone(speaker, i);\
delay(2000); /\* 2 Seconds \*/\
}\
}

![594BB909-93E6-4014-B44A-EC3C4154BE2C.jpg](./Ultrasonic-assets/594BB909-93E6-4014-B44A-EC3C4154BE2C.jpg)