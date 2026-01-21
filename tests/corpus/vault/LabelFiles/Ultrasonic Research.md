# Ultrasonic Research

<https://news.ycombinator.com/item?id=40624709>

Ultrasonic occupancy sensors 25kHz or 40 kHz

Ultrasonic PDC sensors in vehicles /parking sensor

### Article: Ultrasonic investigations in shopping centres

> pilot tones

> Many multi-loudspeaker PA systems (like the [Zenitel VPA](https://wiki.zenitel.com/wiki/VPA\_-\_Public_Address_Amplifiers#Speaker_line_fault_detection) and [Axys End of Line detection unit](https://axystunnel.com/en/products/end-of-line-detection-eol)) employ these roughly 20-kilohertz tones 

<https://www.windytan.com/2024/06/ultrasonic-investigations-in-shopping.html>

I can't remember how I first came across these near-ultrasonic 'beacons' ubiquitous in PA systems. I might have been scrolling through the audio spectrum while waiting for the underground train; or it might have been the screeching 'tinnitus-like' sensation I would often get near the loudspeakers at a local shopping centre.

![Pasted 2024-06-09-19-59-09.png](./Ultrasonic%20Research-assets/Pasted%202024-06-09-19-59-09.png)

Whatever the case, I learned that they are called pilot tones. Many multi-loudspeaker PA systems (like the [Zenitel VPA](https://wiki.zenitel.com/wiki/VPA\_-\_Public_Address_Amplifiers#Speaker_line_fault_detection) and [Axys End of Line detection unit](https://axystunnel.com/en/products/end-of-line-detection-eol)) employ these roughly 20-kilohertz tones to continuously measure the system's health status: no pilot tone means no connection to a loudspeaker. It's usually set to a very high frequency, inaudible to humans, to avoid disturbing customers.

However, these tones are powerful and some people will still hear them, especially if the frequency gets below 20 kHz. There is one such system at 19.595 kHz in my city; it's marked green in the graph above. I've heard of several other people that also hear the sound. I don't believe it to be a sonic weapon like [The Mosquito](https://en.wikipedia.org/wiki/The_Mosquito); those use even lower frequencies, down to 17 kHz. It's probably just a misconfiguration that was never fixed because the people working on it couldn't experientially confirm any issue with it.

### Hidden modulation.

Pretty quickly it became apparent that this sound is almost never a pure tone. Some kind of modulation can always be seen wiggling around it in the spectrogram. Is it caused by the background music being played through the PA system? Is it carrying some information? Or is it something else altogether?

I've found at least one place where the tone appears to be amplitude modulated by the lowest frequencies in the music or commercials playing. It's probably a side effect of some kind of distortion.

Here's a spectrogram plot of this amplitude modulation around the strong pilot tone. It's colour-coded so that the purple colour is coming from the right microphone channel and green from the left. I'm not quite sure what the other purple horizontal stripes are here.

![Pasted 2024-06-09-19-59-09.jpeg](./Ultrasonic%20Research-assets/Pasted%202024-06-09-19-59-09.jpeg)

But this kind of modulation is rare. It's more common to see the tone change in response to things happening around you, like people moving about. More on that in the following.

### Doppler-shifted backscatter.

Look what happens to the pilot tone when a train arrives at an underground station:

![Pasted 2024-06-09-19-59-09 1.jpeg](./Ultrasonic%20Research-assets/Pasted%202024-06-09-19-59-09%201.jpeg)

The wideband screech in the beginning is followed by this interesting tornado-shaped pattern that seems to have a lot of structure inside of it. It lasts for 15 seconds, until the train comes to a stop.

It's my belief that this is backscatter, also known as reverb, from the pilot tone reflecting off the slowing train and getting Doppler-shifted on the way. The pilot tone works as a continuously-transmitting [bistatic sonar](https://en.wikipedia.org/wiki/Bistatic_sonar). Here, the left microphone (green) hears a mostly negative Doppler shift whereas the right channel (purple) hears a positive one, as the train is passing us from right to left. An anti-aliasing filter has darkened the higher frequencies as I wasn't yet aware I would find something interesting up here when recording.

A zoomed-in view of this cone reveals these repeating sweeps from positive to negative and red to green. Are they reflections off of some repeating structure in the passing train? The short horizontal bursts of constant tone could then be surfaces that are angled in a different direction than the rest of the train. Or perhaps this repetition reflects the regular placement of loudspeakers around the station?

![Pasted 2024-06-09-19-59-09 2.jpeg](./Ultrasonic%20Research-assets/Pasted%202024-06-09-19-59-09%202.jpeg)

### Moving the microphone.

Another interesting experiment: I took the lift to another floor and recorded the ride from inside the lift. It wasn't the metal box type, the walls were made of glass, so I thought the pilot tone should be at least somewhat audible inside. Here's what I got during the 10-second ride. It's a little buried in noise.

![Pasted 2024-06-09-19-59-09 3.jpeg](./Ultrasonic%20Research-assets/Pasted%202024-06-09-19-59-09%203.jpeg)

### Scooter calculation.

For the next experiment I went into the underground car park of a shopping centre. I stood right under a PA loudspeaker and recorded a person on a scooter passing by. A lot of interesting stuff is happening in this stereo spectrogram!

![Pasted 2024-06-09-19-59-09 4.jpeg](./Ultrasonic%20Research-assets/Pasted%202024-06-09-19-59-09%204.jpeg)

First of all, there seems to be two pilot tones, one at 19,595 Hz and a much quieter one at 19,500 Hz. Are there two different PA systems in the car park?

Second, there's a clear Doppler shift in the reverb. The frequency shift goes from positive to negative at the same moment that the scooter passes us, seen as the wideband wheel noise changing color. It looks like the pattern is also 'filled in' with noise under this Doppler curve. What all information can we find out just by looking at this image?

If we ignore the fact that this is actually a [bistatic doppler shift](https://en.wikipedia.org/wiki/Bistatic_radar#Doppler_shift) we could try and estimate the speed using [a formula on Wikipedia](https://en.wikipedia.org/wiki/Doppler_radar#Frequency_variation). It was pretty chilly in the car park, I would say 15 °C. The speed of sound at 15 °C is 340 m/s. The maximum Doppler shift here seems to be 350 Hz. Plugging all these into the equation we get 11 km/h, which sounds like a realistic speed for a scooter.

Why is it filled in? My thought is these are reflections off different points on our test subject. There's variation in the reflection angles and, consequently, magnitudes of the velocity component that causes frequency shifting, down to nearly zero Hz.

### What now?

What would you do with this ultrasonic beep all around you? I have some free ideas:

- Automated speed trap in the car park

- Detect when the escalators stop working

- Modulate it with a positioning code to prevent people getting lost in the maze of commerce

- Use it to deliver ads somehow

- Use it to find your way to the quietest spots in a shopping centre

[Oona Räisänen](https://www.windytan.com/p/about.html#who)