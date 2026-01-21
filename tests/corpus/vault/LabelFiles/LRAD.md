# LRAD

> … you'd need two transducers. LRAD(tm) works on a principle that when you cross two ultrasonic beams (say 20 and 25 kHz), you generate the difference between the two (5kHz sound) originating at the point where the two beams cross. Or you could use two same frequencies (23kHz and 23kHz) and FM modulate one of them with voice or audio. And another way of doing that is to phase-shift one of the 23kHz tones instead of frequency-modulating it. That delivers sound to long distances and makes it appear that the sound originates right next to you or inside your head, even though you're dozens of yards away from the transducers. And with more power, distances of hundreds of yards can be reached, which is what LRAD(tm) does.\
> \
> The actual LRAD(tm) uses a sweeping frequency in weapon mode, which can be simulated using our Phaser Sweep IC chip (in animal control mode) driving one side of stereo power amp, and driving the other input with a steady frequency (like 25 kHz). Then connect two air ultrasonic transducers to outputs of the stereo amp, and you should hear the sweeping frequency at the point where the two ultrasonic beams cross.
>
> [source](https://myskunkworks.net/index.php?route=product/product&path=61&product_id=61)

| 

> You don't say if you must build up an amplifier yourself or if you could use a commercially available part. Ideally you would need a linear amplifier with low distortion. This wouldn't necessarily be an audio amplifier as those frequencies are non-audible (at least for human ears).
>
> Here is an example of a module that uses a high power amplifier chip (20W). You could purchase the module complete from ++[here](https://www.digikey.com/en/products/detail/usound-gmbh/UA-R%25203010/9598594?utm_adgroup=&utm_source=google&utm_medium=cpc&utm_campaign=PMax%20Shopping_Product_Low%20ROAS%20Categories&utm_term=&utm_content=&utm_id=go_cmp-20243063506_adg-\_ad-\__dev-c_ext-\_prd-9598594_sig-EAIaIQobChMIrZPctbinggMVCElHAR3gjg46EAQYASABEgJHgPD_BwE&gclid=EAIaIQobChMIrZPctbinggMVCElHAR3gjg46EAQYASABEgJHgPD_BwE)++. Also note that the ++[amplifiers datasheet](https://www.usound.com/wp-content/uploads/2018/10/Amalthea-1.0-datasheet.pdf)++ includes the internal schematic so you could build up something similar if that were needed. The datasheet for the actual chip used in the module is ++[here](https://www.ti.com/lit/ds/symlink/lm1875.pdf?ts=1698928041495&ref_url=https%253A%252F%252Fwww.ti.com%252Fproduct%252FLM1875%253Futm_source%253Dgoogle%2526utm_medium%253Dcpc%2526utm_campaign%253Dasc-null-null-44700045336317122_prodfolderdynamic-cpc-pf-google-wwe_int%2526utm_content%253Dprodfolddynamic%2526ds_k%253DDYNAMIC%2BSEARCH%2BADS%2526DCM%253Dyes%2526gad_source%253D1%2526gclid%253DEAIaIQobChMIi_jejailggMVcotQBh2kEQPWEAAYASAAEgIUuPD_BwE%2526gclsrc%253Daw.ds)++.
>
> Also note that for this amplifier module you may need to tone down (reduce) the input from your system as the datasheet lists a maximum input level of 650 mVrms.
>
> This is just the first part that I found. You should be able to find many other commercially available linear amplifiers from suppliers such as Digikey, Mouser, Amazon, etc. Just be sure to check that the specifications satisfy you frequency range and power requirements.\
> [source](https://electronics.stackexchange.com/questions/687654/power-amplifier-for-ultrasonic-transducers)