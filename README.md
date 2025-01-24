# dho-remote
Control your digital oscilloscope remotely. Create FFT and Bode plots and save raw data.

![dso-remote](https://github.com/user-attachments/assets/7ea66529-6ad8-4621-9ad9-0dd2b22d09b0)


From the user interface you can request a Bode plot from raw data of the oscilloscope. A network connection is needed.
The Bode plot shows the amplitude and phase difference between CH1 and CH2. CH1 of the scope is connected to the input
of your network and CH2 is connected to the output. You need to drive the input of your network (CH1) with a square wave
signal. The program takes an FFT of CH1 and only uses frequency bins that contain sufficient energy to create the plot. 
CH1 should be driven by a signal that contains a lot of harmonics like a square wave or a pulse wave. 
But any input will do as long as there is frequency content (harmonics) at the point you want to see.
Since the harmonics typically lower at higher order (20dB/decade), a Bode plot made from a single capture covers
3 decades in frequencies before the noise plays a role.
A pre-capture options lets you capture multiple times. Each capture should be driven with a different fundamental frequency
to extend the frequency range. All captures will be plotted as a single Bode diagram.

## Example of measuring an audio amplifier

* python dho-remote.py -i 192.168.1.8
* Use a 20Hz square wave signal to drive the input of the amplifier and connect it to CH1.
* Connect the output of the amplifier to CH2.
* Press the "Setup scope" button
* Press "Bode Plot" and you will get a Bode plot from 20Hz to well above 20kHz.

## Example of measuring a crystal
Watch this [youtube video](https://www.youtube.com/watch?v=M2XBamR0O_g) to see how you a crystal can be measured.


# Plot navigation
The plots are made with pyplot. After the zoom button is selected, keep the x or y key pressed to limit the zoom to
the x or y-axis only.
The "L" key toggles the x-axis between a logarithmic and a linear scale.
All shortscuts can be fore
See [Navigation keyboard shortcuts](https://matplotlib.org/stable/users/explain/figure/interactive.html#navigation-keyboard-shortcuts)

# Scope support
I only tested this with a RIGOL DHO804. The brand can be specified with the -m command, but the only option is "rigol"
at the moment. Support for other oscilloscope types/brands should be straightforward by making a copy of the scope model
and changing the SCPI commands.

# Tips
Limit the bandwidth of the channels to 20MHz if possible. The sample rate should at least be 40Ms/s to prevent noise folding. 
This could otherwise limit the dynamic range. The memory depth sets the lowest frequency of the FFT:
* f(min)=SR/MDEPTH
* f(max)=SR/2

Set the horizontal timebase such that only a single rising or falling edge is displayed in the captured data. The response 
on CH2 should also fit completely on the screen. For high-Q circuits the response can take much more time and is seen as
ringing. The button "Setup scope" will setup the scope based on the detected signal on CH1.
The exact settings follows from f(min).

Noise can be lowered by averaging on the oscilloscope. N=16 will give a 6dB lower noise floor.
# Known issues
Pre capture only extends the frequency range if the setting of the scope remains unchanged. If the horizontal timebase 
is changed between captures, the result is wrong.

# Update log
* V1.3
1. Loopgain analysis added (argument --loopgain)
2. FFT window selection menu added
3. Hann window added
4. Selection algorithm for Bode points optimized
* V1.2
1. Auto setup added
2. Trace colors now match the scope
* V1.1
1. corrected max selection when using multiple captures
2. Fixed phase display from radian to degrees.
