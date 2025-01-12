# dho-remote
Control your digital oscilloscope remotely. Create FFT and Bode plots and save raw data.

![Bodeplot_RC_1KHz](https://github.com/user-attachments/assets/5ae93d70-a034-462b-91e0-87efd4508b6e)

From the user interface you can request a Bode plot from raw data of the oscilloscope. A network connection is needed.
The Bode plot shows the amplitude and phase difference between CH2 and CH1.
The program takes an FFT of CH1 and only uses frequency bins that contain sufficient energy. 
CH1 should be driven by a signal that contains a lot of harmonics like a square wave or a pulse wave. 
But any input will do as long as there is frequency content (harmonics) at the point you want to see.
Since the harmonics typically becomes lower at higher order (20dB/decade), a Bode plot made from a single capture covers
3 decades in frequencies before the noise becomes too large. 
A pre-capture options lets you capture multiple times. Each capture can be driven from a different fundamental frequency
to extend the frequency range. All captures will be plotted as a single Bode diagram.

## Example of measuring an audio amplifier

* python dho-remote.py 192.168.1.8
* Use a 20Hz square wave signal at the input and measure the input on CH1.
* Connect the output of the amplifier to CH2.
* Scale the horizontal timebase such that you see a few periods on your screen (or use AUTO setting)
* Scale the vertical setting such that the signal is a big as possible, but don't let it clip (or use AUTO setting)
* Press "Plot" and you will get a Bode plot from 20Hz to well above 20kHz.

## Example of measuring an crystal
Watch this [youtube video](https://www.youtube.com/watch?v=M2XBamR0O_g) to see how you a crystal can be measured.


# Plot navigation
The plots are made by pyplot. See [Navigation keyboard shortcuts](https://matplotlib.org/stable/users/explain/figure/interactive.html#navigation-keyboard-shortcuts)

# Scope support
I only tested this with a RIGOL DHO804. The brand can be specified with the -m command, but the only option is "rigol" at the moment.
Support for other oscillope types/brands should be straigthforward by making a copy of the scope model and changing the SCPI commands.

# Tips
Limit the bandwidth of the channels to 20MHz if possible. The sample rate should at least be 40Ms/s to prevent noise folding. 
This could otherwise limit the dynamic range. The memory depth sets the lowest frequency of the FFT:
* f(min)=SR/MDEPTH
* f(max)=SR/2

A swaw-tooth signal is optimal as it contains both odd and even harmonics. This will give a somewhat better resolution 
in the first decade.

Noise can be lowered by averaging on the oscilloscope. N=16 will give a 6dB lower noise floor.
# Update log
* V1.1
1. corrected max selection when using multiple captures
2. Fixed phase display from radian to degrees.
