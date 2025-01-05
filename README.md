# dho-remote
Control your Rigol DHO remotely. Create FFT and bode plots.

![Bodeplot_RC_1KHz](https://github.com/user-attachments/assets/5ae93d70-a034-462b-91e0-87efd4508b6e)

From the user interface you can request a Bode plot from raw data of the oscilloscope. A network connection is needed.
The Bode plot shows the amplitude and phase difference between the CH2 and CH1.
The script takes an FFT of CH1 and only uses frequency bins that contain sufficient energy. CH1 should be driven by a signal that contains a lot of harmonics like a square wave or a pulse wave. But any input will do as long as there is frequency content (harmonics) at the point you want to see.
## Example of measuring an audio amplifier

* python dho-remote.py 192.168.1.8
* Use a 20Hz square wave signal at the input and measure the input on CH1.
* Connect the output of the amplifier to CH2.
* Scale the horizontal timebase such that you see a few periods on your screen (or use AUTO setting)
* Scale the vertical setting such that the signal is a big as possible, but don't let it clip (or use AUTO setting)
* Press "Plot" and you will get a Bode plot from 20Hz to well above 20kHz.

# Plot navigation
The plots are made by pyplot. See [Navigation keyboard shortcuts](https://matplotlib.org/stable/users/explain/figure/interactive.html#navigation-keyboard-shortcuts)

# Scope support
I only tested this with a RIGOL DHO804. The brand can be specified with the -m command, but the only option is "rigol" at the moment.
Support for other oscillope types/brands should be straigthforward by making a copy of the scope model and changing the SCPI commands.

# Tips
Limit the bandwidth of the channels to 20MHz if you possible. The sample rate should at least be 40Ms/s to prevent noise folding. This could otherwise limit the dynamic range. The memory depth sets the lowest frequency of the FFT:
* f(min)=SR/MDEPTH
* f(max)=SR/2
