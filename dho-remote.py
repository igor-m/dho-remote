#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
from scipy.signal.windows import flattop, blackman, hann, kaiser
from scipy.fft import rfft, rfftfreq
from functools import partial
import pickle
import datetime

from scope.rigol import Scope, Siglent, ChannelNotEnabled

version = "v1.3"

class ScopeUI(tk.Tk):
    # data1: np.array
    # data2: np.array

    def __init__(self, rm):
        super().__init__()
        # self.geometry("240x200")
        self.instr = rm.instr
        self.get_data = rm.get_data
        self.bode_setup = rm.bode_setup

        self.title("DHO-remote")
        self.resizable(False, False)

        self.create_buttons()
        # Figure is created before the mainloop is started to keep the menu at the top.
        #  Not sure why this is needed, but now I can change the axis interactively
        # fig=plt.figure()
        # plt.close(fig)
        self.data1 = []
        self.data2 = []

    def create_buttons(self):
        # Vertical control colors
        self.vfg = {}
        # self.vfg[1] = "#fcfa23"  # yellow
        self.vfg[1] = "gold"  # when on white background
        self.vfg[2] = "#16f7fe"  # cyan
        self.vfg[3] = "#fe02fe"  # magenta
        self.vfg[4] = "#047efa"  # cyan-blue
        self.bfg = {}
        self.bfg[1] = "black"
        self.bfg[2] = self.bfg[1]
        self.bfg[3] = self.bfg[1]
        self.bfg[4] = self.bfg[1]
        self.vlabel = {}
        self.vlabel_val = {}
        vup = {}
        vdown = {}
        vframe = ttk.Labelframe(self, text="Vertical")
        for ch in range(1, 5):
            row = 0
            self.vlabel[ch] = tk.Button(master=vframe, command=partial(self.vertical_toggle, ch), text=f"CH{ch}")
            self.vlabel[ch]["fg"] = self.vfg[ch]
            self.vlabel[ch]["bg"] = self.bfg[ch]
            self.vlabel[ch].grid(column=ch, row=row)
            row = 1
            vup[ch] = tk.Button(master=vframe, command=partial(self.vertical, ch, True), text="^")
            vup[ch].grid(column=ch, row=row)
            vup[ch]["height"] = 1
            vup[ch]["width"] = 3
            row = 2
            self.vlabel_val[ch] = tk.Label(master=vframe, width=5)
            self.vlabel_val[ch].grid(column=ch, row=row)
            self.vlabel_val[ch]["width"] = 5
            # self.vlabel_val[ch].insert('1.0', "V/div")
            # self.vlabel_val[ch].bind('<Return>', partial(self.ReturnPressed, ch))
            row = 3
            vdown[ch] = tk.Button(master=vframe, command=partial(self.vertical, ch, False), text="v")
            vdown[ch].grid(column=ch, row=row)
            vdown[ch]["width"] = vup[ch]["width"]
            vdown[ch]["height"] = vup[ch]["height"]
            self.vlabel_update(ch)

            #fft
            row = 4
            fft_button = tk.Button(master=vframe, text="FFT", command=partial(self.plot_fft, ch))
            fft_button.grid(column=ch, row=row)
            fft_button["width"] = vup[ch]["width"]

            #save
            row = 5
            data_save = tk.Button(master=vframe, text="Save", command=partial(self.save_data, ch))
            data_save.grid(column=ch, row=row)
            data_save["width"] = vup[ch]["width"]

            # FFT window
            row =6
            paddings = {'padx': 5, 'pady': 5}
            label = ttk.Label(master=vframe, text='FFT window:')
            label.grid(column=1, row=row, columnspan=2, sticky=tk.W, **paddings)
            window_list = ['none', "flattop", "blackman", "hann", "kaiser"]
            self.window_var = tk.StringVar()
            self.window_var.set(window_list[4])
            fft_window = tk.OptionMenu(vframe, self.window_var,*window_list )
            fft_window.grid(column=3, row=row, columnspan=2, sticky=tk.W)

        # Horizontal controls
        hframe = ttk.Labelframe(self, text="Horizontal")
        hup = tk.Button(master=hframe, command=partial(self.horizontal, True), width=3, text="<")
        hup.grid(column=0, row=0)
        self.hlabel_val = tk.Label(master=hframe, height=1, width=5)
        self.hlabel_val.grid(column=1, row=0)
        # self.hlabel_val.insert('1.0', "s/div")
        hdown = tk.Button(master=hframe, command=partial(self.horizontal, False), width=3, text=">")
        hdown.grid(column=2, row=0)
        self.hlabel_update()

        # Trigger controls
        trigframe = ttk.Labelframe(self, text="Trigger")
        triggersource = tk.StringVar()
        for ch in range(1, 5):
            rb = ttk.Radiobutton(master=trigframe, text=f"CH{ch}", value=f"CH{ch}", variable=triggersource)
            rb['command'] = partial(self.setTriggerSource, ch)
            rb.grid(row=1, column=ch, sticky=tk.W)
        triggeredge = tk.StringVar()
        rb = ttk.Radiobutton(master=trigframe, text="Rising", value="POS", variable=triggeredge)
        rb['command'] = partial(self.setTriggeredge, "POS")
        rb.grid(row=2, column=1, sticky=tk.W)
        rb = ttk.Radiobutton(master=trigframe, text="Falling", value="NEG", variable=triggeredge)
        rb['command'] = partial(self.setTriggeredge, "NEG")
        rb.grid(row=2, column=2, sticky=tk.W)

        #Bode plot
        bodeframe = ttk.Labelframe(self, text="Bodeplot")
        ttk.Button(master=bodeframe, text="Bode Plot", command=partial(self.plot_bode)).grid(row=0, column=0,columnspan=2)
        ttk.Button(master=bodeframe, text="Pre capture", command=partial(self.plot_bode, pre_capture=True)).grid(row=1,column=0,columnspan=2)
        self.bodefmin = tk.StringVar()
        self.bodefmin.set("1")
        self.bodefmax = tk.StringVar()
        self.bodefmax.set("1e8")
        ttk.Label(master=bodeframe, text="Frequency range").grid(row=2, column=0, columnspan=2)
        ttk.Entry(master=bodeframe, width=6, textvariable=self.bodefmin).grid(row=3, column=0)
        ttk.Entry(master=bodeframe, width=6, textvariable=self.bodefmax).grid(row=3, column=1)
        ttk.Button(master=bodeframe, text="Setup scope", command=partial(self.bode_setup)).grid(row=4,column=0,columnspan=2)

        self.ampfilter = tk.StringVar()
        self.ampfilter.set("60")
        ttk.Label(master=bodeframe, text="dB below 0dBfs").grid(row=5, column=0, columnspan=2)
        ttk.Entry(master=bodeframe, width=5, textvariable=self.ampfilter).grid(row=6, column=0)

        vframe.grid(row=0, column=0, rowspan=3, sticky=tk.N)
        hframe.grid(row=0, column=1, sticky=tk.N)
        trigframe.grid(row=4, column=0)
        bodeframe.grid(row=1, column=1, rowspan=3, sticky=tk.W)

        self.warning = ttk.Label(text=version)
        self.warning.grid(row=4, column=1, sticky=tk.EW)

    # def ReturnPressed(self, ch, event):
    #     text = self.vlabel_val[ch]
    #     print(text.get("1.0", tk.END))

    def plot_bode(self, pre_capture=False):
        instr = self.instr
        if self.debug == 2:
            with open("debug.bplt", "rb") as f:
                self.data1, self.data2 = pickle.load(f)
        else:
            instr.write(":STOP")
            self.data1.append(self.get_data(1))
            self.data2.append(self.get_data(2))
            instr.write(":RUN")

        if pre_capture:
            self.warning['text'] = f"Capture # {len(self.data1)}"
            return

        # merge all runs (ywf data)
        fft1 = np.zeros(1)
        fft2 = np.zeros(1)
        window=self.window_var.get()
        for data in self.data1:
            xf, ywf = self.fft_trace(data,window)
            if len(fft1) == 1:
                fft1 = np.asarray(ywf)
            else:
                #TODO: xf can also change between captures. This is not handled at the moment.
                fft1 = np.vstack((fft1, ywf))
        for data in self.data2:
            xf, ywf = self.fft_trace(data,window)
            if len(fft2) == 1:
                fft2 = np.asarray(ywf)
            else:
                fft2 = np.vstack((fft2, ywf))

        mode = 'fftmax'
        if mode == 'fftmean':
            fft_val1 = np.mean(fft1, axis=0)
            fft_val2 = np.mean(fft2, axis=0)
        else:
            if len(self.data1) == 1:
                fft_val1 = fft1
                fft_val2 = fft2
            else:
                max_indices = np.argmax(np.abs(fft1), axis=0)
                fft_val1 = fft1[max_indices, range(fft1.shape[1])]
                fft_val2 = fft2[max_indices, range(fft2.shape[1])]
        Vrms1 = 2.0 / data["N"] * abs(fft_val1) / np.sqrt(2)
        Vrms2 = 2.0 / data["N"] * abs(fft_val2) / np.sqrt(2)
        if self.loopgain:
            # Note the minus sign is left out such that 0 degrees reads as no margin.
            s21 = (fft_val1 / fft_val2)
        else:
            s21 = (fft_val2 / fft_val1)
        PhaseDiff = np.rad2deg(np.angle(s21))

        data1 = self.data1[0]
        data2 = self.data2[0]
        # plt.style.use('dark_background')
        fig, ax = plt.subplots(3, layout="constrained", figsize=(6, 10))
        ax[0].plot(data1["x"], data1["y"], self.vfg[1],
                   data2["x"], data2["y"], self.vfg[2])
        ax[0].legend([f"CH1", f"CH2"], loc='lower left')
        ax[0].grid(True)

        # ax[1].cla()
        ax[1].semilogx(xf, 20 * np.log10(Vrms1), self.vfg[1],
                       xf, 20 * np.log10(Vrms2), self.vfg[2] )

        #Only plot those points that have a reasonable amplitude
        # max full scale amplitude. For rigol 16 bits vertical resolution. so amplitude is 15 bits
        maxv = (self.data1[-0]["ymax"] - self.data1[-0]["ymin"]) / 2 * np.sqrt(0.5)
        minv = maxv * 10 ** (-float(self.ampfilter.get()) / 20)
        xmask = Vrms1 > minv

        argmax = Vrms1.argmax()
        # Filter out the first 50 odd harmonics
        if argmax > 0 :
            harm_no = 50
            xmask_harmonics=np.ndarray.copy(xmask)
            xmask_harmonics[:harm_no * argmax] = False
            argmax_real = np.average(np.arange(argmax-1,argmax+2), weights=Vrms1[argmax-1:argmax+2])
            xmask_harmonics[np.rint(np.arange(1, harm_no, 2) * (argmax_real + 1)).astype(np.int64) - 1] = True
            # Make sure the harmonic are really above the threshold
            xmask=np.logical_and(xmask, xmask_harmonics)
        else:
            True
            xmask[1::2] = False
        # Try to estimate the real bin using the two neighbour bins
        # xmask=xmask_harmonics
        # xmask[xf < fmax] = False

        # Use range input from user
        xmask[xf < float(self.bodefmin.get())] = False
        xmask[xf > float(self.bodefmax.get())] = False

        xf=xf[xmask]

        # bodedb = 20 * np.log10(Vrms2[xmask]) - 20 * np.log10(Vrms1[xmask])
        bodedb = 20 * np.log10(np.abs(s21))[xmask]
        ax[2].cla()
        ax[2].semilogx(xf, bodedb, "-",zorder=2)
        # if not hasattr(self, 'ax2b'):
        #     pass
        ax2b = ax[2].twinx()
        bodephase = np.unwrap(PhaseDiff[xmask], period=360)
        ax[2].set_ylim([min(bodedb), max(bodedb)])
        ax2b.semilogx(xf, bodephase, "-", color='orange', zorder=1)
        ax[2].grid(True)
        from matplotlib.ticker import MultipleLocator
        if max(bodephase) - min(bodephase) < 500:
            ax2b.yaxis.set_major_locator(MultipleLocator(45))
        if max(bodephase) - min(bodephase) < 90:
            ax2b.set_ylim([min(bodephase) - 45, max(bodephase) + 45])

        #Put phase is the background
        ax[2].set_zorder(ax[2].get_zorder() + 1)
        ax[2].set_frame_on(False)

        if self.loopgain:
            if np.where(bodedb < 0)[0].size:
                bodedb_belowzero_index = np.where(bodedb < 0)[0][0]
                belowzero_freq = xf[bodedb_belowzero_index]
                print(f"Loop gain is below zero at {belowzero_freq:.0f} Hz")
                belowzero_phase = bodephase[bodedb_belowzero_index]
                print(f"Loop phase is  {belowzero_phase:.2f}")
                self.warning["text"] = f"Ph margin {belowzero_phase:.2f}"
                ax2b.annotate(f"PM={belowzero_phase:.1f}",xy=(belowzero_freq,belowzero_phase),
                              xytext=(belowzero_freq,belowzero_phase+20), arrowprops=dict(arrowstyle="->"))
                ax[2].annotate(f"f={belowzero_freq:.0f}", xy=(belowzero_freq, 0),
                      xytext=(belowzero_freq, 0 + 10), arrowprops=dict(arrowstyle="->"))
        # labels
        ax[0].set_xlabel("Time (s)")
        ax[1].set_xlabel("Freq (Hz)")
        ax[2].set_xlabel("Freq (Hz)")
        ax[0].set_ylabel("V")
        ax[1].set_ylabel("dBV")
        ax[2].set_ylabel('Gain (dB)', color='blue')
        ax2b.set_ylabel('Phase (Deg)', color='orange')

        ax[2].set_title("Bode plot")

        ax[1].grid(True)

        if self.debug == 1:
            try:
                date = datetime.datetime.now()
                with open(date.strftime("%Y-%m-%d-%X.bplt"), "wb") as f:
                    pickle.dump((self.data1, self.data2), f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as ex:
                print("Error during pickling object (Possibly unsupported):", ex)

        binwidth = data1["sr"] / data1["mdepth"]
        print(f'Sample rate={data["sr"] / 1e6:g} Ms')
        print(f'FFT max freq={data["sr"] / 2e6:g}MHz')
        print(f"NOTE: Apply frequency below {binwidth} Hz\n")

        self.data1 = []
        self.data2 = []
        # self.warning['text'] = f"OK"
        plt.show()

    def select_array(self, a, b, select):
        # if select use element from b, else a
        a[select] = 0
        b[~select] = 0
        a = a + b
        return a

    def setTriggeredge(self, slope):
        self.instr.write(f":TRIGger:EDGE:SLOPe {slope}")

    def setTriggerSource(self, ch):
        self.instr.write(f":TRIGger:EDGE:SOURce CHAN{ch}")

    def vertical(self, ch, up):
        scale = float(self.instr.query(f":CHANnel{ch}:SCale?"))
        if up:
            # I want a 1:2:5 sequence
            scale = format(scale * 2.251, '.1g')
        else:
            scale = format(scale / 2.101, '.1g')
        self.instr.write(f":CHANnel{ch}:Scale {scale}")
        self.vlabel_update(ch)

    def vlabel_update(self, ch):
        text = self.vlabel_val[ch]
        try:
            scale = float(self.instr.query(f":CHANnel{ch}:SCale?"))
        except:
            text["text"] = "?"
            return
        scale_unit = ""
        if scale < 1e-3:
            scale = int(scale * 1e6)
            scale_unit = "uV"
        else:
            if scale < 1:
                scale = int(scale * 1e3)
                scale_unit = "mV"
            else:
                scale = int(scale)
                scale_unit = "V"
        # text.delete("1.0",tk.END)
        # text.insert("1.0",f"{scale}{scale_unit}")
        text["text"] = f"{scale}{scale_unit}"

    def vertical_toggle(self, ch):
        display = not float(self.instr.query(f":CHANnel{ch}:DISPLAY?"))
        self.instr.write(f":CHANnel{ch}:DISPLAY {display}")
        if display:
            self.vlabel[ch]["fg"] = self.vfg[ch]
        else:
            self.vlabel[ch]["fg"] = "gray"

    def horizontal(self, up):
        scale = float(self.instr.query(f":TIMebase:SCale?"))
        if up:
            # I want a 1:2:5 sequence
            scale = format(scale * 2.251, '.1g')
        else:
            scale = format(scale / 2.101, '.1g')
        self.instr.write(f":TIMebase:SCale {scale}")
        self.hlabel_update()

    def hlabel_update(self):
        text = self.hlabel_val
        try:
            scale = float(self.instr.query(f":TIMebase:SCale?"))
        except:
            text["text"] = "?"
            return
        if scale < 1e-6:
            scale = int(scale * 1e9)
            scale_unit = "ns"
        elif scale < 1e-3:
            scale = int(scale * 1e6)
            scale_unit = "us"
        elif scale < 1:
            scale = int(scale * 1e3)
            scale_unit = "ms"
        else:
            scale_unit = "s"
        # text.delete("1.0", tk.END)
        # text.insert("1.0", f"{scale}{scale_unit}")
        text["text"] = f"{scale}{scale_unit}"

    def plot_fft(self, ch):
        if self.debug == 2:
            with open("debug.bplt", "rb") as f:
                self.data1, self.data2 = pickle.load(f)
                if ch == 2 :
                    data=self.data2[0]
                else:
                    data=self.data1[0]
        else:
            self.instr.write(":STOP")
            try:
                data = self.get_data(ch)
            except ValueError:
                self.warning["text"] = f"Got no data from CH{ch}"
                print("No data retrieved from scope")
                return 1
            except ChannelNotEnabled:
                self.warning["text"] = f"Channel {ch} is not enabled"
                print(f"Channel {ch} is not enabled")
                return 1
            self.instr.write(":RUN")
        print(f'Sample rate={data["sr"] / 1e6:g} Ms')
        print(f'FFT max freq={data["sr"] / 2e6:g}MHz')
        binwidth = data["sr"] / data["mdepth"]
        # If the FFT has been windowed using Hanning the noise bandwidth needs to be corrected
        hann_nbw_correction_db = np.log10(1.5)
        noiseBW = 10 * np.log10(binwidth) + hann_nbw_correction_db
        print(f"FFT min freq={binwidth:g}Hz")
        print(f"FFT bin noise BW={noiseBW:.1f}db")
        print_noise = True
        if data["bwl"] == '20M':
            if data["sr"] < 40e6:
                print_noise = False
                print("Noise is not accurate due to noise folding, please increase sample rate!")
        else:
            if data["sr"] < 500e6:
                print_noise = False
                print("Noise is not accurate due to noise folding, please increase sample rate or set BW to 20MHz!")

        def moving_average(x, w):
            bmw = blackman(w)
            # noinspection PyTypeChecker
            return np.convolve(x, bmw, 'same') / (sum(bmw) / len(bmw))

        xf, ywf = self.fft_trace(data,window=self.window_var.get())
        Vrms = 2.0 / data["N"] * abs(ywf) / np.sqrt(2)
        avg_size = min(500, int(data["N"] / 10))
        print(f'Vrms={np.sqrt(sum(data["y"] ** 2) / data["N"])}')
        Vavg = moving_average(Vrms ** 2, avg_size * 2)
        Vavg = (Vavg / avg_size / 2) ** 0.5

        fig, ax = plt.subplots(3, layout="constrained", figsize=(6, 10))
        if print_noise:
            ax[0].semilogy(xf, Vrms, self.vfg[ch],
                           xf[avg_size:], (Vavg[avg_size:] / 10 ** (noiseBW / 20)))
        else:
            ax[0].semilogy(xf, Vrms, self.vfg[ch])
        ax[0].legend([f"CH{ch} RBW={binwidth:g}Hz", f"Noise V/sqrt(Hz)"], loc='lower left')
        ax[0].grid(True)
        ax[0].set_xlabel("Freq F(Hz)")
        ax[0].set_ylabel("V")

        if print_noise:
            ax[1].semilogx(xf, 20 * np.log10(Vrms), self.vfg[ch],
                           xf[avg_size:], 20 * np.log10(Vavg[avg_size:]) - noiseBW, marker="")
        else:
            ax[1].semilogx(xf, 20 * np.log10(Vrms), self.vfg[ch])
        ax[1].grid(True)
        ax[1].set_xlabel("Freq (Hz)")
        ax[1].set_ylabel("dBV")
        ax[1].legend([f"CH{ch} RBW={binwidth:g}Hz", f"Noise V/sqrt(Hz)"], loc='lower left')

        ax[2].plot(data["x"], data["y"], self.vfg[ch])
        ax[2].grid(True)
        ax[2].set_xlabel("time (s)")
        ax[2].set_ylabel("V")
        # plt.ion()
        # plt.pause(.001)
        plt.show()

    def fft_trace(self, data, window="flat"):
        # 1kHz scope calibration is 3Vpp => db20(2*3/pi)=5.62dBVa and 2.62dBVrms
        if window == "flat" or window == "none":
            win = np.ones(data["N"])
        elif window == "flattop":
            win = flattop(data["N"])
        elif window == "blackman":
            win = blackman(data["N"])
        elif window == "kaiser":
            win = kaiser(data["N"], beta=14)
        elif window == "hann":
            win = hann(data["N"])
        else:
            print("Unknown window")
        # noinspection PyTypeChecker
        cpg = sum(win) / len(win)  # coherent power gain
        # ywf = fft(data["y"] * win / cpg)
        # xf = fftfreq(data["N"], data["xincr"])[:data["N"] // 2]
        y_windowed=data["y"] * win / cpg
        y_windowed=y_windowed - np.mean(y_windowed)
        ywf = rfft(y_windowed)
        xf = rfftfreq(data["N"], data["xincr"])
        return xf[1:], ywf[1:]

    def save_data(self, ch, file=""):
        self.instr.write(":STOP")
        try:
            data = self.get_data(ch)
        except ValueError:
            self.warning["text"] = f"Got no data from CH{ch}"
            print("No data retrieved from scope")
            return 1
        except ChannelNotEnabled:
            self.warning["text"] = f"Channel {ch} is not enabled"
            print(f"Channel {ch} is not enabled")
            return 1
        self.instr.write(":RUN")
        csv_data = np.transpose(np.stack((data["x"], data["y"])))
        np.savetxt(f"CH{ch}.csv", csv_data, delimiter=',')
        self.warning["text"] = f"Saved to CH{ch}.csv"


if __name__ == '__main__':
    import argparse
    epilog='''Connect CH1 to the generator side and CH2 to the internal. The Bode plot shows CH1/CH2. 
    For loopgain analysis connect CH1 to the output (large signal) and 
    CH2 to the error signal (small signal). Now the Bode plot shows CH2/CH1'''


    parser = argparse.ArgumentParser(description="Remote control your oscilloscope", epilog=epilog)
    parser.add_argument("-v", "--version", action="store_true", help="show version")
    parser.add_argument("-i", "--ip", default="192.168.1.8", help="IP or hostname of oscilloscope")
    parser.add_argument("-l", "--loopgain", action="store_true", help="For loopgain analysis")
    parser.add_argument("-m", "--scope_brand", default="rigol", help="scope brand")
    parser.add_argument("-d", "--debug", choices=[0,1,2], type=int, help="0=no debug, 1=save data, 2=read data from debug.bplt")
    args = parser.parse_args()
    if args.scope_brand == "rigol":
        rm = Scope()
    elif args.scope_brand == "siglent":
        rm = Siglent()
    else:
        print("Unkown model")
        quit(1)

    if args.version:
        print(f"dho-remote {version}")
        quit(0)
    rm.debug = args.debug
    rm.scope_init(args.ip)
    ui = ScopeUI(rm)
    ui.debug = args.debug
    ui.loopgain = args.loopgain
    ui.mainloop()
    rm.close()
