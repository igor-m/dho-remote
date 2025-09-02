#!/usr/bin/env python3

#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Modded by Igor-M aka iMo 2025/08/28 for Noise Analysis
#  v 1.5 MOD_06
#  31st August 2025
#  https://github.com/igor-m/dho-remote
#  more on eevblog:
#  https://www.eevblog.com/forum/metrology/simple-noise-analyser-for-scope-owners-who-do-not-have-that-option/msg6028551/#msg6028551
#  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import tkinter as tk
import tkinter.ttk as ttk
from scipy.signal.windows import flattop, blackman, hann, kaiser
from scipy.fft import rfft, rfftfreq
from functools import partial
import pickle
from datetime import datetime
import time

from scope.rigol import Scope, Siglent, ChannelNotEnabled

version = "v1.5 Mod_06 by Igor-M 09/2025"

class ScopeUI(tk.Tk):
    # data1: np.array
    # data2: np.array

    def __init__(self, rm):
        super().__init__()
        # self.geometry("240x200")
        self.demo = None
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

# Modded by Igor-M 08/2025
    def create_buttons(self):
        # --- Vertical controls ---
        self.vfg = {1: "gold", 2: "#16f7fe", 3: "#fe02fe", 4: "#047efa"}
        self.bfg = {1: "black", 2: "black", 3: "black", 4: "black"}
        self.vlabel, self.vlabel_val = {}, {}
        vup, vdown = {}, {}

        vframe = ttk.Labelframe(self, text="Vertical/div / Channels")

        for ch in range(1, 5):
            # channel label button
            self.vlabel[ch] = tk.Button(vframe, command=partial(self.vertical_toggle, ch), text=f"CH{ch}")
            self.vlabel[ch].configure(fg=self.vfg[ch], bg=self.bfg[ch], width=6)
            self.vlabel[ch].grid(column=ch, row=0, padx=2, pady=2)

            # up button
            vup[ch] = tk.Button(vframe, command=partial(self.vertical, ch, True), text="^", width=6)
            vup[ch].grid(column=ch, row=1, padx=2, pady=2)

            # label value
            self.vlabel_val[ch] = tk.Label(vframe, width=6, anchor="center")
            self.vlabel_val[ch].grid(column=ch, row=2, padx=2, pady=2)

            # down button
            vdown[ch] = tk.Button(vframe, command=partial(self.vertical, ch, False), text="v", width=6)
            vdown[ch].grid(column=ch, row=3, padx=2, pady=2)

            self.vlabel_update(ch)

            # FFT + Save buttons
            tk.Button(vframe, text="FFT", width=6, command=partial(self.plot_fft, ch)).grid(column=ch, row=5, padx=2, pady=2)
            tk.Button(vframe, text="Noise", width=6, command=partial(self.plot_fftN, ch)).grid(column=ch, row=6, padx=2, pady=2)
            tk.Button(vframe, text="Save", width=6, command=partial(self.save_data, ch)).grid(column=ch, row=4, padx=2, pady=2)

        # --- Noise analysis frame (FFT window + averaging + gain + filters) ---
        noise_frame = ttk.Labelframe(self, text="Noise Analysis")

        row = 0
        window_list = ["FlatTop", "Blackman", "Hann", "Kaiser", "None"]
        self.window_var = tk.StringVar(value="None")  # default
        ttk.Label(noise_frame, text="FFT window:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=2)
        ttk.OptionMenu(noise_frame, self.window_var, self.window_var.get(), *window_list).grid(
            column=1, row=row, sticky=tk.W, padx=5, pady=2)
        
        row = 1
        self.nshots = tk.StringVar(value="1")
        ttk.Label(noise_frame, text="FFT Averaging (N):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(noise_frame, width=12, textvariable=self.nshots).grid(column=1, row=row, padx=5, pady=2)

        row = 2
        self.lna_gain = tk.StringVar(value="10000")
        ttk.Label(noise_frame, text="LNA Gain (A):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(noise_frame, width=12, textvariable=self.lna_gain).grid(column=1, row=row, padx=5, pady=2)

        row = 3
        self.filter_from = tk.StringVar(value="0.1")
        ttk.Label(noise_frame, text="LNA Filter from (Hz):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(noise_frame, width=12, textvariable=self.filter_from).grid(column=1, row=row, padx=5, pady=2)

        row = 4
        self.filter_to = tk.StringVar(value="10")
        ttk.Label(noise_frame, text="LNA Filter to (Hz):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(noise_frame, width=12, textvariable=self.filter_to).grid(column=1, row=row, padx=5, pady=2)

        # LNA Filter Shape Factor
        # Filter Order selector
        row = 5
        order_list = ["2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5", "6", "6.5", "7", "7.5", "8" ]
        self.filter_order = tk.StringVar(value=order_list[2])  # default = 4th
        ttk.Label(noise_frame, text="Eff. Filter Order HP+LP:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=2)
        ttk.OptionMenu(noise_frame, self.filter_order, self.filter_order.get(), *order_list).grid(column=1, row=row, sticky=tk.W, padx=5, pady=2)

        # Measurement ID name for saving data
        row = 6
        self.meas_id = tk.StringVar()
        self.meas_id.set("DUT#1")  # default ID
        ttk.Label(master=noise_frame, text="Measurement ID:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(master=noise_frame, width=16, textvariable=self.meas_id).grid(column=1, row=row, sticky=tk.W, padx=5, pady=2)
        
        # Save checkbox
        row = 12
        self.save_data_flag = tk.BooleanVar(value=False)  # default: not ticked
        ttk.Checkbutton(
            master=noise_frame,
            text="Save .csv",
            variable=self.save_data_flag
        ).grid(column=0, row=row, columnspan=2, sticky=tk.W, padx=5, pady=2)


        # --- Horizontal controls ---
        hframe = ttk.Labelframe(self, text="Horizontal/div")
        tk.Button(hframe, command=partial(self.horizontal, True), width=3, text="<").grid(column=0, row=0, padx=2, pady=2)
        self.hlabel_val = tk.Label(hframe, width=6)
        self.hlabel_val.grid(column=1, row=0, padx=2, pady=2)
        tk.Button(hframe, command=partial(self.horizontal, False), width=3, text=">").grid(column=2, row=0, padx=2, pady=2)
        self.hlabel_update()

        # --- Trigger controls ---
        trigframe = ttk.Labelframe(self, text="Trigger")
        triggersource = tk.StringVar()
        for ch in range(1, 5):
            rb = ttk.Radiobutton(trigframe, text=f"CH{ch}", value=f"CH{ch}", variable=triggersource,
                                 command=partial(self.setTriggerSource, ch))
            rb.grid(row=1, column=ch, sticky=tk.W, padx=2, pady=2)
        triggeredge = tk.StringVar()
        ttk.Radiobutton(trigframe, text="Rising", value="POS", variable=triggeredge,
                        command=partial(self.setTriggeredge, "POS")).grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Radiobutton(trigframe, text="Falling", value="NEG", variable=triggeredge,
                        command=partial(self.setTriggeredge, "NEG")).grid(row=2, column=2, sticky=tk.W, padx=2, pady=2)

        # --- Bode plot controls ---
        bodeframe = ttk.Labelframe(self, text="Bodeplot")
        ttk.Button(bodeframe, text="Bode Plot", command=partial(self.plot_bode)).grid(row=0, column=0, columnspan=2, pady=2)
        ttk.Button(bodeframe, text="Pre capture", command=partial(self.plot_bode, pre_capture=True)).grid(row=1, column=0, columnspan=2, pady=2)
        self.bodefmin, self.bodefmax = tk.StringVar(value="1"), tk.StringVar(value="1e8")
        ttk.Label(bodeframe, text="Frequency range").grid(row=2, column=0, columnspan=2, pady=2)
        ttk.Entry(bodeframe, width=6, textvariable=self.bodefmin).grid(row=3, column=0, pady=2)
        ttk.Entry(bodeframe, width=6, textvariable=self.bodefmax).grid(row=3, column=1, pady=2)
        ttk.Button(bodeframe, text="Setup scope", command=partial(self.bode_setup)).grid(row=4, column=0, columnspan=2, pady=2)
        self.ampfilter = tk.StringVar(value="60")
        ttk.Label(bodeframe, text="dB below 0dBfs").grid(row=5, column=0, columnspan=2, pady=2)
        ttk.Entry(bodeframe, width=5, textvariable=self.ampfilter).grid(row=6, column=0, pady=2)

        # --- Layout placement ---
        vframe.grid(row=0, column=0, rowspan=3, sticky=tk.N, padx=5, pady=5)
        noise_frame.grid(row=3, column=0, sticky=tk.EW, padx=5, pady=5)
        hframe.grid(row=0, column=1, sticky=tk.N, padx=5, pady=5)
        trigframe.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        bodeframe.grid(row=2, column=1, rowspan=2, sticky=tk.W, padx=5, pady=5)

        # --- Version / warning label ---
        self.warning = ttk.Label(text=version)
        self.warning.grid(row=4, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)


    # def ReturnPressed(self, ch, event):
    #     text = self.vlabel_val[ch]
    #     print(text.get("1.0", tk.END))

    def plot_bode(self, pre_capture=False):
        instr = self.instr
        if self.demo and not self.demo == "save" :
            with open(self.demo, "rb") as f:
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
                # select from the list of fft values the maximum
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
            vd=np.diff(Vrms1)
            max_indices=(vd[:-1]>0) & (vd[1:]<0)
            max_indices=np.insert(max_indices,0,[False,False])
            xmask[np.logical_not(max_indices)] = False
        else:
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

        PhaseDiff = np.rad2deg(np.angle(s21))
        bodephase = np.unwrap(PhaseDiff[xmask], period=360)

        ax[2].set_ylim([min(bodedb), max(bodedb)])
        ax2b.semilogx(xf, bodephase, "-", color='orange', zorder=1)
        ax[2].grid(True)
        if max(bodephase) - min(bodephase) < 500:
            ax2b.yaxis.set_major_locator(MultipleLocator(45))
        if max(bodephase) - min(bodephase) < 90:
            ax2b.set_ylim([min(bodephase) - 45, max(bodephase) + 45])

        #Put phase is the background
        ax[2].set_zorder(ax[2].get_zorder() + 1)
        ax[2].set_frame_on(False)

        if self.loopgain:
            # Find the phase margin
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

        if self.demo == "save":
            try:
                date = datetime.datetime.now()
                with open(date.strftime("%Y-%m-%d-%X.bplt"), "wb") as f:
                    pickle.dump((self.data1, self.data2), f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as ex:
                print("Error during pickling object (Possibly unsupported):", ex)

        binwidth = data1["sr"] / data1["mdepth"]
        print(f'Sample rate={data["sr"] / 1e6:g} Ms')
        print(f'FFT max freq={data["sr"] / 2e6:g}MHz')

        # clear the data since we are going to plot
        self.data1 = []
        self.data2 = []
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
        if self.demo and not self.demo == "save":
            with open(self.demo, "rb") as f:
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
        if data["bwl"] == '20M':
            if data["sr"] < 40e6:
                print_noise = False
                print("Noise is not accurate due to noise folding, please increase sample rate!")
        else:
            if data["sr"] < 500e6:
                print_noise = False
                print("Noise is not accurate due to noise folding, please increase sample rate or set BW to 20MHz!")

        print_noise = True
        def moving_average(x, w):
            bmw = blackman(w)
            # noinspection PyTypeChecker
            return np.convolve(x, bmw, 'same') / (sum(bmw) / len(bmw))

        xf, ywf = self.fft_trace(data,window=self.window_var.get())
        Vrms = 2.0 / data["N"] * abs(ywf) / np.sqrt(2)
        avg_size = min(500, int(data["N"] / 10))
        print(f'Vrms={np.sqrt(sum(data["y"] ** 2) / data["N"]):.3g} V')
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


# Added Noise Analysis 08/2025 by Igor-M
    def plot_fftN(self, ch):
        # === acquire data ===
        # NEW Igor_M
        # Ensure Noise_Measurements folder exists
        self.save_dir = os.path.join(os.getcwd(), "Noise_Measurements")
        os.makedirs(self.save_dir, exist_ok=True)
        
        Nfft = min(max(int(self.nshots.get()), 1), 1000)  # clamp 1–1000
        spectra = []
        data = None
        for i in range(Nfft):
            self.instr.write(":SINGle")
            print(f"N.{i+1} ({Nfft}) FFT from CH{ch} ..")
            # wait until acquisition complete
            timeout = time.time() + 10000.0  # debug only: timeout per shot in seconds
            while True:
                time.sleep(1)
                status = self.instr.query(":TRIGger:STATus?").strip().upper()
                # DHO can answer RUN, AUTO, WAIT, TD, STOP
                if status == "STOP":
                    print(f"N.{i+1} FFT DONE from CH{ch} !")
                    break  # acquisition done
                if time.time() > timeout:
                    print("Timeout waiting for single shot trigger, proceeding anyway.")
                    break
            
            try:
                data = self.get_data(ch)
            except (ValueError, ChannelNotEnabled):
                self.warning["text"] = f"No data from CH{ch}"
                print(f"No data retrieved from CH{ch}")
                self.instr.write(":RUN")
                return 1
            
            self.instr.write(":RUN")
            
            # Save current data with name Measurement_ID and timestamp
            if self.save_data_flag.get():  # only save if checked
                meas_id_name = self.meas_id.get().strip().replace(" ", "_")
                # Timestamp with milliseconds
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # truncate to milliseconds
                # Filename in Noise_Measurements folder
                filename = os.path.join(self.save_dir, f"{meas_id_name}_{timestamp}.csv")

                # Save data
                csv_data = np.transpose(np.stack((data["x"], data["y"])))
                np.savetxt(filename, csv_data, delimiter=',')
                rel_path = os.path.relpath(filename, start=os.getcwd())
                self.warning["text"] = f"Saved to {rel_path}"
                print(f"Saved data to {rel_path}")
            else:
                self.warning["text"] = "Data not saved (Save unchecked)"
            
            
            gain = float(self.lna_gain.get())   # linear gain (not dB!)
            data["y"] = data["y"] / gain
            xf, ywf_single, wenbw = self.fft_traceN(data, window=self.window_var.get())
            spectra.append(np.abs(ywf_single) ** 2) 
            
        # average power spectrum
        mean_power = np.mean(spectra, axis=0)
        # convert back to magnitude (Vrms/bin)
        ywf = np.sqrt(mean_power)

        print(f"Samples read: {Nfft} * {data['N']}")
        print(f'Sample rate={data["sr"] / 1e3:g} ksps')
        print(f'FFT max freq={data["sr"] / 2e3:g} kHz')

        # === RBW / ENBW / noise BW ===
        binwidth = data["sr"] / data["mdepth"]     # Hz per FFT bin
        # Determine ENBW depending on window
        noiseBW_dbHz = 10.0 * np.log10(binwidth * wenbw)   # dB (10*log10 of the equivalent noise bandwidth)
        print(f"FFT min freq={binwidth:g} Hz")
        print(f"FFT bin ENBW={noiseBW_dbHz:.2f} dB-Hz")

        print_noise = True

        # === FFT trace & per-bin RMS voltage ===
        Vrms =  2.0 / data["N"] * (ywf)  / np.sqrt(2.0)              # V RMS per bin (raw bin RMS)

        # === Noise Spectral Density (V/√Hz) and conversions ===
        # NSD = Vavg / sqrt(binwidth * ENBW)
        NSD_V_per_rtHz = Vrms / np.sqrt(binwidth * wenbw)
        NSD_uV_per_rtHz = NSD_V_per_rtHz * 1e6
        # guard for log
        tiny = 1e-30
        NSD_dB_uV = 20.0 * np.log10(np.maximum(NSD_uV_per_rtHz, tiny))


        # === Integrated RMS noise (from low freq up) ===
        # power per bin = (NSD^2) * binwidth  -> sum -> sqrt -> Vrms        
        self.shape_factors = {
            "2": 1.57,
            "2.5": 1.43,
            
            "3": 1.30,
            "3.5": 1.20,
            
            "4": 1.11,
            "4.5": 1.09,
            
            "5": 1.07,
            "5.5": 1.06,
            
            "6": 1.05,
            "6.5": 1.04,
            
            "7": 1.03,
            "7.5": 1.0275,
            
            "8": 1.025,
        }

        shape_factor = self.shape_factors.get(self.filter_order.get(), 1.0)
        effective_binwidth = binwidth * wenbw * shape_factor
        noise_power_per_bin = (NSD_V_per_rtHz ** 2) * effective_binwidth
        cumulative_power = np.cumsum(noise_power_per_bin)   # V^2
        cumulative_rms = np.sqrt(cumulative_power)          # V
        cumulative_rms_uV = cumulative_rms * 1e6            # µV RMS
        
        # === Filtered integration and slope ===
        fmin = float(self.filter_from.get())
        fmax = float(self.filter_to.get())

        mask = (xf >= fmin) & (xf <= fmax)

        if np.any(mask):
            # Integrated Vrms in the band
            noise_power_band = (NSD_V_per_rtHz[mask] ** 2) * effective_binwidth
            Vrms_band = np.sqrt(np.sum(noise_power_band)) * 1e6  # µV

            print(f"Integrated Vrms from {fmin:g} Hz to {fmax:g} Hz: {Vrms_band:.3f} µV")

            # Linear fit slope of PSD = NSD**2 in µV^2/Hz vs log10(f)
            freqs_log = np.log10(xf[mask])
            nsd_db = NSD_dB_uV[mask]
            slope, intercept = np.polyfit(freqs_log, nsd_db*2, 1)
            
            alpha = -slope / 10.0

            print(f"PSD slope between {fmin:g}–{fmax:g} Hz: {slope:.3f} dB/decade")
            #print(f"Equivalent 1/f^alpha exponent: alpha = {alpha:.3f}")
            
            if abs(alpha) < 0.25:
                noise_type = "White (Thermal/Shot)"
            elif 0.6 <= alpha <= 1.4:
                noise_type = "Flicker"
            elif 1.6 <= alpha <= 2.4:
                noise_type = "Random walk"
            elif 2.6 <= alpha <= 3.4:
                noise_type = "Black"
            else:
                noise_type = f"Unclassified.."

            print(f"Est. Noise type (α ≈ {alpha:.2f}): {noise_type}")

        else:
            print(f"No data points in selected filter range {fmin}–{fmax} Hz")
            
        # === thermal noise reference lines (50Ω at 25°C) ===
        k = 1.380649e-23
        T = 298.15
        R = 50.0
        thermal_v_oc = np.sqrt(4 * k * T * R)   # open-circuit V/√Hz (~0.9 nV/√Hz)
        thermal_v_matched = thermal_v_oc / 2.0  # if source+load 50Ω matched (measured across load)
        thermal_uV_oc = thermal_v_oc * 1e6
        thermal_uV_matched = thermal_v_matched * 1e6
        ref_line_open = np.full_like(xf, thermal_uV_oc)
        #ref_line_matched = np.full_like(xf, thermal_uV_matched)
        ref_line_open_db = 20.0 * np.log10(np.maximum(ref_line_open, tiny))
        #ref_line_matched_db = 20.0 * np.log10(np.maximum(ref_line_matched, tiny))

        # === choose start index for plotting: skip DC (index 0) but keep very low freqs ===
        #start_idx = 1 if len(xf) > 1 else 0
        start_idx = 0
        stop_idx = len(xf) - 1

        # === Plotting ===
        fig, ax = plt.subplots(3, 1, figsize=(8, 11), layout="constrained")

        # NSD in µV/√Hz (log-log axis)
        if print_noise:
            ax[0].loglog(xf[start_idx:stop_idx], NSD_uV_per_rtHz[start_idx:stop_idx], label="Noise density [µV/√Hz]")
            ax[0].plot(xf, ref_line_open, 'k--', label=f"Thermal (50\u03A9) ≈ {thermal_uV_oc*1e3:.3f} nV/√Hz")
            #ax[0].plot(xf, ref_line_matched, 'r:', label=f"Thermal (matched) ≈ {thermal_uV_matched*1e3:.3f} nV/√Hz")
            ax[0].legend(loc='lower left')
        else:
            ax[0].plot(xf, ref_line_open, 'k--', label="Thermal reference")
            ax[0].legend(loc='lower left')
        ax[0].grid(True, which="both")
        ax[0].set_xlabel("Frequency (Hz)")
        ax[0].set_ylabel("µV/√Hz")

        # NSD in dBµV/√Hz (semilog x)
        if print_noise:
            ax[1].semilogx(xf[start_idx:stop_idx], NSD_dB_uV[start_idx:stop_idx], label="Noise density [dBµV/√Hz]")
            ax[1].semilogx(xf, ref_line_open_db, 'k--', label="Thermal (50\u03A9) dBµV/√Hz")
            #ax[1].semilogx(xf, ref_line_matched_db, 'r:', label="Thermal (matched) dBµV/√Hz")
            ax[1].legend(loc='lower left')
        else:
            ax[1].semilogx(xf, ref_line_open_db, 'k--', label="Thermal reference dB")
            ax[1].legend(loc='lower left')
        ax[1].grid(True, which="both")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("dBµV/√Hz")

        # Integrated RMS noise in µV RMS vs frequency
        ax[2].semilogx(xf[start_idx:], cumulative_rms_uV[start_idx:], label=f"Integrated Vrms CH{ch} (µV)")
        # Also show theoretical integrated thermal noise for comparison:
        # thermal power per bin = (v_oc^2) * binwidth  (open-circuit across resistor; if matched you'd halve appropriately)
        thermal_power_per_bin_open = (thermal_v_oc ** 2) * binwidth
        # cumulative thermal power up to each bin index
        thermal_cum_power_open = thermal_power_per_bin_open * np.arange(1, len(xf) + 1)
        thermal_cum_rms_open = np.sqrt(thermal_cum_power_open) * 1e6
        ax[2].semilogx(xf[start_idx:], thermal_cum_rms_open[start_idx:], 'k--', label="Thermal (50\u03A9) µV RMS")
        ax[2].grid(True, which="both")
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_ylabel("Integrated noise (µV RMS)")
        ax[2].legend(loc='lower right')

        plt.show()

# Modded for Noise Analysis by Igor-M
    def fft_traceN(self, data, window="None"):
        if window == "None":
            win = np.ones(data["N"])
            wenbw = 1.0
        elif window == "FlatTop":
            win = flattop(data["N"])
            wenbw = 3.77
        elif window == "Blackman":
            win = blackman(data["N"])
            wenbw = 1.73
        elif window == "Kaiser":
            win = kaiser(data["N"], beta=14)
            wenbw = 2.23
        elif window == "Hann":
            win = hann(data["N"])
            wenbw = 1.5
        else:
            print("Unknown window")
        # For tones mainly
        cpg = sum(win) / len(win)  # coherent power gain
        y_windowed=data["y"] * win / cpg
        # For general FFT
        ywf = rfft(y_windowed)
        xf = rfftfreq(data["N"], data["xincr"])
        return xf[1:], ywf[1:], wenbw
        
        
    def fft_trace(self, data, window="None"):
        # 1kHz scope calibration is 3Vpp => db20(2*3/pi)=5.62dBVa and 2.62dBVrms
        if  window == "None":
            win = np.ones(data["N"])
        elif window == "FlatTop":
            win = flattop(data["N"])
        elif window == "Blackman":
            win = blackman(data["N"])
        elif window == "Kaiser":
            win = kaiser(data["N"], beta=14)
        elif window == "Hann":
            win = hann(data["N"])
        else:
            print("Unknown window")
        # noinspection PyTypeChecker
        cpg = sum(win) / len(win)  # coherent power gain
        y_windowed=data["y"] * win / cpg
        ywf = rfft(y_windowed)
        xf = rfftfreq(data["N"], data["xincr"])
        return xf[1:], ywf[1:]        

    def save_data(self, ch, file=""):
        if self.demo and not self.demo == "save":
            with open(self.demo, "rb") as f:
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
        # Modded by Igor-M 08/2025            
        # Create timestamp string
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # truncate to milliseconds
        filename = f"CH{ch}_{timestamp}.csv"
        csv_data = np.transpose(np.stack((data["x"], data["y"])))
        np.savetxt(filename, csv_data, delimiter=',')
        self.warning["text"] = f"Saved to {filename}"


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
    parser.add_argument("-d", "--demo", help='"save" stores Bode data to <datetime>.bplt. To load specify <filename>.bplt')
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
    rm.demo = args.demo
    rm.scope_init(args.ip)
    ui = ScopeUI(rm)
    ui.demo = args.demo
    ui.loopgain = args.loopgain
    ui.mainloop()
    rm.close()
