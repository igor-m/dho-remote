#!/usr/bin/env python3

#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Modded by Igor-M aka iMo 2025/08/28 for Noise Measurements
#  v 1.4_01
#  more on eevblog:
#  https://www.eevblog.com/forum/metrology/diy-low-frenquency-noise-meter/msg6023949/#msg6023949
#  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import tkinter as tk
import tkinter.ttk as ttk
from scipy.signal.windows import flattop, blackman, hann, kaiser
from scipy.fft import rfft, rfftfreq
from functools import partial
import pickle
import datetime
import time

from scope.rigol import Scope, Siglent, ChannelNotEnabled

version = "v1.4"

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
        
        #NEW IMO
        
        # FFT averaging (N measurements)
        row = 7
        self.nshots = tk.StringVar()
        self.nshots.set("1")  # default = 1 measurement
        ttk.Label(master=vframe, text="FFT Aver. Noise (N)").grid(column=1, row=row, columnspan=2, sticky=tk.W)
        ttk.Entry(master=vframe, width=6, textvariable=self.nshots).grid(column=3, row=row, columnspan=2, sticky=tk.W)

        # LNA Gain
        row = 8
        self.lna_gain = tk.StringVar()
        self.lna_gain.set("1.0")  # default gain (dB)
        ttk.Label(master=vframe, text="LNA Gain (A)").grid(column=1, row=row, columnspan=2, sticky=tk.W)
        ttk.Entry(master=vframe, width=6, textvariable=self.lna_gain).grid(column=3, row=row, columnspan=2, sticky=tk.W)

        # END NEW IMO

        self.warning = ttk.Label(text=version)
        self.warning.grid(row=4, column=1, sticky=tk.EW)

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
        # === load or acquire data ===
        if self.demo and not self.demo == "save":
            with open(self.demo, "rb") as f:
                self.data1, self.data2 = pickle.load(f)
                data = self.data2[0] if ch == 2 else self.data1[0]
        else:
            # NEW IMO
            Nfft = min(max(int(self.nshots.get()), 1), 1000)  # clamp 1–1000
            spectra = []
            data = None
            for i in range(Nfft):
                self.instr.write(":SINGle")
                print(f"N.{i+1} {Nfft} FFT from CH{ch} ..")
                time.sleep(1)
                # wait until acquisition complete
                timeout = time.time() + 2000.0  # debug only: timeout per shot
                while True:
                    #time.sleep(1)
                    status = self.instr.query(":TRIGger:STATus?").strip().upper()
                    # DHO can answer RUN, AUTO, WAIT, TD, STOP
                    if status in ("STOP"):
                        print(f"N.{i+1} {Nfft} FFT DONE from CH{ch} !")
                        break  # acquisition done
                    if time.time() > timeout:
                        print("Timeout waiting for single shot trigger, proceeding anyway.")
                        break
                #time.sleep(2)
                
                try:
                    data = self.get_data(ch)
                except (ValueError, ChannelNotEnabled):
                    self.warning["text"] = f"No data from CH{ch}"
                    print(f"No data retrieved from CH{ch}")
                    return 1
                
                gain = float(self.lna_gain.get())   # linear gain (not dB!)
                data["y"] = data["y"] / gain
                xf, ywf_single, enbw = self.fft_trace(data, window=self.window_var.get())
                spectra.append(np.abs(ywf_single) ** 2) 
                
            # average power spectrum
            mean_power = np.mean(spectra, axis=0)
            # convert back to magnitude (Vrms/bin)
            ywf = np.sqrt(mean_power)
            # END NEW IMO

        print(f"Samples read: {Nfft} * {data['N']}")
        print(f'Sample rate={data["sr"] / 1e6:g} Ms')
        print(f'FFT max freq={data["sr"] / 2e6:g} MHz')

        # === RBW / ENBW / noise BW ===
        binwidth = data["sr"] / data["mdepth"]     # Hz per FFT bin
        # Determine ENBW depending on window
        win_name = str(self.window_var.get()).lower()
        # enbw = 1.5 if "hann" in win_name or "hanning" in win_name else 1.0
        noiseBW_db = 10.0 * np.log10(binwidth * enbw)   # dB (10*log10 of the equivalent noise bandwidth)
        print(f"FFT min freq={binwidth:g} Hz")
        print(f"FFT bin noise BW={noiseBW_db:.2f} dB (includes ENBW={enbw})")

        print_noise = True

        # === helper: weighted moving average of power (normalized) ===
        def weighted_moving_avg_power(x, w):
            if w <= 1:
                return x
            w = int(w)
            win = np.blackman(w)
            s = np.sum(win)
            if s == 0:
                return x
            win = win / s
            return np.convolve(x, win, mode='same')

        # === FFT trace & per-bin RMS voltage ===
        #xf, ywf = self.fft_trace(data, window=self.window_var.get())  # xf in Hz, ywf complex
        #Vrms = 2.0 / data["N"] * np.abs(ywf) / np.sqrt(2)              # V RMS per bin (raw bin RMS)
        Vrms =  2.0 / data["N"] * (ywf)  / np.sqrt(2.0)              # V RMS per bin (raw bin RMS)

        # smoothing window size (used only for smoothing the noise floor estimate)
        # avg_size = min(500, max(1, int(data["N"] / 10)))

        # local averaged power then sqrt -> smoothed Vrms (same length as Vrms)
        # Vpower = Vrms ** 2
        # Vpower_avg = weighted_moving_avg_power(Vpower, avg_size * 2)
        Vavg = Vrms # np.sqrt(Vpower_avg)   # smoothed Vrms (V)

        # === Noise Spectral Density (V/√Hz) and conversions ===
        # NSD = Vavg / sqrt(binwidth * ENBW)
        NSD_V_per_rtHz = Vavg / np.sqrt(binwidth * enbw)
        NSD_uV_per_rtHz = NSD_V_per_rtHz * 1e6
        # guard for log
        tiny = 1e-30
        NSD_dB_uV = 20.0 * np.log10(np.maximum(NSD_uV_per_rtHz, tiny))

        # === Integrated RMS noise (from low freq up) ===
        # power per bin = (NSD^2) * binwidth  -> sum -> sqrt -> Vrms
        noise_power_per_bin = (NSD_V_per_rtHz ** 2) * binwidth
        cumulative_power = np.cumsum(noise_power_per_bin)   # V^2
        cumulative_rms = np.sqrt(cumulative_power)          # V
        cumulative_rms_uV = cumulative_rms * 1e6            # µV RMS

        # === thermal noise reference lines (50Ω at 25°C) ===
        k = 1.380649e-23
        T = 298.15
        R = 50.0
        thermal_v_oc = np.sqrt(4 * k * T * R)   # open-circuit V/√Hz (~0.9 nV/√Hz)
        thermal_v_matched = thermal_v_oc / 2.0  # if source+load 50Ω matched (measured across load)
        thermal_uV_oc = thermal_v_oc * 1e6
        thermal_uV_matched = thermal_v_matched * 1e6
        ref_line_open = np.full_like(xf, thermal_uV_oc)
        ref_line_matched = np.full_like(xf, thermal_uV_matched)
        ref_line_open_db = 20.0 * np.log10(np.maximum(ref_line_open, tiny))
        ref_line_matched_db = 20.0 * np.log10(np.maximum(ref_line_matched, tiny))

        # === choose start index for plotting: skip DC (index 0) but keep very low freqs ===
        start_idx = 1 if len(xf) > 1 else 0

        # === Plotting ===
        fig, ax = plt.subplots(3, 1, figsize=(8, 11), layout="constrained")

        # NSD in µV/√Hz (log-log axis)
        if print_noise:
            ax[0].loglog(xf[start_idx:], NSD_uV_per_rtHz[start_idx:], label="Noise density [µV/√Hz]")
            ax[0].plot(xf, ref_line_open, 'k--', label=f"Thermal (open) ≈ {thermal_uV_oc*1e3:.3f} nV/√Hz")
            ax[0].plot(xf, ref_line_matched, 'r:', label=f"Thermal (matched) ≈ {thermal_uV_matched*1e3:.3f} nV/√Hz")
            ax[0].legend(loc='lower left')
        else:
            ax[0].plot(xf, ref_line_open, 'k--', label="Thermal reference")
            ax[0].legend(loc='lower left')
        ax[0].grid(True, which="both")
        ax[0].set_xlabel("Frequency (Hz)")
        ax[0].set_ylabel("µV/√Hz")

        # NSD in dBµV/√Hz (semilog x)
        if print_noise:
            ax[1].semilogx(xf[start_idx:], NSD_dB_uV[start_idx:], label="Noise density [dBµV/√Hz]")
            ax[1].semilogx(xf, ref_line_open_db, 'k--', label="Thermal (open) dBµV/√Hz")
            ax[1].semilogx(xf, ref_line_matched_db, 'r:', label="Thermal (matched) dBµV/√Hz")
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
        ax[2].semilogx(xf[start_idx:], thermal_cum_rms_open[start_idx:], 'k--', label="Thermal integrated (open) µV RMS")
        ax[2].grid(True, which="both")
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_ylabel("Integrated noise (µV RMS)")
        ax[2].legend(loc='lower right')

        plt.show()



    def fft_trace(self, data, window="flat"):
        # 1kHz scope calibration is 3Vpp => db20(2*3/pi)=5.62dBVa and 2.62dBVrms
        if window == "flat" or window == "none":
            win = np.ones(data["N"])
            enbw = 1.0
        elif window == "flattop":
            win = flattop(data["N"])
            enbw = 3.77
        elif window == "blackman":
            win = blackman(data["N"])
            enbw = 1.73
        elif window == "kaiser":
            win = kaiser(data["N"], beta=14)
            enbw = 2.23
        elif window == "hann":
            win = hann(data["N"])
            enbw = 1.5
        else:
            print("Unknown window")
        # For tones only:
        cpg = sum(win) / len(win)  # coherent power gain
        y_windowed=data["y"] * win / cpg
        # For FFT
        ywf = rfft(y_windowed)
        xf = rfftfreq(data["N"], data["xincr"])
        return xf[1:], ywf[1:], enbw

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
