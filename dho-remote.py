import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
from scipy.signal.windows import flattop, blackman
from scipy.fft import fft, fftfreq
from functools import partial

from scope.rigol import scope, siglent, ChannelNotEnabled


class ScopeUI(tk.Tk):
    # data1: np.array
    # data2: np.array

    def __init__(self, rm):
        super().__init__()
        # self.geometry("240x200")
        self.instr = rm.instr
        self.get_data = rm.get_data

        self.title("DHO-remote")
        self.resizable(False, False)

        self.create_buttons()
        # Figure is created before the mainloop is started to keep the menu at the top.
        #  Not sure why this is needed, but now I can change the axis interactively
        # fig=plt.figure()
        # plt.close(fig)
        data1=[]
        data2=[]

    def create_buttons(self):
        # Vertical control colors
        self.vfg = {}
        self.vfg[1] = "#fcfa23"  # yellow
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
            fft_button = tk.Button(master=vframe, text="Save", command=partial(self.save_data, ch))
            fft_button.grid(column=ch, row=row)
            fft_button["width"] = vup[ch]["width"]

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
        ttk.Button(master=bodeframe, text="Plot", command=self.bodeplot).grid(row=0, column=0, columnspan=2)
        self.bodefmin = tk.StringVar()
        self.bodefmin.set("1")
        self.bodefmax = tk.StringVar()
        self.bodefmax.set("1e8")
        ttk.Label(master=bodeframe, text="Frequency range").grid(row=1, column=0, columnspan=2)
        ttk.Entry(master=bodeframe, width=5, textvariable=self.bodefmin).grid(row=2, column=0)
        ttk.Entry(master=bodeframe, width=5, textvariable=self.bodefmax).grid(row=2, column=1)

        ttk.Button(master=bodeframe, text="Capture low", command=partial(self.bodeplot, capture_a=True)).grid(row=4,
                                                                                                              column=0,
                                                                                                              columnspan=1)
        ttk.Button(master=bodeframe, text="Capture high", command=partial(self.bodeplot, capture_b=True)).grid(row=4,
                                                                                                               column=1,
                                                                                                               columnspan=1)

        vframe.grid(row=0, column=0, rowspan=3)
        hframe.grid(row=0, column=1, rowspan=1, sticky=tk.N)
        trigframe.grid(row=2, column=1)
        bodeframe.grid(row=0, column=2)
        self.warning = ttk.Label(text="OK")
        self.warning.grid(row=2, column=2)

    # def ReturnPressed(self, ch, event):
    #     text = self.vlabel_val[ch]
    #     print(text.get("1.0", tk.END))

    def bodeplot(self, capture_a=False, capture_b=False):
        instr = self.instr
        instr.write(":STOP")
        if capture_a:
            self.data1.append(self.get_data(1))
            self.data2.append(self.get_data(2))
            instr.write(":RUN")
            return
        data1 = self.get_data(1)
        data2 = self.get_data(2)
        instr.write(":RUN")

        fig, ax = plt.subplots(3, layout="constrained", figsize=(6, 10))
        ax[0].plot(data1["x"], data1["y"],
                   data2["x"], data2["y"], "-")
        ax[0].legend([f"CH1", f"CH2"], loc='lower left')
        ax[0].grid(True)

        xf1, ywf1 = self.fft_trace(data1)
        xf2, ywf2 = self.fft_trace(data2)

        Vrms1 = 2.0 / data1["N"] * abs(ywf1[:data1["N"] // 2]) / np.sqrt(2)
        Vrms2 = 2.0 / data2["N"] * abs(ywf2[:data2["N"] // 2]) / np.sqrt(2)
        Phase1 = np.angle(ywf1[:data1["N"] // 2], deg=True)
        Phase2 = np.angle(ywf2[:data2["N"] // 2], deg=True)
        PhaseDiff = Phase2 - Phase1
        if capture_b:
            # merge both runs
            xf1_pre, ywf1_pre = self.fft_trace(self.data1)
            xf2_pre, ywf2_pre = self.fft_trace(self.data2)
            Phase1_pre = np.angle(ywf1_pre[:self.data1["N"] // 2], deg=True)
            Phase2_pre = np.angle(ywf2_pre[:self.data2["N"] // 2], deg=True)
            PhaseDiff_pre = Phase2_pre - Phase1_pre
            Vrms1_pre = 2.0 / data1["N"] * abs(ywf1_pre[:data1["N"] // 2]) / np.sqrt(2)
            Vrms2_pre = 2.0 / data2["N"] * abs(ywf2_pre[:data2["N"] // 2]) / np.sqrt(2)
            # select the largest of the two
            sel_pre = Vrms1_pre > Vrms1
            Vrms1 = self.select_array(Vrms1, Vrms1_pre, sel_pre)
            Vrms2 = self.select_array(Vrms2, Vrms2_pre, sel_pre)
            PhaseDiff = self.select_array(PhaseDiff, PhaseDiff_pre, sel_pre)

        ax[1].semilogx(xf1, 20 * np.log10(Vrms1),
                       xf2, 20 * np.log10(Vrms2))
        #Only plot those points that have a reasonable amplitude
        xbode = Vrms1 > (max(Vrms1[2:]) / 1000)
        # Skip DC
        xbode[-2:] = False
        # Use range input from user
        xbode[xf1 < float(self.bodefmin.get())] = False
        xbode[xf1 > float(self.bodefmax.get())] = False

        bodedb = 20 * np.log10(Vrms2[xbode] / Vrms1[xbode])
        ax[2].semilogx(xf1[xbode], bodedb)
        ax2b = ax[2].twinx()
        bodephase = np.unwrap(PhaseDiff[xbode], period=360)
        # if (max(bodephase)-min(bodephase))>180:
        #     ax2b.set_ylim([-100, 0])
        ax[2].set_ylim([min(bodedb), max(bodedb)])
        ax2b.semilogx(xf1[xbode], bodephase, color='orange')
        ax2b.set_ylim([min(bodephase), max(bodephase)])
        ax[2].grid(True)
        from matplotlib.ticker import MultipleLocator
        if max(bodephase) - min(bodephase) < 1000:
            ax2b.yaxis.set_major_locator(MultipleLocator(45))

        # labels
        ax[0].set_xlabel("Time (s)")
        ax[1].set_xlabel("Freq (Hz)")
        ax[2].set_xlabel("Freq (Hz)")
        ax[0].set_ylabel("V")
        ax[1].set_ylabel("dBV")
        ax[2].set_ylabel('Amplitude (dB)', color='blue')
        ax2b.set_ylabel('Phase (Deg)', color='orange')

        ax[2].set_title("Bode plot")

        ax[1].grid(True)
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
        match scale:
            case _ if scale < 1e-6:
                scale = int(scale * 1e9)
                scale_unit = "ns"
            case _ if scale < 1e-3:
                scale = int(scale * 1e6)
                scale_unit = "us"
            case _ if scale < 1:
                scale = int(scale * 1e3)
                scale_unit = "ms"
            case _:
                scale_unit = "s"
        # text.delete("1.0", tk.END)
        # text.insert("1.0", f"{scale}{scale_unit}")
        text["text"] = f"{scale}{scale_unit}"

    def plot_fft(self, ch):
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
        # data["N"] = data["n"]
        self.instr.write(":RUN")
        print(f"Sample rate={data["sr"] / 1e6:g} Ms")
        print(f"FFT max freq={data["sr"] / 2e6:g}MHz")
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

        xf, ywf = self.fft_trace(data)
        # ax[0].plot(data["x"],data["y"],label="CH1")
        Vrms = 2.0 / data["N"] * abs(ywf[:data["N"] // 2]) / np.sqrt(2)
        avg_size = min(500, int(data["N"] / 10))
        print(f"Vrms={np.sqrt(sum(data["y"] ** 2) / data["N"])}")
        Vavg = moving_average(Vrms ** 2, avg_size * 2)
        Vavg = (Vavg / avg_size / 2) ** 0.5

        fig, ax = plt.subplots(3, layout="constrained", figsize=(6, 10))
        if print_noise:
            ax[0].semilogy(xf, Vrms,
                           xf[avg_size:], (Vavg[avg_size:] / 10 ** (noiseBW / 20)), "-")
        else:
            ax[0].semilogy(xf, Vrms, "-")
        ax[0].legend([f"CH{ch} RBW={binwidth:g}Hz", f"Noise V/sqrt(Hz)"], loc='lower left')
        ax[0].grid(True)
        ax[0].set_xlabel("Freq F(Hz)")
        ax[0].set_ylabel("V")

        if print_noise:
            ax[1].semilogx(xf, 20 * np.log10(Vrms),
                           xf[avg_size:], 20 * np.log10(Vavg[avg_size:]) - noiseBW)
        else:
            ax[1].semilogx(xf, 20 * np.log10(Vrms))
        ax[1].grid(True)
        ax[1].set_xlabel("Freq (Hz)")
        ax[1].set_ylabel("dBV")
        ax[1].legend([f"CH{ch} RBW={binwidth:g}Hz", f"Noise V/sqrt(Hz)"], loc='lower left')

        ax[2].plot(data["x"], data["y"], "-")
        ax[2].grid(True)
        ax[2].set_xlabel("time (s)")
        ax[2].set_ylabel("V")
        # plt.ion()
        # plt.pause(.001)
        plt.show()

    def fft_trace(self, data):
        # 1kHz scope calibration is 3Vpp => db20(2*3/pi)=5.62dBVa and 2.62dBVrms
        ftw = flattop(data["N"])
        bmw = blackman(data["N"])
        bwall = np.ones(data["N"])
        window = bmw
        # noinspection PyTypeChecker
        cpg = sum(window) / len(window)  # coherent power gain
        ywf = fft(data["y"] * window / cpg)
        # ywf=np.ones(n)
        xf = fftfreq(data["N"], data["xincr"])[:data["N"] // 2]
        return xf, ywf

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

    parser = argparse.ArgumentParser()
    parser.add_argument("ip_address", help="IP or hostname of oscilloscope")
    parser.add_argument("-m", "--scope_brand", default="rigol", help="scope brand")
    args = parser.parse_args()
    match args.scope_brand:
        case "rigol":
            rm = scope()
        case "siglent":
            rm = siglent()
        case _:
            print("Unkown model")
            quit(1)
    rm.scopeInit(args.ip_address)
    ui = ScopeUI(rm)
    ui.mainloop()
