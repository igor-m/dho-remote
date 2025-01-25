import time

import pyvisa
import numpy as np

class ChannelNotEnabled(Exception):
    pass

class Scope(pyvisa.ResourceManager):
    def __init__(self):
        super().__init__()

    def scope_init(self, ipaddress):
        if self.debug == 2:
            self.instr = False
            self.get_data = False
            return
        try:
            self.instr = self.open_resource(f"TCPIP::{ipaddress}::INSTR")
        except pyvisa.errors.VisaIOError:
            print("Could not connect to scope")
            quit(1)
        self.instr.chunk_size = 1000000
        self.instr.read_termination = '\n'
        self.instr.write_termination = '\n'
        try:
            print(self.instr.query('*IDN?'))
        except pyvisa.errors.VisaIOError:
            print("Scope does not respond on IDN query. Please reboot scope")
            # self.instr.query("*WAI")
            quit(1)
    def scope_close(self):
        self.close()

    def get_data(self,ch):
        data = {}
        instr=self.instr
        if float(instr.query(f":CHAN{ch}:DISPLAY?")):
            instr.write(f":WAV:SOURCE CHAN{ch}")
        else:
            raise ChannelNotEnabled(f"Channel {ch} is not enabled")

        instr.write(":WAV:MODE RAW")
        instr.write(":WAV:FORM WORD")
        data["yincr"] = float(instr.query(":WAVeform:YINCrement?"))
        data["yorg"] = float(instr.query(":WAVeform:YORIGIN?"))
        data["yref"] = float(instr.query(":WAVeform:YREFERENCE?"))
        data["xincr"] = float(instr.query(":WAVeform:XINCrement?"))
        data["xorg"] = float(instr.query(":WAVeform:XORIGIN?"))
        data["mdepth"] = float(instr.query(":ACQuire:MDEPth?"))
        data["sr"] = float(instr.query(":ACQuire:SRATe?"))
        data["bwl"] = instr.query(f":CHANnel{ch}:BWLimit?")


        data["y"] = instr.query_binary_values(":WAVeform:data?", datatype='H', is_big_endian=False, container=np.array)

        if min(data["y"]) == 0:
            print(f"WARNING:CH{ch} data clips on minimum!")
        if max(data["y"]) == 2**16-1:
            print(f"WARNING:CH{ch} data clips on maximum!")
        data["y"] = (data["y"] - data["yref"] - data["yorg"]) * data["yincr"]
        data["ymax"] = (2**16-1 - data["yref"] - data["yorg"]) * data["yincr"]
        data["ymin"] = (      0 - data["yref"] - data["yorg"]) * data["yincr"]
        data["N"] = len(data["y"])
        data["x"] = np.linspace(0, data["N"] * data["xincr"], data["N"])
        data["x"] = data["x"] + data["xorg"]
        return data

    def bode_setup(self):
        self.instr.write(":AUToset")
        self.instr.write(":ACQuire:TYPE AVER")
        self.instr.write(":ACQuire:AVERages 16")
        self.instr.write(":CHANnel1:BWLimit 20M")
        self.instr.write(":CHANnel2:BWLimit 20M")
        self.instr.write(":ACQuire:MDEPth 1M")
        # enabled=self.instr.query(":COUNter:ENABle?")
        self.instr.write(":COUNter:ENABle ON")
        self.instr.write(":COUNter:SOURce CHAN1")
        self.instr.write(":COUNter:MODe FREQ")

        # self.instr.write("*OPC")
        # ready=self.instr.query("*OPC?")
        # while  ready == "0" :
        #     print("waiting...")
        #     time.sleep(1)
        #     ready=self.instr.query("*OPC?")
        fch1=0
        time.sleep(2)
        tries=0
        while tries < 3 and (fch1 <1 or fch1 >1e8) :
            try:
                fch1=float(self.instr.query(":COUNter:CURRent?"))
            except:
                True
            tries=tries+1
            time.sleep(1)
        if (fch1 <1 or fch1 >1e8):
            print("No signal detected on CH1")
            return
        print(f"Detected a signal on CH1 with freq={fch1}")

        htime=20e-3
        self.instr.write(f":TIMebase:SCALe {htime}")
        sr = float(self.instr.query(":ACQuire:SRATe?"))
        mdepth = float(self.instr.query(":ACQuire:MDEPth?"))
        trace_time=mdepth/sr
        while trace_time > 1/fch1 and htime >= 1e-6:
            # print(f"Input period t={1/fch1}")
            # print(f"lowest t={total_time}")
            htime=float(self.instr.query(f":TIMebase:SCALe?"))
            htime=htime/2
            self.instr.write(f":TIMebase:SCALe {htime}")
            sr = float(self.instr.query(":ACQuire:SRATe?"))
            trace_time=mdepth/sr

class Siglent(Scope):
    def __init__(self):
        print("Siglent scope init")
