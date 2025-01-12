import pyvisa
import numpy as np

class ChannelNotEnabled(Exception):
    pass

class scope(pyvisa.ResourceManager):
    def __init__(self):
        super().__init__()

    def scopeInit(self, ipaddress):
        if self.debug:
            self.instr = False
            self.get_data = False
            return
        try:
            self.instr = self.open_resource(f"TCPIP::{ipaddress}::INSTR")
        except:
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
        # if not len(data["y"]):
        #     print("Not data returned by scope")
        #     return
        if min(data["y"]) == 0:
            print(f"CH{ch} data clips on minimum!")
        if max(data["y"]) == 2**16-1:
            print(f"CH{ch} data clips on maximum!")
        data["y"] = (data["y"] - data["yref"] - data["yorg"]) * data["yincr"]
        data["N"] = len(data["y"])
        data["x"] = np.linspace(0, data["N"] * data["xincr"], data["N"])
        data["x"] = data["x"] + data["xorg"]
        return data

class siglent(scope):
    def __init__(self):
        print("Siglent scope init")
