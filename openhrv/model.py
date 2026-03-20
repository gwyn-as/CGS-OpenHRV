from config import MEANHRV_BUFFER_SIZE, HRV_BUFFER_SIZE, IBI_BUFFER_SIZE, LFHF_BUFFER_SIZE
from PySide6.QtCore import QObject, Signal, Slot, Property
import numpy as np
from utils import find_indices_to_average
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy import signal

class Model(QObject):

    # Signal format is (name, value).
    ibis_buffer_update = Signal(tuple)    # tuple(string, np.ndarray)
    mean_hrv_update = Signal(tuple)    # tuple(string, np.ndarray)
    lfhf_update = Signal(tuple)
    addresses_update = Signal(tuple)    # tuple(string, list)
    pacer_disk_update = Signal(tuple)    # tuple(string, list)
    pacer_rate_update = Signal(tuple)    # tuple(string, float)
    baseline_lfhf_update = Signal(tuple)
    hrv_target_update = Signal(tuple)    # tuple(string, int)
    biofeedback_update = Signal(tuple)    # tuple(string, float)

    def __init__(self):
        super().__init__()

        self._ibis_buffer = np.full(IBI_BUFFER_SIZE, 1000, dtype=int)
        self._ibis_seconds = np.arange(-IBI_BUFFER_SIZE, 0, dtype=float)
        self._mean_hrv_buffer = np.full(MEANHRV_BUFFER_SIZE, -1, dtype=int)
        self._mean_hrv_seconds = np.arange(-MEANHRV_BUFFER_SIZE, 0, dtype=float)
        self._lfhf_buffer = np.full(70, -1, dtype=int)
        self._lfhf_values_buffer = np.full(MEANHRV_BUFFER_SIZE, -1, dtype=float)
        self._lfhf_seconds = np.arange(-MEANHRV_BUFFER_SIZE, 0, dtype=float)
        self._hrv_buffer = np.full(HRV_BUFFER_SIZE, -1, dtype=int)
        self._current_ibi_phase = -1
        self._last_ibi_phase = -1
        self._last_ibi_extreme = 0
        self._sensors = []
        self._breathing_rate = 6.
        self._baseline_lfhf = 2.1
        self._hrv_mean_window = 15
        self._hrv_target = 30
        self._duration_current_phase = 0

    @Property(object)
    def ibis_buffer(self):
        return self._ibis_buffer

    @Slot(object)
    def set_ibis_buffer(self, value):
        self.ibis_seconds = value
        self._ibis_buffer = np.roll(self._ibis_buffer, -1)
        self._ibis_buffer[-1] = self.validate_ibi(value)
        self.ibis_buffer_update.emit(("InterBeatInterval", self.ibis_buffer))
        self.compute_local_hrv()
        self.compute_lfhf()
        
    def validate_ibi(self, value):
        """ Replace IBIs corresponding to instantaneous heart rate of more than
        220, or less than 30 beats per minute with local median."""
        if value < 273 or value > 2000:
            print(f"Correcting invalid IBI: {value}")
            return np.median(self._ibis_buffer[-11:])
        return value

    def compute_local_hrv(self):
        self._duration_current_phase += self._ibis_buffer[-1]
        current_ibi_phase = np.sign(self._ibis_buffer[-1] - self._ibis_buffer[-2])    # 1: IBI rises, -1: IBI falls, 0: IBI constant
        if current_ibi_phase == 0:
            return
        if current_ibi_phase == self._last_ibi_phase:
            return

        current_ibi_extreme = self._ibis_buffer[-2]
        local_hrv = abs(self._last_ibi_extreme - current_ibi_extreme)
        self.hrv_buffer = local_hrv

        seconds_current_phase = np.floor(self._duration_current_phase / 1000)
        self.mean_hrv_seconds = seconds_current_phase
        self._duration_current_phase = 0

        self._last_ibi_extreme = current_ibi_extreme
        self._last_ibi_phase = current_ibi_phase

    def compute_lfhf(self):
        # change to lfhf seconds
        self._duration_current_phase += self._ibis_buffer[-1]
        #current_ibi_phase = np.sign(self._ibis_buffer[-1] - self._ibis_buffer[-2])    # 1: IBI rises, -1: IBI falls, 0: IBI constant
        seconds_current_phase = np.floor(self._duration_current_phase / 1000)
        #if current_ibi_phase == 0:
        #    return
        #if current_ibi_phase == self._last_ibi_phase:
        #    return
        #seconds_current_phase = np.floor(self._duration_current_phase / 1000)
        self.lfhf_seconds = seconds_current_phase
        #self._duration_current_phase = 0
        
        #self.lfhf_seconds = seconds_current_phase
        #self._last_ibi_phase = current_ibi_phase
        
        self._lfhf_buffer = np.roll(self._lfhf_buffer, -1)
        self._lfhf_buffer[-1] = self._ibis_buffer[-1]
        # print(seconds_current_phase+', '+self._ibis_buffer[-1])
        # moved below:  
        #self.lfhf_values_buffer = self._lfhf_buffer[-1]

        
        # build a new array with the last window size seconds of the buffer
        if self._lfhf_buffer[1] != -1: # if the array is full of IBIs
            window_index = -1
            subarray = self._lfhf_buffer[window_index:-1]
            while np.sum(subarray) < 30000:
                window_index = window_index -1
                subarray = self._lfhf_buffer[window_index:-1]
            # while loop will overshoot to a total just over 30s: drop the oldest IBI
            subarray = self._lfhf_buffer[window_index+1:-1]
            # print(window_index, np.sum(subarray))
        else:
            neg1count = 0
            for x in self._lfhf_buffer:
                if x == -1:
                    neg1count += 1
            print("Buffering IBIs: "+str(70 - neg1count)+"/70", end='\r')
            #self.lfhf_label.setText(str(70 - neg1count)+"/70")
            return
        
        x = np.cumsum(subarray) / 1000.0
       #original: f = interp1d(x, subarray, kind='cubic')
        f = interp1d(x, subarray, kind='cubic', bounds_error=None, fill_value="extrapolate")
        # print("array f: " + str(f))
        fs = 16.0 # original was 4.0 but didn't give enough points for Welch's
        steps = 1/fs
        #        xx=np.arange(1, np.max(x), steps)
        xx=np.arange(1, np.max(x), steps)
        rr_interpolated = f(xx)
        # print('Interpolated: '+str(rr_interpolated))

        fxx, pxx = signal.welch(x=rr_interpolated, fs=fs)
        
        cond_vlf = (fxx >= 0) & (fxx < 0.04)
        cond_lf = (fxx >= 0.04) & (fxx < 0.15)
        cond_hf = (fxx >= 0.15) & (fxx < 0.4)
        
        vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
        lf = trapz(pxx[cond_lf], fxx[cond_lf])
        hf = trapz(pxx[cond_hf], fxx[cond_hf])
        
        print('VLF: '+str(vlf)+'LF: '+str(lf)+'HF: '+str(hf))
        print('LF/HF: '+str(lf/hf))
        
        self._lfhf_values_buffer = np.roll(self._lfhf_values_buffer, -1)
        self._lfhf_values_buffer[-1] = lf/hf
        # moved from above, this makes the reported value the actual lf/hf
        self.lfhf_values_buffer = lf/hf
        #print('added lfhf: '+str(lf/hf))
        

    def compute_biofeedback(self, x):
        """Hill equation.

        Biofeedback target (value of x at which half of the maximum reward is
        obtained) is equivalent to K parameter.

        Parameters
        ----------
        x : float
            Input value.

        Returns
        -------
        y : float
            Biofeedback value in the range [0, 1].

        References
        ----------
        [1] https://www.physiologyweb.com/calculators/hill_equation_interactive_graph.html
        [2] https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)
        """
        K = self.hrv_target * .5    # divide by half to make K the value at which maximum reward is obtained
        Vmax = 1    # Upper limit of y values
        n = 3    # Hill coefficient, determines steepness of curve
        y = Vmax * x**n / (K**n + x**n)

        self.biofeedback_update.emit(("Biofeedback", y))

    @property
    def hrv_buffer(self):
        return self._hrv_buffer

    @hrv_buffer.setter
    def hrv_buffer(self, value):
        if self._hrv_buffer[0] != -1:    # wait until buffer is full
            threshold = np.amax(self._hrv_buffer) * 4
            if value > threshold:    # correct hrv values that considerably exceed threshold
                print(f"Correcting outlier HRV {value} to {threshold}")
                value = threshold
        self._hrv_buffer = np.roll(self._hrv_buffer, -1)
        self._hrv_buffer[-1] = value
        average_idcs = find_indices_to_average(self._ibis_seconds[-HRV_BUFFER_SIZE:],
                                               self._hrv_mean_window)
        self.mean_hrv_buffer = self._hrv_buffer[average_idcs].mean()

    @property
    def mean_hrv_buffer(self):
        return self._mean_hrv_buffer

    @property
    def lfhf_values_buffer(self):
        return self._lfhf_values_buffer

    @mean_hrv_buffer.setter
    def mean_hrv_buffer(self, value):
        self.compute_biofeedback(value)
        self._mean_hrv_buffer = np.roll(self._mean_hrv_buffer, -1)
        self._mean_hrv_buffer[-1] = value
        self.mean_hrv_update.emit(("MeanHrv", self._mean_hrv_buffer))

    @lfhf_values_buffer.setter
    def lfhf_values_buffer(self, value):
        self._lfhf_values_buffer = np.roll(self._lfhf_values_buffer, -1)
        self._lfhf_values_buffer[-1] = value
        self.lfhf_update.emit(("LFHF", self._lfhf_values_buffer))
        # should go in view self.view.lfhf_label.setText("30 Second LF/HF: ", value)
        
        #print('Emitted LFHF Signal')

    @property
    def ibis_seconds(self):
        return self._ibis_seconds

    @ibis_seconds.setter
    def ibis_seconds(self, value):
        self._ibis_seconds = self._ibis_seconds - value / 1000
        self._ibis_seconds = np.roll(self._ibis_seconds, -1)
        self._ibis_seconds[-1] = -value / 1000

    @property
    def mean_hrv_seconds(self):
        return self._mean_hrv_seconds

    @mean_hrv_seconds.setter
    def mean_hrv_seconds(self, value):
        self._mean_hrv_seconds = self._mean_hrv_seconds - value
        self._mean_hrv_seconds = np.roll(self._mean_hrv_seconds, -1)
        self._mean_hrv_seconds[-1] = -value

    @Property(float)
    def breathing_rate(self):
        return self._breathing_rate

    @Property(float)
    def baseline_lfhf(self):
        return self._baseline_lfhf
    
    @Slot(float)
    def set_breathing_rate(self, value):
        self._breathing_rate = (value + 8) / 2    # force values into [4, 7], step .5
        self.pacer_rate_update.emit(("PacerRate", self._breathing_rate))

    @Slot(float)
    def set_baseline_lfhf(self, value):
        self._baseline_lfhf = value / 10
        self.baseline_lfhf_update.emit(("BaselineLFHF", self._baseline_lfhf))
        # print("baseline lfhf set to: ",value)
        

    @Property(int)
    def hrv_target(self):
        return self._hrv_target

    @Slot(int)
    def set_hrv_target(self, value):
        self._hrv_target = value
        self.hrv_target_update.emit(("HrvTarget", value))

    @property
    def pacer_coordinates(self):
        return self._pacer_coordinates

    @pacer_coordinates.setter
    def pacer_coordinates(self, value):
        self._pacer_coordinates = value
        self.pacer_disk_update.emit(("PacerCoordinates", value))

    @property
    def current_ibi_phase(self):
        return self._current_ibi_phase

    @current_ibi_phase.setter
    def current_ibi_phase(self, value):
        self._current_ibi_phase = value

    @property
    def last_ibi_phase(self):
        return self._last_ibi_phase

    @last_ibi_phase.setter
    def last_ibi_phase(self, value):
        self._last_ibi_phase = value

    @property
    def last_ibi_extreme(self):
        return self._last_ibi_extreme

    @last_ibi_extreme.setter
    def last_ibi_extreme(self, value):
        self._last_ibi_extreme = value

    @Property(object)
    def sensors(self):
        return self._sensors

    @Slot(object)
    def set_sensors(self, sensors):
        self._sensors = sensors
        self.addresses_update.emit(("Sensors",
                                    [f"{s.name()}, {s.address().toString()}"
                                     for s in sensors]))
