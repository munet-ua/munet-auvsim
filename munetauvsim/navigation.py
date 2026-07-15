"""
Navigation functions and sensor classes for AUV state estimation.

Implements the Navigation block of GNC design for determining vehicle position,
attitude, velocity, course, and distance traveled. Provides sensor abstractions,
coordinate transformations, state observers, and filtering algorithms.


Classes
-------
Sensor
    Abstract base class for sensor implementations.
OceanCurrentSensor
    Measures ocean current speed and direction.
OceanDepthSensor
    Measures ocean floor depth at vehicle position.

    
Functions
---------
**Coordinate Transformations:**

    - attitudeEuler(vehicle) : Integrate vehicle attitude using Euler angles.
    - Rzyx(phi, theta, psi) : Rotation matrix in SO(3) using zyx convention.
    - Tzyx(phi, theta) : Attitude transformation matrix using zyx convention.

**State Computation:**

    - statePT(vehicle, pt1, pt2) : Path-tangential angle and track errors.
    - stateSpeed(vehicle) : Vehicle speed magnitude.

**Observers and Filters:**

    - headingFilterLOS(vehicle, psi_ref) : LOS heading observer with yaw rate
      estimation.
    - depthFilter(vehicle, pt) : Exponential moving average depth filter.
    - maxDepthLimit(vehicle, z) : Enforce depth safety limits.

    
Notes
-----
- Navigation block inputs: Sensors, vehicle motion
- Navigation block outputs: State vectors to Guidance and Control blocks


References
----------
[1] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
Control. 2nd Edition, Wiley. https://www.fossen.biz/wiley

[2] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
https://github.com/cybergalactic/PythonVehicleSimulator

[3] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
https://github.com/cybergalactic/MSS
"""

from __future__ import annotations
from typing import Any, List, Tuple, TYPE_CHECKING
from numpy.typing import NDArray
from abc import ABC, abstractmethod
if (TYPE_CHECKING):
    from munetauvsim.vehicles import Vehicle
    from munetauvsim.environment import Ocean
import numpy as np
import math
from scipy.signal import hilbert, find_peaks
from munetauvsim import gnc
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]

# Globarl Variables
log = logger.addLog('nav')
signal = {}
"""Module-level registry of in-flight sonar transmissions, keyed by the
transmitting vehicle's id. Updated by SonarSensor.transmit and consumed by
Simulator._apply_channel_effects."""

###############################################################################

class Sensor(ABC):
    """
    Abstract base class for sensor implementations.
    
    Defines interface for sensor objects that collect data from the simulation
    environment. Subclasses must implement collectData() method.
    

    Notes
    -----
    - Sensors as objects supports ability to create a data log or to cache data
      on the sensor itself.
    - Consider writing an __init__ method for class-wide attributes, such as a
      'name' string that provides a default for the AUV.sensor dictionary to use
      as a reference.
    """

    @abstractmethod
    def collectData(self, **kwargs)->Any:
        """
        Collect sensor measurement data.
        
        Returns
        -------
        data : Any
            Sensor-specific measurement data.
        """

###############################################################################

class OceanCurrentSensor(Sensor):
    """
    Sensor for measuring ocean current speed and direction.
    
    Reads current data from Ocean object at specified simulation iteration.
    """

    def collectData(self, 
                    i:int=None,
                    ocean:Ocean=None, 
                    **kwargs)->List[float]:
        """
        Measure ocean current at simulation iteration i.
        

        Parameters
        ----------
        i : int
            Simulation iteration counter.
        ocean : Ocean
            Ocean object with current.speed and current.angle arrays.
        **kwargs
            Unused. Required for AUV sensor interface compatibility.
            

        Returns
        -------
        speed : float
            Current speed in m/s.
        direction : float
            Current direction in radians.
            

        Notes
        -----
        Returns [-1.0, -1.0] on error with log message.
        """
        
        if (ocean is None) or (i is None):
            log.error("%s requires 'ocean' and 'i' arguments.",
                      self.__class__.__name__)
            return [-1.0, -1.0]
        
        return [ocean.current.speed[i], ocean.current.angle[i]]

###############################################################################

class OceanDepthSensor(Sensor):
    """
    Sensor for measuring ocean floor depth at vehicle position.
    
    Queries Ocean.floor() method at vehicle's (x, y) coordinates.
    """

    def collectData(self, 
                    ocean:Ocean=None, 
                    eta:NPFltArr=None,
                    **kwargs)->float:
        """
        Measure ocean depth at vehicle position.
        

        Parameters
        ----------
        ocean : Ocean
            Ocean object with floor(x, y) method.
        eta : ndarray, shape (6,)
            Vehicle position/attitude [x, y, z, phi, theta, psi].
        **kwargs
            Unused. Required for AUV sensor interface compatibility.

               
        Returns
        -------
        depth : float
            Ocean floor depth in meters at position (eta[0], eta[1]).

             
        Notes
        -----
        Returns -1.0 on error with log message.
        Returns np.inf when ocean.floor is None (no bathymetry modeled),
        signalling an unbounded depth below the vehicle.
        """

        if (ocean is None) or (eta is None):
            log.error("%s requires 'ocean' and 'eta' arguments.",
                      self.__class__.__name__)
            return -1.0

        if (ocean.floor is None):
            return np.inf

        return ocean.floor(eta[0],eta[1])

###############################################################################

class PollutionSensor(Sensor):
    """
    Sensor for measuring pollutant concentration level at vehicle position.

    Queries Ocean.pollution.calculate_concentration() at the vehicle's (x, y)
    coordinates, returning a discrete concentration level used by the SUSD
    source-seeking guidance law. Requires a pollution field on the ocean (see
    Ocean(createPlume=True)) with num_levels configured.
    """

    def collectData(self,
                    ocean:Ocean=None,
                    eta:NPFltArr=None,
                    **kwargs)->float:
        """
        Measure pollutant concentration level at vehicle position.


        Parameters
        ----------
        ocean : Ocean
            Ocean object with a pollution model exposing
            calculate_concentration([x, y]).
        eta : ndarray, shape (6,)
            Vehicle position/attitude [x, y, z, phi, theta, psi].
        **kwargs
            Unused. Required for AUV sensor interface compatibility.


        Returns
        -------
        concentration : float
            Concentration level at position (eta[0], eta[1]).


        Notes
        -----
        Returns 0.0 when the ocean has no pollution model or on missing
        arguments (logged as an error).
        """

        if (ocean is None) or (eta is None):
            log.error("%s requires 'ocean' and 'eta' arguments.",
                      self.__class__.__name__)
            return 0.0

        if (getattr(ocean, 'pollution', None) is None):
            return 0.0

        return ocean.pollution.calculate_concentration(eta[:2])

###############################################################################

class SonarSensor(Sensor):
    """
    Acoustic sonar sensor for inter-vehicle ranging and data sharing.

    Simulates the physical acoustic link used by the SUSD source-seeking
    swarm: each vehicle periodically transmits an LFM chirp pulse train whose
    pulse gaps encode its measured pollutant concentration level. Receiving
    vehicles matched-filter the (attenuated, delayed, noisy) signal to decode:

    - Range, from the time of arrival of the first correlation peak. Clock
      drift between vehicles (vehicle.timedrift) injected by the channel
      produces realistic ranging errors.
    - Neighbor concentration, from the interval between the last two peaks.
    - Bearing (relative position direction), via beamforming on two
      orthogonal receiver arrays.

    Decoded values are written into ``rx_vehicle.neighbors[tx_vehicle.id]``
    for consumption by the SUSD guidance law (guidance.velSUSD).


    Notes
    -----
    Unlike environmental sensors, this sensor is not read through
    collectSensorData/readAllSensors; it is driven by the Simulator's
    TDMA-scheduled loop (Simulator._simulateSonarTDMA) through the
    transmit/receive pair. Its collectData method is a no-op returning None
    for Sensor interface compatibility.


    //GQ 11/2024 (ported to munetauvsim 07/2026)
    """

    def __init__(self):
        # Acoustic signal parameters
        self.sample_rate = 8e3          # sampling rate (Hz)
        self.f_carrier = 0.024e6        # carrier frequency (Hz)
        self.pulse_duration = 0.02      # pulse duration (s)
        self.c = 1500                   # speed of sound in water (m/s)
        self.tx_power = 1000            # transmit power (W)
        self.f0 = 6.75e4                # chirp start frequency (Hz)
        self.f1 = 7.25e4                # chirp end frequency (Hz)
        self.N = 10000                  # number of samples
        self.bw = 8e3                   # bandwidth (Hz)
        self.AOA_error = np.pi/48       # angle-of-arrival error bound (rad)
        self.TOA_error = 0.0            # time-of-arrival jitter bound (s)

        # Pre-calculate the matched filter reference chirp
        self.ref_chirp = self.lfm_baseband(self.pulse_duration, self.bw,
                                           self.sample_rate)
        self._matched_filter_kernel = np.flip(self.ref_chirp.conj())

    #--------------------------------------------------------------------------
    def collectData(self, **kwargs)->None:
        """No-op for Sensor interface compatibility. Returns None."""
        return None

    #--------------------------------------------------------------------------
    @staticmethod
    def lfm_baseband(T_lfm:float, bw:float, sampling_rate:float)->NPFltArr:
        """
        Generate a baseband Linear Frequency Modulated (LFM) chirp signal.


        Parameters
        ----------
        T_lfm : float
            LFM signal duration (s).
        bw : float
            Nominal passband signal bandwidth, fmax - fmin (Hz).
        sampling_rate : float
            Sampling rate of the LFM (Hz).


        Returns
        -------
        sigx : ndarray
            Complex LFM signal at the baseband.


        References
        ----------
        [1] A. Hein, Processing of SAR data: Fundamentals, Signal Processing,
        Interferometry, Springer, New York, 2004.
        """

        mu = bw / T_lfm                                 # frequency sweep rate
        ttl_samples = round(sampling_rate * T_lfm)
        t_lfm = ((np.arange(0, ttl_samples) - ttl_samples / 2) / sampling_rate)
        sigx = np.exp(1j * np.pi * mu * t_lfm**2)
        return sigx

    #--------------------------------------------------------------------------
    def matched_filter_complex_correlation(self,
                                           rx:NPFltArr,
                                           tx:NPFltArr,
                                           cir_length:int=None,
                                           toa_error:float=None,
                                           )->Tuple[NPFltArr,NPFltArr]:
        """
        Perform matched filtering via complex correlation.

        Correlates the received signal with the time-reversed conjugate of
        the transmitted signal to maximize SNR and detect signal arrival.


        Parameters
        ----------
        rx : ndarray
            Received signal.
        tx : ndarray
            Transmitted (reference) signal.
        cir_length : int, optional
            Desired length of the channel impulse response output.
        toa_error : float, optional
            Time-of-arrival error bound in seconds. Each call samples a
            random value uniformly from [-toa_error, toa_error]. Defaults to
            self.TOA_error.


        Returns
        -------
        mf_result : ndarray
            Matched filter output (complex correlation result).
        cir_cmplx : ndarray
            Complex envelope (Hilbert magnitude for real signals).
        """

        rx = np.asarray(rx).ravel()
        tx = np.asarray(tx).ravel()
        iscomplex = np.iscomplexobj(rx) or np.iscomplexobj(tx)

        # Matched filter: normalized time-reversed conjugate of tx
        matched_filter = np.conj(tx[::-1])
        matched_filter = matched_filter / np.sqrt(
            np.sum(np.abs(matched_filter) ** 2))

        # Matched filtering via convolution; drop initial transient samples
        mf_result = np.convolve(matched_filter, rx, mode='full')
        start_idx = len(matched_filter) - 1
        mf_result = mf_result[start_idx:]

        # Inject TOA error sampled uniformly from [-toa_error, toa_error]
        if (toa_error is None):
            toa_error = self.TOA_error
        toa_error = abs(float(toa_error))
        if (toa_error > 0):
            toa_error_sample = np.random.uniform(-toa_error, toa_error)
            signed_shift = int(round(toa_error_sample * self.sample_rate))
            if (signed_shift != 0):
                shifted = np.zeros_like(mf_result)
                if (signed_shift > 0):
                    shifted[signed_shift:] = mf_result[:-signed_shift]
                else:
                    n = abs(signed_shift)
                    shifted[:-n] = mf_result[n:]
                mf_result = shifted

        if (cir_length is not None):
            mf_result = mf_result[:cir_length]

        if (iscomplex):
            cir_cmplx = mf_result
        else:
            cir_cmplx = np.abs(hilbert(mf_result))

        return mf_result, cir_cmplx

    #--------------------------------------------------------------------------
    def transmit(self, vehicle:Vehicle)->dict:
        """
        Build the vehicle's sonar pulse train encoding its concentration.

        Creates a train of LFM chirp pulses. The gaps between the final
        pulses are stretched to (concentration + 3) ms, physically encoding
        the vehicle's measured concentration level in the signal timing so
        receivers can decode it after matched filtering.


        Parameters
        ----------
        vehicle : Vehicle
            The transmitting vehicle. Must carry a ``concentration``
            attribute (discrete level, see Pollution.num_levels).


        Returns
        -------
        dict
            ``{vehicle.id: {'signal': ndarray, 'sample_num': int}}`` suitable
            for merging into the module-level ``signal`` registry.
        """

        num_pulses = 3
        concentration = vehicle.concentration
        sample_rate = self.sample_rate
        num_samples = self.N
        pulse_duration = self.pulse_duration
        bandwidth = self.bw

        # Chirp signal scaled to 180 dB source level
        t = np.arange(0, pulse_duration, 1/sample_rate)
        chirp_signal = self.lfm_baseband(pulse_duration, bandwidth,
                                         sample_rate)
        signal_power = 10 ** (180 / 10)
        chirp_signal = chirp_signal * np.sqrt(
            signal_power / np.mean(np.abs(chirp_signal) ** 2))

        # Pulse gaps: last two intervals encode concentration as
        # (concentration + 3) ms instead of the default 40 ms
        default_gap_ms = 40
        pulse_gaps_ms = []
        for i in range(num_pulses - 1):
            if (i >= num_pulses - 2):
                pulse_gaps_ms.append(concentration + 3)
            else:
                pulse_gaps_ms.append(default_gap_ms)
        pulse_gaps_samples = [int(ms / 1000 * sample_rate)
                              for ms in pulse_gaps_ms]
        total_samples = len(t) * num_pulses + sum(pulse_gaps_samples)

        # Assemble the pulse train
        s_tx = np.zeros(total_samples, dtype=complex)
        current_idx = 0
        for i in range(num_pulses):
            s_tx[current_idx:current_idx + len(t)] = chirp_signal
            current_idx += len(t)
            if (i < num_pulses - 1):
                current_idx += pulse_gaps_samples[i]

        return {
            vehicle.id: {
                'signal': s_tx,
                'sample_num': num_samples,
            }
        }

    #--------------------------------------------------------------------------
    def receive(self,
                s_rx:NPFltArr,
                rx_vehicle:Vehicle,
                tx_vehicle:Vehicle,
                delay_samples:int,
                )->None:
        """
        Process a received sonar transmission and update neighbor data.

        Matched-filters the received signal to decode range (first peak time
        of arrival) and the transmitter's concentration level (interval
        between the last two peaks), then estimates the bearing to the
        transmitter via beamforming. Results are written into
        ``rx_vehicle.neighbors[tx_vehicle.id]``.


        Parameters
        ----------
        s_rx : ndarray
            Received signal (already delayed, attenuated, and noisy; see
            Simulator._apply_channel_effects).
        rx_vehicle : Vehicle
            Receiving vehicle whose neighbor table is updated.
        tx_vehicle : Vehicle
            Transmitting vehicle (used for ground-truth signal direction in
            the beamforming geometry).
        delay_samples : int
            Channel propagation delay in samples (informational).
        """

        sample_rate = self.sample_rate
        num_samples = self.N
        pulse_duration = self.pulse_duration

        signal_direction = tx_vehicle.eta[0:2] - rx_vehicle.eta[0:2]

        # Matched filter via correlation against the reference chirp
        mf_result, cir_cmplx = self.matched_filter_complex_correlation(
            s_rx, self.ref_chirp, cir_length=num_samples)

        # Time axis for the correlation output (ms)
        t_mf_length = 16000
        lag_samples = np.arange(t_mf_length)
        t_mf = lag_samples / sample_rate * 1000

        # Peak detection on the correlation output
        peaks, properties = find_peaks(
            np.abs(mf_result),
            height=0.9 * np.max(np.abs(mf_result)),
            distance=100
        )

        peak_times = t_mf[peaks] if (len(peaks) > 0) else []
        first_peak_time = peak_times[0] if (len(peak_times) > 0) else None

        if (len(peak_times) >= 2):
            last_peak_interval = peak_times[-1] - peak_times[-2]
        else:
            # Not enough peaks decoded; skip this reception
            return None

        # Skip updates when this transmitter is not in the receiver's
        # neighbor map
        neighbor_data = rx_vehicle.neighbors.get(tx_vehicle.id)
        if (neighbor_data is None):
            return None

        # Decode range from time of arrival, concentration from the encoded
        # pulse gap (23 ms = 20 ms pulse + 3 ms base gap offset)
        neighbor_data['range'] = ((first_peak_time + pulse_duration * 1000)
                                  * self.c / 1000)
        neighbor_data['concentration'] = last_peak_interval - 23

        # Isolate the first pulse for beamforming
        signal_energy = np.abs(s_rx)**2
        threshold = 0.1 * np.max(signal_energy)
        pulse_start = np.where(signal_energy > threshold)[0][0]
        pulse_duration_samples = int(0.002 * self.sample_rate)
        signal_for_beamforming = s_rx[pulse_start:
                                      pulse_start + pulse_duration_samples]

        # Receiver array orientation: vehicle velocity direction, falling
        # back to the heading vector when nearly stationary
        array1_direction = np.asarray(rx_vehicle.velocity[0:2], dtype=float)
        if (np.linalg.norm(array1_direction) < 1e-6):
            psi = float(rx_vehicle.eta[5])
            array1_direction = np.array([np.cos(psi), np.sin(psi)])

        # Estimate relative position direction via beamforming
        neighbor_data['rel_pos'] = self.beam_forming(
            signal_for_beamforming,
            signal_direction,
            array1_direction
        )

    #--------------------------------------------------------------------------
    def receive_direct(self, rx_vehicle:Vehicle, tx_vehicle:Vehicle)->None:
        """
        Update neighbor data from ground truth, bypassing signal processing.

        Same result as transmit/receive through an ideal channel: exact
        range, exact relative direction, and the transmitter's current
        concentration are written into
        ``rx_vehicle.neighbors[tx_vehicle.id]``.


        Parameters
        ----------
        rx_vehicle : Vehicle
            Receiving vehicle whose neighbor data will be updated.
        tx_vehicle : Vehicle
            Transmitting vehicle whose info is written into the neighbor
            entry.
        """

        neighbor_data = rx_vehicle.neighbors.get(tx_vehicle.id)
        if (neighbor_data is None):
            return

        rel_vec = tx_vehicle.eta[0:2] - rx_vehicle.eta[0:2]
        dist = float(np.linalg.norm(rel_vec))

        neighbor_data['range'] = dist
        neighbor_data['concentration'] = float(
            getattr(tx_vehicle, 'concentration', 0.0))
        if (dist > 1e-9):
            neighbor_data['rel_pos'] = tuple(rel_vec / dist)

    #--------------------------------------------------------------------------
    def beam_forming(self,
                     delayed_signal:NPFltArr,
                     signal_direction:NPFltArr,
                     array1_direction:NPFltArr,
                     aoa_error:float=None,
                     )->NPFltArr:
        """
        Estimate the relative position direction of a transmitter.

        Simulates two orthogonal uniform linear receiver arrays and sweeps
        candidate steering angles to find the angle of arrival on each. The
        two estimates are combined into a 2-D direction vector, with uniform
        AOA error injected.


        Parameters
        ----------
        delayed_signal : ndarray
            Received signal segment used for the response power estimate.
        signal_direction : ndarray, shape (2,)
            Ground-truth direction from receiver to transmitter (sets the
            true arrival angles on the arrays).
        array1_direction : ndarray, shape (2,)
            Orientation of the first receiver array (typically the vehicle
            velocity direction). Must be nonzero.
        aoa_error : float, optional
            Angle-of-arrival error bound in radians. Each estimated angle
            samples a random value uniformly from [-aoa_error, aoa_error].
            Defaults to self.AOA_error.


        Returns
        -------
        ndarray, shape (2,)
            Estimated relative position direction vector (not normalized).
        """

        # Second array is orthogonal to the first
        rotation_90 = np.array([[0, 1], [-1, 0]])
        array2_direction = array1_direction @ rotation_90

        norm_signal_direction = np.linalg.norm(signal_direction)
        norm_array1_direction = np.linalg.norm(array1_direction)
        norm_array2_direction = np.linalg.norm(array2_direction)

        dot_sig_arr1 = np.dot(signal_direction, array1_direction)
        dot_sig_arr2 = np.dot(signal_direction, array2_direction)

        d = 0.5                     # half wavelength element spacing
        Nr1 = 6                     # array 1 element count
        Nr2 = 4                     # array 2 element count

        # True arrival angles on each array
        theta1 = np.arccos(dot_sig_arr1 /
                           (norm_signal_direction * norm_array1_direction))
        theta2 = np.arccos(dot_sig_arr2 /
                           (norm_signal_direction * norm_array2_direction))
        if (theta1 > np.pi / 2):
            theta1 -= np.pi
        if (theta2 > np.pi / 2):
            theta2 -= np.pi

        # True steering vectors for the incoming signal angle
        n1 = np.arange(Nr1, dtype=float)
        n2 = np.arange(Nr2, dtype=float)
        v1 = np.exp(-2j * np.pi * d * n1 * np.sin(theta1))
        v2 = np.exp(-2j * np.pi * d * n2 * np.sin(theta2))

        # Vectorized angle search:
        #   y(a) = signal * c(a), c(a) = v_true @ conj(w(a)) / Nr
        #   var(y(a)) = |c(a)|^2 * var(signal)
        thetas     = np.arange(-90, 90.1, 0.5)
        thetas_rad = np.radians(thetas)

        W_1 = np.exp(-2j * np.pi * d * n1[None, :]
                     * np.sin(thetas_rad[:, None]))
        W_2 = np.exp(-2j * np.pi * d * n2[None, :]
                     * np.sin(thetas_rad[:, None]))

        c1 = (W_1 @ v1.conj()) / Nr1
        c2 = (W_2 @ v2.conj()) / Nr2

        var_sig = float(np.var(delayed_signal))
        eps     = float(np.finfo(float).eps)

        responses_1 = 10 * np.log10(np.abs(c1) ** 2 * var_sig + eps)
        responses_2 = 10 * np.log10(np.abs(c2) ** 2 * var_sig + eps)
        responses_1 = responses_1 - np.max(responses_1)
        responses_2 = responses_2 - np.max(responses_2)

        # Angles of maximum response
        aoa_1 = thetas[np.argmax(responses_1)]
        if (aoa_1 < 0):
            aoa_1 += 180
        aoa_2 = thetas[np.argmax(responses_2)]
        if (aoa_2 < 0):
            aoa_2 += 180

        # Convert to radians and inject uniform AOA error
        if (aoa_error is None):
            aoa_error = self.AOA_error
        aoa_error = abs(float(aoa_error))
        aoa_1_rad = np.radians(aoa_1) + np.random.uniform(-aoa_error,
                                                          aoa_error)
        aoa_2_rad = np.radians(aoa_2) + np.random.uniform(-aoa_error,
                                                          aoa_error)

        # Candidate directions: array orientations rotated by +/- each AOA;
        # pick the rotation pair that agrees best between the two arrays
        c1r, s1r = np.cos(aoa_1_rad), np.sin(aoa_1_rad)
        c2r, s2r = np.cos(aoa_2_rad), np.sin(aoa_2_rad)

        R1 = np.array([[ c1r, -s1r], [ s1r, c1r]])
        R2 = np.array([[ c1r,  s1r], [-s1r, c1r]])
        R3 = np.array([[ c2r, -s2r], [ s2r, c2r]])
        R4 = np.array([[ c2r,  s2r], [-s2r, c2r]])

        rot1_1 = R1 @ array1_direction
        rot1_2 = R2 @ array1_direction
        rot2_1 = R3 @ array2_direction
        rot2_2 = R4 @ array2_direction

        min_dist_1 = min(np.linalg.norm(rot1_1 - rot2_1),
                         np.linalg.norm(rot1_1 - rot2_2))
        min_dist_2 = min(np.linalg.norm(rot1_2 - rot2_1),
                         np.linalg.norm(rot1_2 - rot2_2))

        return rot1_1 if (min_dist_1 < min_dist_2) else rot1_2

###############################################################################

def attitudeEuler(vehicle:Vehicle)->Tuple[NPFltArr,NPFltArr]:
    """
    Integrate the generalized position/Euler angles vector (eta[k+1]), and the
    velocity vector in END reference frame (p_dot).

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with sampleTime, eta, nu attributes.

        - sampleTime: Simulation time step.
        - eta : [x, y, z, phi, theta, psi], vehicle position / attitude vector.
        - nu : [u, v, w, p, q, r], vehicle linear / angular velocity vector in
          BODY frame.
    
          
    Returns
    -------
    eta : ndarray, shape (6,)
        Updated position/attitude [x, y, z, phi, theta, psi].
    p_dot : ndarray, shape (3,)
        Velocity in END frame [x_dot, y_dot, z_dot].

         
    Notes
    -----
    - Uses forward Euler integration.
    - Position integrated from END velocities, attitude from body angular rates
      via transformation matrices.
    - Based on Fossens Python Vehicle Simulator.

    
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """
   
    # Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime
    eta = np.copy(vehicle.eta)
    nu = vehicle.nu

    p_dot   = np.matmul(Rzyx(eta[3], eta[4], eta[5]), nu[0:3])
    v_dot   = np.matmul(Tzyx(eta[3], eta[4]), nu[3:6])

    # Forward Euler integration
    eta[0:3] = eta[0:3] + h * p_dot
    eta[3:6] = eta[3:6] + h * v_dot

    return eta, p_dot

###############################################################################

def Rzyx(phi:float,
         theta:float,
         psi:float,
         )->NPFltArr:
    """
    Compute the 3x3 Euler angle rotation matrix R in SO(3) using the zyx
    convention.

    
    Parameters
    ----------
    phi : float
        Roll angle in radians.
    theta : float
        Pitch angle in radians.
    psi : float
        Yaw angle in radians.
        

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix from BODY to END frame.
    
        
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R = np.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi,  spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R

###############################################################################

def Tzyx(phi:float,theta:float)->NPFltArr:
    """
    Compute the 3x3 Euler angle attitude transformation matrix T using the zyx
    convention.

    
    Parameters
    ----------
    phi : float
        Roll angle in radians.
    theta : float
        Pitch angle in radians.
        

    Returns
    -------
    T : ndarray, shape (3, 3)
        Transformation matrix mapping body angular rates to Euler angle rates.
        

    Notes
    -----
    - Singular at theta = +/-90 degrees. Logs error on singularity.

    
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)    

    try: 
        T = np.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])
        
    except (ZeroDivisionError):  
        log.error("Tzyx is singular for theta = +-90 degrees." )
        
    return T

###############################################################################

def statePT(vehicle:Vehicle,
            pt1:List[float],
            pt2:List[float],
            )->List[float]:
    """
    Compute path-tangential angle and track errors for line segment following.

    Computes the path-tangential (azimuth) angle (pi_h) with respect to the
    East axis, and the along-track (x_e) and cross-track (y_e) errors of a
    vehicle on a path between two points. 

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with eta attribute.
        eta : [x, y, z, phi, theta, psi], vehicle position/attitude vector
    pt1 : list of float, [x, y, z]
        Start point in NED coordinates (m).
    pt2 : list of float, [x, y, z]
        End point in NED coordinates (m).

        
    Returns
    -------
    x_e : float
        Along-track error from start point pt1 (m).
    y_e:
        Cross-track error from path (m).
    pi_h:
        Path-tangential (azimuth) angle w.r.t. East axis (rad).
    
        
    Notes
    -----
    Based on a section in Fossens ALOSpsi.m function.

    
    References
    ----------
    [1] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS). 
    https://github.com/cybergalactic/MSS
    """

    # Input Parameters Loaded on Vehicle
    eta = vehicle.eta

    # Guidance Positions
    x, y = eta[0:2]
    x1, y1 = pt1[0:2]
    x2, y2 = pt2[0:2]

    # Compute the Path-Tangential Angle w.r.t. East
    pi_h = math.atan2((y2 - y1), (x2 - x1))

    # Compute the Along-Track and Cross-Track Errors
    x_e =  (x - x1) * math.cos(pi_h) + (y - y1) * math.sin(pi_h)
    y_e = -(x - x1) * math.sin(pi_h) + (y - y1) * math.cos(pi_h)

    return [x_e, y_e, pi_h]

###############################################################################

def stateSpeed(vehicle:Vehicle)->float:
    """
    Compute vehicle speed magnitude in END frame.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with velocity attribute.
        velocity : [vx, vy, vz], vehicle linear velocity vector in END frame.

          
    Returns
    -------
    speed : float
        Speed magnitude in m/s.
    """

    return np.linalg.norm(vehicle.velocity)

###############################################################################

def headingFilterLOS(vehicle:Vehicle,
                     psi_ref:float,
                     )->None:
    """
    Update the vehicle heading command and yaw rate using LOS observer.

    Propagates heading estimate with feedback from reference angle and
    estimates yaw rate via numerical differentiation.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with sampleTime, psi_d, r_d, K_f attributes.
        
        - sampletime: Simulation time step (s).
        - psi_d: Desired heading angle (rad).
        - r_d: Desired yaw rate (rad/s).
        - K_f: Observer gain for desired yaw angle (typically 0.1-0-5).

    psi_ref : float
        Reference LOS angle (rad) computed from guidance system.

        
    Notes
    -----
    Based on Fossen's LOSobserver.m function. The observer propagates the
    estimate of the LOS angle according to

        psi_d = psi_d + h * (r_d + K_f * ssa(psi_ref - psi_d))
    
    where the yaw rate estimate (r_d) is computed by numerical differentiation

        r_d = T_f * s / (T_f * s + 1) * psi_d

    where T_f is the differentiator time constant, which can be determined by
    pole-placement and inspection of the closed-loop system

        psi_d / psi_ref = w_n^2 * (T_f*s + 1) / (s^2 + 2*w_n*s + w_n^2)

    If K_f > 0, it follows that T_f = 1 / (K_f + 2*sqrt(K_f) + 1) and that the
    natural frequency is w_n = K_f + sqrt(K_f). Exact discretization of the
    observer gives

        r_d = psi_d - xi
        xi = exp(-h/T_f) * xi + (1 - exp(-h/T_f)) * psi_d

        
    References
    ----------
    [1] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
    https://github.com/cybergalactic/MSS
    """

    # Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime      # sample time (s)
    psi_d = vehicle.psi_d       # desired heading angle (rad)
    r_d = vehicle.r_d           # desired yaw rate (rad/s)
    K_f = vehicle.K_f           # observer gain for desired yaw angle

    # Differentiator Time Constant
    T_f = 1 / (K_f + 2 * math.sqrt(K_f) + 1)

    # Internal differentiator state
    xi = psi_d - r_d

    # Observer for the LOS angle
    psi_d += h * (r_d + K_f * gnc.ssa(psi_ref - psi_d))
    vehicle.psi_d = psi_d

    # Propagate the differentiator state
    phi = math.exp(-h / T_f)
    xi = (phi * xi) + ((1 - phi) * psi_d)
    vehicle.r_d = psi_d - xi

###############################################################################

def depthFilter(vehicle:Vehicle,pt:List[float])->None:
    """
    Update vehicle depth command using exponential moving average filter.
    
    Smooths desired depth command with an EMA low-pass filter to reduce control
    chattering and enforce safety limits.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with sampleTime, z_d, wn_d_z, z_max, z_bed, z_safe attributes.

        - sampletime: Simulation time step (s).  
        - z_d: Desired depth command (m)
        - wn_d_z: Desired natural frequency (Hz), depth
        - z_max: Maximum operating depth (m).
        - seabed_z: Sensed ocean floor depth (m).
        - z_safe: Safety distance from ocean floor (m).

    pt : list of float, [x, y, z]
        Target waypoint coordinates in END (m).
    
        
    Notes
    -----
    - Based on Fossens Remus100 autopilot function.
    - Filter: 

        z_d = alpha * z_d_prev + (1 - alpha) * z_target

      where:

        alpha = exp(-h * wn_d_z).

    - Calls maxDepthLimit() to enforce maximum depth limit before filtering.
    - Updates vehicle.z_d in place.

    
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """

    # Input Parameters
    h = vehicle.sampleTime
    z = pt[2]
    z_d = vehicle.z_d
    wn_d_z = vehicle.wn_d_z

    # Enforce Depth Limit
    z = maxDepthLimit(vehicle,z)
        
    # Filter
    alpha = math.exp(-h * wn_d_z)
    vehicle.z_d = (alpha * z_d) + ((1 - alpha) * z)

###############################################################################

def maxDepthLimit(vehicle:Vehicle,z:float)->float:
    """
    Enforce vehicle depth limits based on operating limit and ocean floor depth.

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with z_max, seabed_z, z_safe attributes.

        - z_max: Maximum operating depth (m).
        - seabed_z: Sensed ocean floor depth (m).
        - z_safe: Safety distance from ocean floor (m).

    z : float
        Intended depth in meters.

         
    Returns
    -------
    z_limited : float
        Depth bounded by min(z_max, seabed_z - z_safe).
    """

    # Input Parameters
    z_max = vehicle.z_max
    seabed_z = vehicle.seabed_z
    z_safe = vehicle.z_safe

    # Enforce Maximum Depth Limit
    max_depth = min(z_max, seabed_z - z_safe)
    z = min(z, max_depth)

    return z

###############################################################################