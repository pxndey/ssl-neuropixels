import argparse
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from pathlib import Path
import numpy as np

global_job_kwargs = dict(n_jobs=4, chunk_duration="1s", mp_context="spawn")
si.set_global_job_kwargs(**global_job_kwargs)

def detect_spikes(
    recording_path,
    output_path,
    ):
    # READ RECORDING - pass full file path
    recording_path_obj = Path(recording_path)
    file_suffix = recording_path_obj.suffix.lower()
    
    if file_suffix == ".cbin":
        recording = se.read_cbin_ibl(
            recording_path, stream_name="ap"
        )
    elif file_suffix == ".nwb":
        recording = se.read_nwb_recording(
            file_path=recording_path,
            electrical_series_path="acquisition/ElectricalSeriesAP"
        )
    else:
        recording = se.read_spikeglx(
            recording_path, stream_id="imec0.ap"
        )
    
    # BANDPASS FILTER
    recording = sp.bandpass_filter(
        recording, 
        freq_min=300, 
        freq_max=6000,
        )
    # COMMON MEDIAN REFERENCE
    recording = sp.common_reference(
        recording,
        reference="global",
        operator="median"
        )

    #DETECT SPIKES
    spikes = detect_peaks(
        recording,
        method="locally_exclusive",
    )

    spike_times = spikes["sample_index"]
    spike_channels = spikes["channel_index"]
    spike_times = np.array(spike_times, dtype=np.int64)
    spike_channels = np.array(spike_channels, dtype=np.int64)
    return spike_times, spike_channels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    spike_times, spike_channels = detect_spikes(args.recording_path, output_path)
    np.save(output_path / "spike_times.npy", spike_times)
    np.save(output_path / "spike_channels.npy", spike_channels)
    