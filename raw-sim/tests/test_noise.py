import torch

from raw_sim.noise import analog_gain, iso_value, noise_params, noise_std_map, simulate_noise


def camera_with_gain(gain: float, binning_type: str = "none"):
    return {
        "camera": {"analog_gain": gain},
        "sensor": {"bit_depth": 10, "black_level": "auto", "black_level_10bit_reference": 32, "white_level": "auto"},
        "cfa": {"type": "binning" if binning_type != "none" else "bayer", "pattern": "RGGB"},
        "readout": {"mode": "full", "binning_type": binning_type, "binning_operation": "average"},
        "noise": {
            "strength_multiplier": 1.0,
            "calibration": {
                "base_analog_gain": 1.0,
                "channels": {
                    "R": {"K": 0.02, "sigma_read": 0.001},
                    "Gr": {"K": 0.02, "sigma_read": 0.001},
                    "Gb": {"K": 0.02, "sigma_read": 0.001},
                    "B": {"K": 0.02, "sigma_read": 0.001},
                    "bin_R": {"K": 0.02, "sigma_read": 0.001},
                    "bin_Gr": {"K": 0.02, "sigma_read": 0.001},
                    "bin_Gb": {"K": 0.02, "sigma_read": 0.001},
                    "bin_B": {"K": 0.02, "sigma_read": 0.001},
                },
            },
        },
    }


def test_iso_and_noise_params_follow_analog_gain():
    camera = camera_with_gain(8.0)
    params = noise_params(camera, ["R"])
    assert iso_value(camera) == 800
    assert params["R"]["K"] == 0.16
    assert params["R"]["sigma_read"] == 0.008


def test_camera_json_analog_gain_controls_noise_level():
    camera = camera_with_gain(16.0)
    params = noise_params(camera, ["R"])
    assert analog_gain(camera) == 16.0
    assert iso_value(camera) == 1600
    assert params["R"]["K"] == 0.32
    assert params["R"]["sigma_read"] == 0.016


def test_noise_variance_increases_with_signal():
    camera = camera_with_gain(1.0)
    params = noise_params(camera, ["R", "Gr", "Gb", "B"])
    names = ["R", "Gr", "Gb", "B"]
    low = torch.full((128, 128, 4), 0.05)
    high = torch.full((128, 128, 4), 0.8)
    noisy_low = simulate_noise(low, params, names, camera, samples_per_channel=1, seed=1)
    noisy_high = simulate_noise(high, params, names, camera, samples_per_channel=1, seed=1)
    assert (noisy_high - high).var() > (noisy_low - low).var()


def test_noise_std_map_is_spatial_and_single_channel():
    camera = camera_with_gain(1.0)
    names = ["R", "Gr", "Gb", "B"]
    params = noise_params(camera, names)
    clean = torch.ones(4, 5, 4) * 0.5
    std = noise_std_map(clean, params, names, camera, samples_per_channel=1)
    assert std.shape == (4, 5, 1)
    assert torch.all(std > 0)


def test_shot_noise_uses_black_level_subtracted_signal():
    camera = camera_with_gain(1.0)
    names = ["R", "Gr", "Gb", "B"]
    params = noise_params(camera, names)
    black = 32.0 / 1023.0
    at_black = torch.full((2, 2, 4), black)
    above_black = torch.full((2, 2, 4), black + 0.5)
    std_black = noise_std_map(at_black, params, names, camera, samples_per_channel=1, reduce="none")
    std_signal = noise_std_map(above_black, params, names, camera, samples_per_channel=1, reduce="none")
    expected_read = torch.full_like(std_black, 0.001)
    assert torch.allclose(std_black, expected_read, atol=1e-6)
    assert torch.all(std_signal > std_black)
