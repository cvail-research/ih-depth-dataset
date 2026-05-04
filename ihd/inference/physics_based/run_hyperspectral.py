"""run_hyperspectral.py

Hyperspectral baseline (optimization) inspired by Dorken et al.

This script originally depended on .mat files (attenuation.mat, I_downwelling_res.mat).
In this repo, the coherent version uses:
    - HSI scene: .hdr/.bsq
    - Downwelling: data/standard/downwelling_*.npz (includes wavelength)
    - Attenuation:
      * standard: computed online from data/standard/transmittance_atten_1mAir.npy
      * ozone_cues: loaded from data/ozone_cues/attenuation.npz
    - T_air: estimated via brightness temperature at the maximum-attenuation band
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

from ihd.inference.physics_based.utils.io_utils import load_lidar, load_scene
from ihd.inference.physics_based.utils.metrics import evaluate, save_results
from ihd.inference.physics_based.utils.physics import estimate_T_air
from ihd.inference.physics_based.utils.vis import save_distance_png, save_error_png

V_mean = .0
T_mean = 300
emissivity_mean = 0.9
d_mean = 120

V_std = 1
T_std = 10
emissivity_std = 10
d_std = 1000

class BlackbodyFunction(Function):
    @staticmethod
    def forward(ctx, lambda_um, T):
        """
        Forward pass for the blackbody function.
        """
        ctx.save_for_backward(lambda_um, T)

        # Convert wavelength from micrometers to meters
        lambda_m = lambda_um * 1e-6

        h = 6.63e-34  # Planck's constant (J*s)
        c = 3e8  # Speed of light (m/s)
        k_B = 1.38e-23  # Boltzmann constant (J/K)

        # Calculate exponent term and clamp to avoid overflow
        exponent = h * c / (lambda_m * k_B * T)
        exponent = torch.clamp(exponent, max=700)  # Clamp to prevent overflow
        exp_term = torch.exp(exponent)

        # Calculate microflicks (spectral radiance)
        microflicks = 1e-4 * (2 * h * c ** 2 / (lambda_m ** 5 * (exp_term - 1)))

        return microflicks

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the blackbody function.
        """
        lambda_um, T = ctx.saved_tensors
        lambda_m = lambda_um * 1e-6

        h = 6.63e-34  # Planck's constant (J*s)
        c = 3e8  # Speed of light (m/s)
        k_B = 1.38e-23  # Boltzmann constant (J/K)

        # Calculate exponent term and clamp to avoid overflow
        exponent = h * c / (lambda_m * k_B * T)
        exponent = torch.clamp(exponent, max=700)  # Clamp to prevent overflow
        exp_term = torch.exp(exponent)

        # Compute the gradient with respect to temperature T
        # dT is the partial derivative of the blackbody function w.r.t T
        dI_dT = (2 * h * c ** 2 / lambda_m ** 5) * (exp_term * exponent / (T ** 2 * (exp_term - 1) ** 2))

        # Multiply by the incoming gradient from the next layer
        dT = dI_dT * grad_output

        return None, dT

def load_data_from_hdr(hsi_hdr: Path, data_dir: Path, *, downwelling_flag: bool, t_air: float | None,
                       lambda_min: float, lambda_max: float, attenuation_profile: str):
    meas, lambda_um, attenuation, downwelling, _sensor = load_scene(
        str(hsi_hdr),
        str(data_dir / "ozone_cues"),
        str(data_dir / "standard"),
        attenuation_profile=attenuation_profile,
    )

    if t_air is None:
        t_air_est, _ = estimate_T_air(meas, lambda_um, attenuation, lambda_min=lambda_min, lambda_max=lambda_max)
        T_air = float(t_air_est)
    else:
        T_air = float(t_air)
        print(f"  T_air (manual): {T_air:.2f} K")

    # Shapes expected by the solver (compatible with the original code)
    # meas: (H,W,K,1)
    HSI = meas.astype(np.float32)[:, :, :, None]
    # wavelength: (1,1,K,1)
    wavelength = lambda_um.astype(np.float32).reshape(1, 1, -1, 1)
    # attenuation: (1,1,K,1)
    attenuation4 = attenuation.astype(np.float32).reshape(1, 1, -1, 1)
    # downwelling: (1,1,K,L)
    dw = downwelling.astype(np.float32).T  # (K,L)
    dw_r = dw.reshape(1, 1, dw.shape[0], dw.shape[1])
    if not downwelling_flag:
        dw_r = np.zeros_like(dw_r)

    return HSI, wavelength, dw_r, attenuation4, T_air

def standardize_data(data, mean, std):
    return (data - mean) / std

def destandardize_data(data, mean, std):
    return (data * std) + mean


def compute_incident_light(V, wavelength, dw_r, T_env):
    """
    Calculate the incident light based on the blackbody radiation and provided parameters.

    Parameters:
    V (torch.Tensor): Light intensity with shape (batch_size, height, width, num_wavelengths).
    wavelength (torch.Tensor): Wavelengths in micrometers with shape (num_wavelengths,).
    dw_r (torch.Tensor): Angular direction with shape (batch_size, height, width, num_wavelengths).
    T_env (float): Environmental temperature in Kelvin.

    Returns:
    torch.Tensor: Incident light with shape (batch_size, height, width, 1).
    """
    # Calculate the blackbody spectral radiance
    bb_env = BlackbodyFunction.apply(wavelength, T_env)

    # Concatenate along the last dimension to match dw_r
    light_dir = torch.cat((dw_r, bb_env), dim=3)  # Shape: (batch_size, height, width, num_wavelengths + 1)

    # Calculate the incident light by summing over the last dimension
    incident_light = torch.sum(V * light_dir, dim=3, keepdim=True)

    return incident_light


def forward_model(wavelength, T, emissivity, attenuation, d, T_air, incident_light):

    # Compute the object emission
    obj_emission = BlackbodyFunction.apply(wavelength, T) * emissivity

    # Compute the object reflection
    obj_reflection = incident_light * (1 - emissivity)

    # Compute the attenuation factor
    tau = 10 ** (-attenuation * d / 10)

    # Compute the sensor radiance
    sensor_radiance = tau * (obj_emission + obj_reflection) + (1 - tau) * BlackbodyFunction.apply(wavelength, T_air)
    return sensor_radiance


def l2_loss(measured, model_output):
    return F.mse_loss(model_output, measured)


def tikhonov_regularization(emissivity, alpha=1.0):
    """
    Compute the Tikhonov regularization loss by taking the circular difference along axis 2,
    squaring it, and summing the results.

    Parameters:
    - emissivity (Tensor): The emissivity values with shape [M, N, K, 1].
    - alpha (float): Regularization parameter (lambda).

    Returns:
    - Tensor: Regularization loss.
    """
    # Compute the forward difference along axis 2
    diff = emissivity[:, :, 1:, :] - emissivity[:, :, :-1, :]

    # Compute the circular difference between the last and first element along axis 2
    circular_diff = emissivity[:, :, 0, :] - emissivity[:, :, -1, :]

    # Concatenate forward differences with circular difference
    diff = torch.cat([diff, circular_diff.unsqueeze(2)], dim=2)

    # Square the differences
    squared_diff = diff ** 2

    # Sum the squared differences
    reg_term = torch.mean(squared_diff)

    return alpha * reg_term

def total_variation_regularization(d, alpha=1.0):
    """
    Compute the Total Variation (TV) regularization loss for the parameter d using absolute differences.

    Parameters:
    - d (Tensor): The parameter d with shape [M, N, 1, 1].
    - alpha (float): Regularization parameter (lambda).

    Returns:
    - Tensor: Regularization loss.
    """
    # Compute the forward difference along dim=0
    diff_d0 = d[:, 1:, :, :] - d[:, :-1, :, :]

    # Compute the forward difference along dim=1
    diff_d1 = d[1:, :, :, :] - d[:-1, :, :, :]

    # Compute the absolute differences
    abs_diff_d0 = torch.abs(diff_d0)
    abs_diff_d1 = torch.abs(diff_d1)

    # Sum the absolute differences
    reg_term_d0 = torch.mean(abs_diff_d0)
    reg_term_d1 = torch.mean(abs_diff_d1)

    # Total variation regularization loss
    reg_term = alpha * (reg_term_d0 + reg_term_d1)

    return reg_term


def total_loss(measured, model_output, emissivity, d, alpha=1.0, alpha_2=1.0):

    l2 = l2_loss(measured, model_output)
    reg = tikhonov_regularization(emissivity, alpha)
    reg2 = total_variation_regularization(d, alpha_2)
    return l2 + reg + reg2


def solve(wavelength, dw_r, T_env, measured, attenuation, T_air, num_iterations=100, lr=0.1, alpha=1.0, alpha_2 = 0, start_point = None, optimizer_type='SGD'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move tensors to GPU
    wavelength = wavelength.to(device)
    dw_r = dw_r.to(device)
    T_env = torch.tensor(T_env, dtype=torch.float32, device=device)
    T_air = torch.tensor(T_air, dtype=torch.float32, device=device)
    measured = measured.to(device)
    attenuation = attenuation.to(device, dtype=torch.float32)

    if start_point:
        V = torch.tensor(start_point['V'], dtype=torch.float32, device=device, requires_grad=True)
        T = torch.tensor(start_point['T'], dtype=torch.float32, device=device, requires_grad=True)
        emissivity = torch.tensor(start_point['emissivity'], dtype=torch.float32, device=device, requires_grad=True)
        d = torch.tensor(start_point['d'], dtype=torch.float32, device=device, requires_grad=True)
        print(V.shape)
        print(T.shape)
        print(emissivity.shape)
        print(d.shape)

        print("Initialized from starting point")
    else:

        # # Initialize parameters
        # V = torch.full((measured.shape[0], measured.shape[1], 1, 7), 1e-3, dtype=torch.float32, device=device,
        #            requires_grad=True)  # Estimated V
        # T = torch.full((measured.shape[0], measured.shape[1], 1, 1), 295, dtype=torch.float32, device=device,
        #            requires_grad=True)  # Temperature
        # emissivity = torch.full((measured.shape[0], measured.shape[1], measured.shape[2], 1), 1, dtype=torch.float32, device=device,
        #            requires_grad=True)  # Temperature
        # d = torch.full((measured.shape[0], measured.shape[1], 1, 1), 200, dtype=torch.float32, device=device, requires_grad=True)

        # Initialize parameters
        V = torch.full((measured.shape[0], measured.shape[1], 1, 11), 0, dtype=torch.float32, device=device,
                       requires_grad=True)  # Estimated V
        T = torch.full((measured.shape[0], measured.shape[1], 1, 1), 0, dtype=torch.float32, device=device,
                       requires_grad=True)  # Temperature
        emissivity = torch.full((measured.shape[0], measured.shape[1], measured.shape[2], 1), 0, dtype=torch.float32,
                                device=device,
                                requires_grad=True)  # Temperature
        d = torch.full((measured.shape[0], measured.shape[1], 1, 1), 0, dtype=torch.float32, device=device,
                       requires_grad=True)

        print("No starting point")


    # Optimizer
    if optimizer_type == 'SGD':
        optimizer = optim.SGD([V, T, emissivity, d], lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam([V, T, emissivity, d], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Lists to store losses
    total_losses = []
    l2_losses = []
    reg_losses = []

    iter_t0 = time.perf_counter()
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        V_r = destandardize_data(V, V_mean, V_std)
        T_r = destandardize_data(T, T_mean, T_std)
        d_r = destandardize_data(d, d_mean, d_std)
        emissivity_r = destandardize_data(emissivity, emissivity_mean, emissivity_std)

        # Compute incident light
        incident_light = compute_incident_light(V_r, wavelength, dw_r, T_env)

        # Compute model output
        model_output = forward_model(wavelength, T_r, emissivity_r, attenuation, d_r, T_air, incident_light)

        # Compute losses
        #loss = total_loss(measured, model_output, emissivity_r, d_r, alpha, alpha_2=5e-8)
        loss = total_loss(measured, model_output, emissivity_r, d_r, alpha, alpha_2)
        l2_loss_value = l2_loss(measured, model_output).item()
        reg_loss_value = tikhonov_regularization(emissivity, alpha).item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_([emissivity], max_norm=1.0)

        # Update parameters
        optimizer.step()

        # # Apply constraints
        with torch.no_grad():

            # Ensure V is positive
            V.data = torch.relu(destandardize_data(V.data, V_mean, V_std))

            # Normalize V to ensure sum along the third axis is less than 1
            V_sum = V.data.sum(dim=3, keepdim=True)
            V.data = torch.where(V_sum > 1, V.data / V_sum * 0.99, V.data)  # Scale to ensure sum < 1
            V.data = standardize_data(V.data, V_mean, V_std)

            # Clamp T to avoid NaNs
            T.data = destandardize_data(T.data, T_mean, T_std)
            T.data = torch.clamp(T.data, min=0.0, max=400)  # Adjust the range as needed
            T.data = standardize_data(T.data, T_mean, T_std)

            # Clamp d to avoid NaNs
            d.data = destandardize_data(d.data, d_mean, d_std)
            d.data = torch.clamp(d.data, min=0.0, max= 200)  # Adjust the range as needed
            d.data = standardize_data(d.data, d_mean, d_std)

            # Emiss d to avoid NaNs
            emissivity.data =  destandardize_data(emissivity.data, emissivity_mean, emissivity_std)
            emissivity.data = torch.clamp(emissivity.data, min=0.0, max=1)  # Adjust the range as needed
            emissivity.data = standardize_data(emissivity.data, emissivity_mean, emissivity_std)

        #Store losses
        total_losses.append(loss.item())
        l2_losses.append(l2_loss_value)
        reg_losses.append(reg_loss_value)

        if iteration % 1000 == 0:
            elapsed_s = time.perf_counter() - iter_t0
            ts = dt.datetime.now().isoformat(timespec="seconds")
            print(
                f"[{ts}] Iteration {iteration}/{num_iterations}, Loss: {loss.item()}, Elapsed_s: {elapsed_s:.3f}",
                flush=True,
            )

    return V_r.detach().cpu().numpy(), T_r.detach().cpu().numpy(), emissivity_r.detach().cpu().numpy(), d_r.detach().cpu().numpy()

def solve_full_scene(hsi_hdr: Path, data_dir: Path, downwelling_flag=True, chunk_size=128, lr=1e-2,
                     num_iterations=100000, emiss_reg=1e7, TV_reg=1e-4, t_air: float | None = None,
                     lambda_min: float = 8.5, lambda_max: float = 12.0,
                     attenuation_profile: str = "auto"):
    HSI, wavelength, dw_r, attenuation, T_air = load_data_from_hdr(
        hsi_hdr,
        data_dir,
        downwelling_flag=downwelling_flag,
        t_air=t_air,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        attenuation_profile=attenuation_profile,
    )

    # Crop the first and second dimensions to be multiples of the chunk size
    crop_size_0 = (HSI.shape[0] // chunk_size) * chunk_size
    crop_size_1 = (HSI.shape[1] // chunk_size) * chunk_size
    HSI = HSI[:crop_size_0, :crop_size_1, :, :]

    # Converting the data to PyTorch tensors
    HSI = torch.from_numpy(HSI).float()
    wavelength = torch.from_numpy(wavelength).float()
    attenuation = torch.from_numpy(attenuation).float()
    dw_r = torch.from_numpy(dw_r).float()
    

    T_env = 0  # Environmental temperature (ignore)

    # Initialize tensors to store the results
    V_full = torch.zeros(HSI.shape[0], HSI.shape[1], 1, 11)
    T_full = torch.zeros(HSI.shape[0], HSI.shape[1], 1, 1)
    emissivity_full = torch.zeros(HSI.shape[0], HSI.shape[1], HSI.shape[2], 1)
    d_full = torch.zeros(HSI.shape[0], HSI.shape[1], 1, 1)

    # Process data in chunks along the first and second dimensions
    for i in range(0, HSI.shape[0], chunk_size):
        for j in range(0, HSI.shape[1], chunk_size):
            # Define the chunk
            HSI_chunk = HSI[i:i + chunk_size, j:j + chunk_size, :, :]

            # Solve the optimization problem for the chunk
            if downwelling_flag:
                V, T, emissivity, d = solve(wavelength, dw_r, T_env, HSI_chunk, attenuation, num_iterations=num_iterations,
                                         T_air=T_air, lr=lr, alpha=emiss_reg, alpha_2=TV_reg, start_point=None, optimizer_type='SGD')
            else:
                V, T, emissivity, d = solve(wavelength, dw_r, T_env, HSI_chunk, attenuation, num_iterations=num_iterations,
                                         T_air=T_air, lr=lr, alpha=emiss_reg, alpha_2=TV_reg, start_point=None, optimizer_type='Adam')
            # Convert the results to PyTorch tensors
            V = torch.from_numpy(V).float()
            T = torch.from_numpy(T).float()
            emissivity = torch.from_numpy(emissivity).float()
            d = torch.from_numpy(d).float()

            # Store the results in the full tensors
            V_full[i:i + chunk_size, j:j + chunk_size, :, :] = V
            T_full[i:i + chunk_size, j:j + chunk_size, :, :] = T
            emissivity_full[i:i + chunk_size, j:j + chunk_size, :, :] = emissivity
            d_full[i:i + chunk_size, j:j + chunk_size, :, :] = d

    return V_full.cpu().numpy(), T_full.cpu().numpy(), emissivity_full.cpu().numpy(), d_full.cpu().numpy(), float(T_air)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default_data = here / "data"
    p.add_argument("--hsi-hdr", type=Path, required=True, help="Path to the scene .hdr")
    p.add_argument("--data-dir", type=Path, default=default_data, help="Data directory (default: baselines/.../data)")
    p.add_argument(
        "--attenuation-profile",
        choices=["auto", "standard", "ozone_cues"],
        default="auto",
        help="Attenuation source profile.",
    )
    p.add_argument("--out-dir", type=Path, default=here / "outputs", help="Output directory")
    p.add_argument("--lidar-mat", type=Path, default=None, help="Path to lidar.mat (optional, for evaluation)")
    p.add_argument("--downwelling", action=argparse.BooleanOptionalAction, default=True,
                   help="Use downwelling data (use --no-downwelling to ignore it)")
    p.add_argument("--chunk-size", type=int, default=128, help="GPU pixel chunk size")
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optimizer learning rate. If omitted: 0.01 with downwelling, 0.0005 without.",
    )
    p.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Number of optimization iterations. If omitted: 100000 with downwelling, 20000 without.",
    )
    p.add_argument("--emiss-reg", type=float, default=1e7, help="Emissivity smoothness regularization")
    p.add_argument("--tv-reg", type=float, default=1e-4, help="Total-variation regularization on distance d")
    p.add_argument("--t-air", type=float, default=None, help="Set T_air manually (K). If omitted, it is estimated")
    p.add_argument("--lambda-min", type=float, default=8.5, help="Range for automatic T_air estimation")
    p.add_argument("--lambda-max", type=float, default=12.0, help="Range for automatic T_air estimation")
    p.add_argument("--save-npy", action="store_true", help="Save the estimated distance map as .npy")
    p.add_argument("--save-fig", action="store_true", help="Save a PNG visualization of the distance map")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    downwelling_flag = bool(args.downwelling)
    lr = float(args.lr) if args.lr is not None else (1e-2 if downwelling_flag else 5e-4)
    num_iterations = int(args.num_iterations) if args.num_iterations is not None else (100000 if downwelling_flag else 20000)

    print(
        "  Hyperspectral settings: "
        f"downwelling={downwelling_flag}, chunk_size={int(args.chunk_size)}, "
        f"lr={lr:g}, num_iterations={num_iterations}, emiss_reg={float(args.emiss_reg):g}, tv_reg={float(args.tv_reg):g}"
    )

    V, T, emissivity, d, T_air = solve_full_scene(
        args.hsi_hdr,
        args.data_dir,
        downwelling_flag=downwelling_flag,
        chunk_size=int(args.chunk_size),
        lr=lr,
        num_iterations=num_iterations,
        emiss_reg=float(args.emiss_reg),
        TV_reg=float(args.tv_reg),
        t_air=args.t_air,
        lambda_min=float(args.lambda_min),
        lambda_max=float(args.lambda_max),
        attenuation_profile=args.attenuation_profile,
    )

    # d: (H,W,1,1)
    d_map = d[:, :, 0, 0].astype(np.float64)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_name = args.hsi_hdr.stem
    method = "hyperspectral"

    if args.save_npy:
        out_npy = out_dir / f"{scene_name}_{method}_d.npy"
        np.save(out_npy, d_map.astype(np.float32))
        print(f"  Saved: {out_npy}")

    if args.save_fig:
        out_png = out_dir / f"{scene_name}_{method}_d.png"
        save_distance_png(d_map, out_png, title=f"{scene_name} | {method}")
        print(f"  Saved: {out_png}")

    if args.lidar_mat is not None:
        gt = load_lidar(str(args.lidar_mat))
        if gt.shape != d_map.shape:
            raise ValueError(f"Shape mismatch: pred={d_map.shape} vs gt={gt.shape}")
        if args.save_fig:
            out_err_png = out_dir / f"{scene_name}_{method}_error.png"
            save_error_png(d_map, gt, out_err_png, title=f"{scene_name} | {method} | error")
            print(f"  Saved: {out_err_png}")
        results = evaluate(d_map, gt, method_name=method, verbose=True)
        results["T_air"] = float(T_air)
        save_results(results, str(out_dir), scene_name, method)

if __name__ == "__main__":
    main()
