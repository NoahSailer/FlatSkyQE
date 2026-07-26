import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


DEFAULT_CONFIG = {
    "lmax_range": [50000.0, 60000.0, 70000.0, 90000.0],
    "psf_fwhm_range": [None],
    "nsim": 5,
    "pixel_fn_correct": False,
    "lensmode": "unlensed",
    "nbar": 1.0e5,
    "exact_beam": True,
    "use_precomp": False,
    "save": True,
    "plot": True,
    "add_noise": True,
    "apply_mask": True,
    "use_beam": True,
    "lmin": 1.0e4,
    "N_CIB_PER_PIXEL": 0.2,
    "N_G_PER_PIXEL": 0.2,
    "alpha": 0.0,
    "skew_filter_mode": "bandpass",
    "s_max": 10.0,
    "datestr": "122625",
    "res_root": "res",
    "fig_root": "figures",
    "verbose": 1,
    "save_intermediate_plots": False,
    "enable_lensing": False,
    "kappa_amplitude": 1.0,
    "kappa_seed": 12345,
    "mode": "default",
}


def _repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, ".."))


def _ensure_repo_on_path() -> None:
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def _parse_optional_float(value: str) -> Optional[float]:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _coerce_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for {key}: {value!r}")


def _coerce_optional_float(value: Any, key: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value for {key}: {value!r}") from exc


def _ensure_list(value: Any, key: str) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_config_types(cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(cfg)

    normalized["lmax_range"] = [
        float(x) for x in _ensure_list(normalized["lmax_range"], "lmax_range")
    ]
    normalized["psf_fwhm_range"] = [
        _coerce_optional_float(x, "psf_fwhm_range")
        for x in _ensure_list(normalized["psf_fwhm_range"], "psf_fwhm_range")
    ]

    normalized["nsim"] = int(normalized["nsim"])
    normalized["verbose"] = int(normalized["verbose"])
    normalized["nbar"] = float(normalized["nbar"])
    normalized["lmin"] = float(normalized["lmin"])
    normalized["N_CIB_PER_PIXEL"] = float(normalized["N_CIB_PER_PIXEL"])
    normalized["N_G_PER_PIXEL"] = float(normalized["N_G_PER_PIXEL"])
    normalized["alpha"] = float(normalized["alpha"])
    normalized["s_max"] = float(normalized["s_max"])

    normalized["lensmode"] = str(normalized["lensmode"]).strip()
    normalized["skew_filter_mode"] = str(normalized["skew_filter_mode"]).strip()
    normalized["datestr"] = str(normalized["datestr"]).strip()
    normalized["res_root"] = str(normalized["res_root"]).strip()
    normalized["fig_root"] = str(normalized["fig_root"]).strip()
    normalized["mode"] = str(normalized["mode"]).strip()

    normalized["kappa_amplitude"] = float(normalized["kappa_amplitude"])
    normalized["kappa_seed"] = int(normalized["kappa_seed"])

    for bool_key in [
        "pixel_fn_correct",
        "exact_beam",
        "use_precomp",
        "save",
        "plot",
        "add_noise",
        "apply_mask",
        "use_beam",
        "save_intermediate_plots",
        "enable_lensing",
    ]:
        normalized[bool_key] = _coerce_bool(normalized[bool_key], bool_key)

    return normalized


def _load_config_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return json.loads(raw)

    if ext in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with `pip install pyyaml` or use JSON config."
            ) from exc
        loaded = yaml.safe_load(raw)
        return loaded or {}

    raise ValueError("Config file must end in .json, .yml, or .yaml")


def _slugify_float(value: float) -> str:
    if value >= 1000 and float(int(value)) == value:
        return f"{int(value)}"
    return str(value).replace(".", "p")


def _config_subdir(cfg: Dict[str, Any]) -> str:
    psf_vals = cfg["psf_fwhm_range"]
    if len(psf_vals) == 1:
        if psf_vals[0] is None:
            psf_tag = "ciberbeam" if cfg["use_beam"] else "deltatn"
        else:
            psf_tag = f"psf{_slugify_float(psf_vals[0])}"
    else:
        psf_tag = "psf-sweep"

    if cfg["enable_lensing"]:
        lensmode_tag = f"lens-lensed-amp{_slugify_float(cfg['kappa_amplitude'])}"
    else:
        lensmode_tag = f"lens-{cfg['lensmode']}"

    mode = cfg.get("mode", "default")
    tags = [
        f"mode-{mode}",
        lensmode_tag,
        f"nbar-{_slugify_float(float(cfg['nbar']))}",
        psf_tag,
        "noise" if cfg["add_noise"] else "no-noise",
        "mask" if cfg["apply_mask"] else "no-mask",
    ]
    return "_".join(tags)


def _build_resname(cfg: Dict[str, Any], lmax: float, psf_pix_fwhm: Optional[float]) -> str:
    grab_cib_sim = psf_pix_fwhm is None
    resname = "clkg_" + cfg["lensmode"] + "_srcs_with_prediction_lmin" + str(int(cfg["lmin"]))
    resname += "_lmax" + str(int(lmax))

    if grab_cib_sim:
        resname += "_CIBERbeam"
    else:
        resname += "_PSFFWHM=" + str(psf_pix_fwhm)
    resname += "_wnoise" if cfg["add_noise"] else "_noiseless"
    resname += "_wmask" if cfg["apply_mask"] else "_unmasked"
    if cfg["exact_beam"]:
        resname += "_exactbeam"
    if cfg["pixel_fn_correct"]:
        resname += "_wpixcorr"
    return resname


def _resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        cfg.update(_load_config_file(args.config))

    overrides = {
        "lmax_range": args.lmax_range,
        "psf_fwhm_range": args.psf_fwhm_range,
        "nsim": args.nsim,
        "verbose": args.verbose,
        "nbar": args.nbar,
        "lensmode": args.lensmode,
        "lmin": args.lmin,
        "datestr": args.datestr,
        "res_root": args.res_root,
        "fig_root": args.fig_root,
        "mode": args.mode,
    }

    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    # Tri-state bool flags: only override when user supplied one.
    if args.add_noise is not None:
        cfg["add_noise"] = args.add_noise
    if args.apply_mask is not None:
        cfg["apply_mask"] = args.apply_mask
    if args.exact_beam is not None:
        cfg["exact_beam"] = args.exact_beam
    if args.pixel_fn_correct is not None:
        cfg["pixel_fn_correct"] = args.pixel_fn_correct
    if args.use_precomp is not None:
        cfg["use_precomp"] = args.use_precomp
    if args.plot is not None:
        cfg["plot"] = args.plot
    if args.save is not None:
        cfg["save"] = args.save
    if args.save_intermediate_plots is not None:
        cfg["save_intermediate_plots"] = args.save_intermediate_plots
    if args.use_beam is not None:
        cfg["use_beam"] = args.use_beam
    if args.enable_lensing is not None:
        cfg["enable_lensing"] = args.enable_lensing
    if args.kappa_amplitude is not None:
        cfg["kappa_amplitude"] = args.kappa_amplitude
    if args.kappa_seed is not None:
        cfg["kappa_seed"] = args.kappa_seed

    return _normalize_config_types(cfg)


def _validate_config(cfg: Dict[str, Any]) -> None:
    if not cfg["lmax_range"]:
        raise ValueError("lmax_range cannot be empty")
    if not cfg["psf_fwhm_range"]:
        raise ValueError("psf_fwhm_range cannot be empty")
    if cfg["nsim"] < 1:
        raise ValueError("nsim must be >= 1")
    if cfg["lmin"] <= 0:
        raise ValueError("lmin must be > 0")
    if cfg["verbose"] < 0:
        raise ValueError("verbose must be >= 0")
    if any(lmax <= cfg["lmin"] for lmax in cfg["lmax_range"]):
        raise ValueError("All lmax values must be > lmin")
    if cfg["kappa_amplitude"] <= 0:
        raise ValueError("kappa_amplitude must be > 0")
    if cfg["kappa_seed"] < 0:
        raise ValueError("kappa_seed must be >= 0")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FlatSkyQE mock tests as a configurable sweep.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to JSON/YAML config file")

    parser.add_argument("--lmax-range", nargs="+", type=float, default=None, help="List of lMax values")
    parser.add_argument(
        "--psf-fwhm-range",
        nargs="+",
        type=_parse_optional_float,
        default=None,
        help="List of PSF FWHM values in pixels; use 'none' for CIBER beam",
    )
    parser.add_argument("--nsim", type=int, default=None, help="Number of simulations")
    parser.add_argument("--verbose", type=int, default=None, help="Verbosity level (0=quiet, 1=normal, 2=debug)")
    parser.add_argument("--nbar", type=float, default=None, help="Source number density")
    parser.add_argument("--lensmode", type=str, default=None, help="Lens mode string")
    parser.add_argument("--lmin", type=float, default=None, help="Minimum ell")
    parser.add_argument("--datestr", type=str, default=None, help="Date string for output filenames")
    parser.add_argument("--res-root", type=str, default=None, help="Root directory for result files")
    parser.add_argument("--fig-root", type=str, default=None, help="Root directory for figure files")
    parser.add_argument("--mode", type=str, default=None, help="Estimator mode (e.g., qe_kappa_ln_hardened)")

    parser.add_argument("--add-noise", dest="add_noise", action="store_true")
    parser.add_argument("--no-noise", dest="add_noise", action="store_false")
    parser.set_defaults(add_noise=None)

    parser.add_argument("--apply-mask", dest="apply_mask", action="store_true")
    parser.add_argument("--no-mask", dest="apply_mask", action="store_false")
    parser.set_defaults(apply_mask=None)

    parser.add_argument("--exact-beam", dest="exact_beam", action="store_true")
    parser.add_argument("--approx-beam", dest="exact_beam", action="store_false")
    parser.set_defaults(exact_beam=None)

    parser.add_argument("--pixel-fn-correct", dest="pixel_fn_correct", action="store_true")
    parser.add_argument("--no-pixel-fn-correct", dest="pixel_fn_correct", action="store_false")
    parser.set_defaults(pixel_fn_correct=None)

    parser.add_argument("--use-precomp", dest="use_precomp", action="store_true")
    parser.add_argument("--no-use-precomp", dest="use_precomp", action="store_false")
    parser.set_defaults(use_precomp=None)

    parser.add_argument("--save", dest="save", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.set_defaults(save=None)

    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.set_defaults(plot=None)

    parser.add_argument("--save-intermediate-plots", dest="save_intermediate_plots", action="store_true")
    parser.add_argument("--no-save-intermediate-plots", dest="save_intermediate_plots", action="store_false")
    parser.set_defaults(save_intermediate_plots=None)

    parser.add_argument("--use-beam", dest="use_beam", action="store_true")
    parser.add_argument("--no-beam", dest="use_beam", action="store_false")
    parser.set_defaults(use_beam=None)

    parser.add_argument("--enable-lensing", dest="enable_lensing", action="store_true")
    parser.add_argument("--no-lensing", dest="enable_lensing", action="store_false")
    parser.set_defaults(enable_lensing=None)

    parser.add_argument("--kappa-amplitude", type=float, default=None, help="Lensing kappa amplitude scalar (default 1.0)")
    parser.add_argument("--kappa-seed", type=int, default=None, help="Seed for kappa realization (default 12345)")

    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and planned jobs")
    return parser


@dataclass
class SweepJob:
    lmax: float
    psf_pix_fwhm: Optional[float]


class TestSweepRunner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.repo_root = _repo_root()
        self.subdir = _config_subdir(cfg)

        self.res_dir = os.path.join(self.repo_root, cfg["res_root"], self.subdir)
        self.fig_dir = os.path.join(self.repo_root, cfg["fig_root"], self.subdir)
        self.intermediate_dir = os.path.join(self.res_dir, "intermediate_plots")
        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        if self.cfg["save_intermediate_plots"]:
            os.makedirs(self.intermediate_dir, exist_ok=True)

    def generate_test_configs(self) -> List[SweepJob]:
        jobs: List[SweepJob] = []
        for lmax in self.cfg["lmax_range"]:
            for psf in self.cfg["psf_fwhm_range"]:
                jobs.append(SweepJob(lmax=lmax, psf_pix_fwhm=psf))
        return jobs

    def run_single_test(self, job: SweepJob) -> str:
        from mock_lens_test import delta_fn_sources_test
        from kappa_plotting_fns import gen_suptitle, plot_recov_components, plot_input_recovered_kappa

        cfg = self.cfg
        verbose = cfg["verbose"]
        # grab_cib_sim means load CIBER beam from simulation; only do this if use_beam=True
        grab_cib_sim = job.psf_pix_fwhm is None and cfg["use_beam"]
        sigma_noise_pix = 40 if (job.psf_pix_fwhm is None) else 5

        resname = _build_resname(cfg, job.lmax, job.psf_pix_fwhm)
        res_fpath = os.path.join(self.res_dir, f"res_{resname}_{cfg['datestr']}.npz")

        if os.path.exists(res_fpath) and cfg["use_precomp"]:
            if verbose >= 1:
                print("Loading pre-computed configuration results from", res_fpath)
            loaded = np.load(res_fpath, allow_pickle=True)
            res = {key: loaded[key] for key in loaded.files}
        else:
            if verbose >= 1:
                print("Computing configuration:", resname)
            mockstr = "noiseless_JHlt16_nbar" + str(cfg["nbar"])
            intermediate_plot_dir = None
            if cfg["save_intermediate_plots"]:
                intermediate_plot_dir = os.path.join(
                    self.intermediate_dir,
                    f"{resname}_{cfg['datestr']}",
                )
            res = delta_fn_sources_test(
                nsim=cfg["nsim"],
                lMax=job.lmax,
                lMin=cfg["lmin"],
                add_noise=cfg["add_noise"],
                sigma_noise_pix=sigma_noise_pix,
                psf_pix_fwhm=job.psf_pix_fwhm,
                N_CIB_PER_PIXEL=cfg["N_CIB_PER_PIXEL"],
                N_G_PER_PIXEL=cfg["N_G_PER_PIXEL"],
                alpha=cfg["alpha"],
                plot=cfg["plot"],
                grab_cib_sim=grab_cib_sim,
                datestr=cfg["datestr"],
                apply_mask=cfg["apply_mask"],
                pixel_fn_correct=cfg["pixel_fn_correct"],
                lensmode=cfg["lensmode"],
                mockstr=mockstr,
                use_beam_in_norm=cfg["exact_beam"],
                skew_filter_mode=cfg["skew_filter_mode"],
                s_max=cfg["s_max"],
                verbose=cfg["verbose"],
                save_intermediate_plots=cfg["save_intermediate_plots"],
                intermediate_plot_dir=intermediate_plot_dir,
                enable_lensing=cfg["enable_lensing"],
                kappa_amplitude=cfg["kappa_amplitude"],
                kappa_seed=cfg["kappa_seed"],
                mode=cfg["mode"],
                add_foreground=cfg.get("add_foreground", False),
                foreground_alpha=cfg.get("foreground_alpha", 2.0),
                foreground_seed=cfg.get("foreground_seed", 12345),
            )

        if cfg["plot"]:
            suptitle = gen_suptitle(
                grab_cib_sim,
                cfg["add_noise"],
                cfg["apply_mask"],
                exact_beam=False,
                psf_pix_fwhm=job.psf_pix_fwhm,
            )
            suptitle += "\n$" + str(int(cfg["lmin"])) + "<\\ell<" + str(int(job.lmax)) + "$"

            if "enable_lensing" in cfg.keys() and cfg["enable_lensing"]:
                print([k for k in res.keys()])
                fig_in_out = plot_input_recovered_kappa(
                    res,
                    bbox_to_anchor=None,
                    ncol=1,
                    ylim=None,
                    figsize=(6, 5),
                    markersize=5,
                    legend_fs=10,
                    xlim=[100, 1.0e5],
                    loc=3,
                    lMax=job.lmax,
                    lMin=cfg["lmin"],
                )
                fig_fpath = os.path.join(self.fig_dir, f"{resname}_{cfg['datestr']}_nbar{cfg['nbar']}_kappa_recover.png")
                fig_in_out.savefig(fig_fpath, dpi=200)

            fig = plot_recov_components(
                res,
                bbox_to_anchor=None,
                ncol=1,
                ylim=None,
                figsize=(9, 6.5),
                markersize=5,
                legend_fs=10,
                xlim=[100, 1.0e5],
                loc=3,
                suptitle=suptitle,
                lMax=job.lmax,
                lMin=cfg["lmin"],
                show=False,
            )
            fig_fpath = os.path.join(self.fig_dir, f"{resname}_{cfg['datestr']}_nbar{cfg['nbar']}.png")
            fig.savefig(fig_fpath, dpi=200)

        if cfg["save"]:
            np.savez(res_fpath, **res)

        return res_fpath

    def run_sweep(self) -> Dict[str, List[str]]:
        jobs = self.generate_test_configs()
        verbose = self.cfg["verbose"]
        saved_paths: List[str] = []
        failed_jobs: List[str] = []

        for idx, job in enumerate(jobs, start=1):
            psf_label = "None(CIBER beam)" if job.psf_pix_fwhm is None else str(job.psf_pix_fwhm)
            if verbose >= 1:
                print(f"[{idx}/{len(jobs)}] lMax={job.lmax}, psf_pix_fwhm={psf_label}")
            try:
                saved_paths.append(self.run_single_test(job))
            except Exception as exc:  # noqa: BLE001
                import traceback
                failed_jobs.append(f"lMax={job.lmax}, psf={psf_label}: {exc}")
                print("ERROR:", failed_jobs[-1])
                print("\nFull traceback:")
                traceback.print_exc()

        return {"saved": saved_paths, "failed": failed_jobs}


def main() -> int:
    _ensure_repo_on_path()
    parser = _make_parser()
    args = parser.parse_args()

    try:
        cfg = _resolve_config(args)
        _validate_config(cfg)
    except Exception as exc:  # noqa: BLE001
        print("Configuration error:", exc)
        return 2

    runner = TestSweepRunner(cfg)
    jobs = runner.generate_test_configs()

    if args.dry_run:
        print("Resolved config:")
        print(json.dumps(cfg, indent=2, default=str))
        print(f"Planned jobs: {len(jobs)}")
        for job in jobs:
            print(f"- lMax={job.lmax}, psf_pix_fwhm={job.psf_pix_fwhm}")
        print("Result directory:", runner.res_dir)
        print("Figure directory:", runner.fig_dir)
        return 0

    outcome = runner.run_sweep()
    print("\nSweep complete")
    print("Saved result files:", len(outcome["saved"]))
    for path in outcome["saved"]:
        print("-", path)

    if outcome["failed"]:
        print("Failed jobs:", len(outcome["failed"]))
        for failure in outcome["failed"]:
            print("-", failure)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

