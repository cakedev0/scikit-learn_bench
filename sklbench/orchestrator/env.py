import json
import os
import subprocess
from typing import Dict

import pandas as pd

from ..utils.logger import logger


def get_threadpool_info():
    try:
        from threadpoolctl import threadpool_info
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to get threadpool info with "threadpoolctl" module')
        return []

    threadpools = threadpool_info()
    for threadpool in threadpools:
        threadpool.pop("filepath", None)
    return threadpools


def get_software_info() -> Dict:
    result = {}
    result["threadpool_info"] = get_threadpool_info()

    pixi_project_root = os.environ.get("PIXI_PROJECT_ROOT")
    pixi_environment_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    result["pixi_environment_name"] = pixi_environment_name
    pixi_list = subprocess.check_output(
        [
            "pixi",
            "list",
            "--manifest-path",
            pixi_project_root,
            "--environment",
            pixi_environment_name,
            "--json",
        ],
        shell=False,
        text=True,
    )
    pixi_packages = json.loads(pixi_list)
    result["pixi_packages"] = {pkg.pop("name"): pkg for pkg in pixi_packages}
    return result


def get_oneapi_devices() -> pd.DataFrame:
    try:
        import dpctl

        devices = dpctl.get_devices()
        devices = {
            device.filter_string: {
                "name": device.name,
                "vendor": device.vendor,
                "type": str(device.device_type).split(".")[1],
                "driver version": device.driver_version,
                "memory size[GB]": round(device.global_mem_size / 2**30),
            }
            for device in devices
        }
        if len(devices) > 0:
            return pd.DataFrame(devices).T
        logger.warning("dpctl device table is empty")
    except (ImportError, ModuleNotFoundError):
        logger.warning("dpctl can not be imported")
    return pd.DataFrame({"type": []})


def decode_nvml_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def get_nvml_value(getter, *args):
    try:
        return decode_nvml_value(getter(*args))
    except Exception:
        return None


def format_cuda_driver_version(version):
    if version is None:
        return None
    return f"{version // 1000}.{version % 1000 // 10}"


def get_nvidia_devices() -> pd.DataFrame:
    try:
        import pynvml
    except (ImportError, ModuleNotFoundError):
        logger.warning("pynvml can not be imported")
        return pd.DataFrame({"type": []})

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as exc:
        logger.warning(f"Unable to get NVIDIA devices with NVML: {exc}")
        return pd.DataFrame({"type": []})

    driver_version = get_nvml_value(pynvml.nvmlSystemGetDriverVersion)
    cuda_driver_version = None
    if hasattr(pynvml, "nvmlSystemGetCudaDriverVersion_v2"):
        cuda_driver_version = get_nvml_value(
            pynvml.nvmlSystemGetCudaDriverVersion_v2
        )
    elif hasattr(pynvml, "nvmlSystemGetCudaDriverVersion"):
        cuda_driver_version = get_nvml_value(pynvml.nvmlSystemGetCudaDriverVersion)
    cuda_driver_version = format_cuda_driver_version(cuda_driver_version)

    devices = {}
    for device_index in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except pynvml.NVMLError as exc:
            logger.warning(
                f"Unable to get NVIDIA device {device_index} handle with NVML: {exc}"
            )
            continue

        device_info = {
            "name": get_nvml_value(pynvml.nvmlDeviceGetName, handle),
            "vendor": "NVIDIA Corporation",
            "type": "gpu",
            "driver version": driver_version,
            "cuda driver version": cuda_driver_version,
            "index": device_index,
        }

        memory_info = get_nvml_value(pynvml.nvmlDeviceGetMemoryInfo, handle)
        if memory_info is not None:
            device_info["memory size[GB]"] = round(memory_info.total / 2**30)

        if hasattr(pynvml, "nvmlDeviceGetCudaComputeCapability"):
            compute_capability = get_nvml_value(
                pynvml.nvmlDeviceGetCudaComputeCapability, handle
            )
            if compute_capability is not None:
                device_info["cuda compute capability"] = ".".join(
                    map(str, compute_capability)
                )

        devices[f"cuda:{device_index}"] = device_info

    if len(devices) > 0:
        return pd.DataFrame(devices).T
    logger.warning("NVML device table is empty")
    return pd.DataFrame({"type": []})


def get_higher_isa(cpu_flags: str) -> str:
    ordered_sets = ["avx512", "avx2", "avx", "sse4_2", "ssse3", "sse2"]
    for isa in ordered_sets:
        if isa in cpu_flags:
            return isa
    return "unknown"


def get_hardware_info() -> Dict:
    result = {}
    oneapi_devices = get_oneapi_devices()
    if len(oneapi_devices) > 0:
        logger.info(f"DPCTL listed devices:\n{oneapi_devices}\n")
    nvidia_devices = get_nvidia_devices()
    if len(nvidia_devices) > 0:
        logger.info(f"NVML listed NVIDIA devices:\n{nvidia_devices}\n")

    try:
        from cpuinfo import get_cpu_info
        import joblib

        cpu_info = get_cpu_info()
        fields_map = {
            "arch": "architecture",
            "brand_raw": "name",
            "flags": "flags",
            "count": "logical_cpus",
        }
        for key in list(cpu_info.keys()):
            value = cpu_info.pop(key)
            if key in fields_map:
                cpu_info[fields_map[key]] = value
        cpu_info["flags"] = " ".join(cpu_info["flags"])
        cpu_info["physical_cores"] = joblib.cpu_count(only_physical_cores=True)
        result["CPU"] = cpu_info
        logger.info(f'CPU name: {cpu_info["name"]}')
        logger.info(
            "Highest supported ISA: " f'{get_higher_isa(cpu_info["flags"]).upper()}'
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse CPU info with "cpuinfo" module')

    result["GPU(s)"] = {}
    try:
        oneapi_gpus = oneapi_devices[oneapi_devices["type"] == "gpu"]
        result["GPU(s)"].update(oneapi_gpus.T.to_dict())
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to get devices with "dpctl" module')
    try:
        nvidia_gpus = nvidia_devices[nvidia_devices["type"] == "gpu"]
        result["GPU(s)"].update(nvidia_gpus.T.to_dict())
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to get NVIDIA devices with "pynvml" module')

    try:
        import psutil

        result["RAM size[GB]"] = round(psutil.virtual_memory().total / 2**30)
        logger.info(f'RAM size[GB]: {result["RAM size[GB]"]}')
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse memory info with "psutil" module')
    return result


def get_environment_info() -> Dict:
    return {"hardware": get_hardware_info(), "software": get_software_info()}
