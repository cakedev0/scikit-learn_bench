# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import json
import os
import subprocess
import sys
from typing import Dict

import pandas as pd

from .common import read_output_from_command
from .logger import logger


def get_numa_cpus_conf() -> Dict[int, str]:
    try:
        _, lscpu_text, _ = read_output_from_command("lscpu")
        return {
            i: numa_cpus
            for i, numa_cpus in enumerate(
                map(
                    lambda x: x.split(" ")[-1],
                    filter(
                        lambda line: "NUMA" in line and "CPU(s)" in line,
                        lscpu_text.split("\n"),
                    ),
                )
            )
        }
    except FileNotFoundError:
        logger.warning("Unable to get numa cpus configuration via lscpu")
        return dict()


def get_number_of_sockets():
    if sys.platform == "win32":
        try:
            command = ["wmic", "cpu", "get", "DeviceID"]
            result = subprocess.check_output(command, shell=False, text=True)
            n_sockets = len(
                list(filter(lambda x: x.startswith("CPU"), result.split("\n")))
            )
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
            logger.warning("Unable to get number of sockets via wmic")
            n_sockets = 1
    elif sys.platform == "linux":
        try:
            _, lscpu_text, _ = read_output_from_command("lscpu")
            for line in lscpu_text.split("\n"):
                if "Socket(s):" in line:
                    n_sockets = int(line.split(":")[1].strip())
                    break
            else:
                logger.warning("Unable to find Socket(s) information in lscpu output")
                n_sockets = 1
        except (FileNotFoundError, ValueError, IndexError):
            logger.warning("Unable to get number of sockets via lscpu")
            n_sockets = 1
    else:
        logger.warning("Unable to get number of sockets due to unknown sys.platform")
        n_sockets = 1
    return n_sockets


def get_software_info() -> Dict:
    result = dict()
    # pixi list
    pixi_project_root = os.environ.get("PIXI_PROJECT_ROOT")
    pixi_environment_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if pixi_project_root and pixi_environment_name:
        result["pixi_project_root"] = pixi_project_root
        result["pixi_environment_name"] = pixi_environment_name
        try:
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
        except (
            FileNotFoundError,
            PermissionError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ):
            logger.warning("Unable to get python packages list via pixi")

    # conda list
    try:
        _, conda_list, _ = read_output_from_command("conda list --json")
        conda_packages = json.loads(conda_list)
        result["conda_packages"] = {pkg.pop("name"): pkg for pkg in conda_packages}
    # pip list
    except (FileNotFoundError, PermissionError, AttributeError):
        logger.warning("Unable to get python packages list via conda")
        try:
            _, pip_list, _ = read_output_from_command("pip list --format json")
            pip_packages = json.loads(pip_list)
            result["pip_packages"] = {pkg.pop("name"): pkg for pkg in pip_packages}
        except (FileNotFoundError, PermissionError, AttributeError):
            logger.warning("Unable to get python packages list via pip")
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
        else:
            logger.warning("dpctl device table is empty")
    except (ImportError, ModuleNotFoundError):
        logger.warning("dpctl can not be imported")
    # 'type' is left for device type selection only
    return pd.DataFrame({"type": list()})


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
        return pd.DataFrame({"type": list()})

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as exc:
        logger.warning(f"Unable to get NVIDIA devices with NVML: {exc}")
        return pd.DataFrame({"type": list()})

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

        uuid = get_nvml_value(pynvml.nvmlDeviceGetUUID, handle)
        if uuid is not None:
            device_info["uuid"] = uuid

        pci_info = get_nvml_value(pynvml.nvmlDeviceGetPciInfo, handle)
        if pci_info is not None:
            bus_id = decode_nvml_value(getattr(pci_info, "busId", None))
            if bus_id is not None:
                device_info["pci bus id"] = bus_id

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
    else:
        logger.warning("NVML device table is empty")
    return pd.DataFrame({"type": list()})


def get_higher_isa(cpu_flags: str) -> str:
    # TODO: add non-x86 sets
    ordered_sets = ["avx512", "avx2", "avx", "sse4_2", "ssse3", "sse2"]
    for isa in ordered_sets:
        if isa in cpu_flags:
            return isa
    return "unknown"


def get_hardware_info() -> Dict:
    result = dict()
    oneapi_devices = get_oneapi_devices()
    if len(oneapi_devices) > 0:
        logger.info(f"DPCTL listed devices:\n{oneapi_devices}\n")
    nvidia_devices = get_nvidia_devices()
    if len(nvidia_devices) > 0:
        logger.info(f"NVML listed NVIDIA devices:\n{nvidia_devices}\n")
    # CPU
    try:
        from cpuinfo import get_cpu_info

        cpu_info = get_cpu_info()
        # remap cpu info values to better understandable names
        fields_map = {
            "arch": "architecture",
            "brand_raw": "name",
            "flags": "flags",
            "count": "logical_cpus",
        }
        for key in list(cpu_info.keys()):
            value = cpu_info.pop(key)
            if key in fields_map.keys():
                cpu_info[fields_map[key]] = value
        # squash CPU flags
        cpu_info["flags"] = " ".join(cpu_info["flags"])
        result["CPU"] = cpu_info
        logger.info(f'CPU name: {cpu_info["name"]}')
        logger.info(
            "Highest supported ISA: " f'{get_higher_isa(cpu_info["flags"]).upper()}'
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse CPU info with "cpuinfo" module')
    # GPUs
    result["GPU(s)"] = dict()
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
    # RAM size
    try:
        import psutil

        result["RAM size[GB]"] = round(psutil.virtual_memory().total / 2**30)
        logger.info(f'RAM size[GB]: {result["RAM size[GB]"]}')
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse memory info with "psutil" module')
    return result


def get_environment_info() -> Dict:
    return {"hardware": get_hardware_info(), "software": get_software_info()}
