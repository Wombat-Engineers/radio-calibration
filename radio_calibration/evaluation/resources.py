"""Resource use of a block of code: run time, GPU memory, GPU use, process memory, CPU use."""

import threading
import time

import pandas as pd
import psutil
import pynvml

SAMPLE_COLUMNS = ("seconds", "gpu_memory_mib", "gpu_utilization_percent", "rss_mib", "cpu_percent")
_MIB = 2**20


class ResourceMonitor:
    """Sample the resource use of the process while a `with` block runs.

    A background thread takes one sample at each interval. The monitor also takes one sample
    when the block starts and one sample when the block stops. The GPU values are those of
    the full device: they include the memory cache of Dr.Jit and the memory of other
    processes. If NVML is not available, the GPU values are NaN.

    Example:
        with ResourceMonitor() as monitor:
            model, history = RadioMapModel.pretrain(...)
        monitor.summary()["peak_gpu_memory_mib"]

    Args:
        interval_s: Time between two samples, in seconds.
        gpu_index: NVML index of the GPU. Assumption: Dr.Jit uses the first CUDA device.

    Raises:
        ValueError: If `interval_s` is not more than 0.
    """

    def __init__(self, interval_s: float = 1.0, gpu_index: int = 0) -> None:
        if not interval_s > 0:
            raise ValueError(f"interval_s must be more than 0, not {interval_s}.")
        self._interval_s = interval_s
        self._gpu_index = gpu_index
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu = None
        self._start = 0.0
        self._rows: list[tuple[float, float, float, float, float]] = []

    def __enter__(self) -> "ResourceMonitor":
        self._rows = []
        self._stop.clear()
        self._gpu = self._open_gpu()
        self._start = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._sample()
        if self._gpu is not None:
            pynvml.nvmlShutdown()
            self._gpu = None

    @property
    def samples(self) -> pd.DataFrame:
        """The samples, one row for each sample, with the columns of `SAMPLE_COLUMNS`.

        `seconds` is the time from the start of the block. `cpu_percent` is the CPU use of the
        process since the previous sample. It can be more than 100 on a CPU with more than one
        core. It is NaN in the first row.

        Raises:
            RuntimeError: If the block did not run, or if the block did not stop.
        """
        if self._thread is not None or not self._rows:
            raise RuntimeError("The samples are available only after the with block.")
        return pd.DataFrame(self._rows, columns=list(SAMPLE_COLUMNS))

    def summary(self) -> dict[str, float]:
        """Return the totals of the samples.

        Returns:
            The keys `seconds` (run time of the block), `peak_gpu_memory_mib`,
            `mean_gpu_utilization_percent`, `peak_rss_mib`, and `mean_cpu_percent`. A value
            is NaN if no sample has it.

        Raises:
            RuntimeError: If the block did not run, or if the block did not stop.
        """
        samples = self.samples
        return {
            "seconds": float(samples["seconds"].iloc[-1]),
            "peak_gpu_memory_mib": float(samples["gpu_memory_mib"].max()),
            "mean_gpu_utilization_percent": float(samples["gpu_utilization_percent"].mean()),
            "peak_rss_mib": float(samples["rss_mib"].max()),
            "mean_cpu_percent": float(samples["cpu_percent"].mean()),
        }

    def _open_gpu(self):
        """Return the NVML handle of the GPU, or None if NVML is not available."""
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return None
        try:
            return pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        except pynvml.NVMLError:
            pynvml.nvmlShutdown()
            return None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def _sample(self) -> None:
        memory = utilization = float("nan")
        if self._gpu is not None:
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used / _MIB
                utilization = float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu)
            except pynvml.NVMLError:
                pass
        cpu = self._process.cpu_percent(None)
        # The first call of cpu_percent(None) gives 0.0 with no meaning (psutil documentation,
        # Process.cpu_percent).
        if not self._rows:
            cpu = float("nan")
        self._rows.append(
            (
                time.perf_counter() - self._start,
                memory,
                utilization,
                self._process.memory_info().rss / _MIB,
                cpu,
            )
        )
