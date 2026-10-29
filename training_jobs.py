"""Process-local training jobs survive Streamlit script reruns/reconnections."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
import time


@dataclass
class TrainingJob:
    started: float = field(default_factory=time.monotonic)
    stage: str = "Queued for training"
    future: object = None

    def report(self, stage):
        self.stage = stage


class TrainingJobs:
    def __init__(self):
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast")
        self._jobs = {}

    def get_or_start(self, key, train):
        with self._lock:
            existing = self._jobs.get(key)
            if existing is not None:
                if not existing.future.done() or existing.future.exception() is None:
                    return existing
                del self._jobs[key]  # failed jobs can be retried
            # Bound retained models; never discard an in-flight job.
            for old_key, old in list(self._jobs.items()):
                if old.future.done():
                    del self._jobs[old_key]
            job = TrainingJob()
            job.future = self._executor.submit(train, job.report)
            self._jobs[key] = job
            return job


jobs = TrainingJobs()
