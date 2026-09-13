import unittest
from types import SimpleNamespace
from unittest.mock import patch

from expman.gpu_memory import sample


class GpuMemoryTests(unittest.TestCase):
    def test_pytorch_allocator_usage_is_reported_in_kilobytes(self):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: 1024,
            memory_reserved=lambda: 2048,
            max_memory_allocated=lambda: 3072,
            max_memory_reserved=lambda: 4096,
        )
        with patch(
            "expman.gpu_memory.import_module", return_value=SimpleNamespace(cuda=cuda)
        ):
            self.assertEqual(
                sample(),
                {
                    "gpu_allocated_kb": 1.0,
                    "gpu_reserved_kb": 2.0,
                    "gpu_peak_allocated_kb": 3.0,
                    "gpu_peak_reserved_kb": 4.0,
                },
            )

    def test_missing_pytorch_leaves_gpu_telemetry_unavailable(self):
        with patch("expman.gpu_memory.import_module", side_effect=ModuleNotFoundError):
            self.assertIsNone(sample())


if __name__ == "__main__":
    unittest.main()
