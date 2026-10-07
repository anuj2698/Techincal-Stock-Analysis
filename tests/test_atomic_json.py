import tempfile
import unittest
from pathlib import Path

from atomic_json import read_json, update_json_atomic, write_json_atomic


class AtomicJsonTests(unittest.TestCase):
    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "value.json"
            write_json_atomic(target, {"count": 1})
            self.assertEqual(read_json(target, {}), {"count": 1})
            update_json_atomic(target, {}, lambda value: {"count": value["count"] + 1})
            self.assertEqual(read_json(target, {}), {"count": 2})


if __name__ == "__main__":
    unittest.main()
