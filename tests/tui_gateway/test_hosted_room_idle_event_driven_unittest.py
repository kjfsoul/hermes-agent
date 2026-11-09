"""Idle hosted-room supervisors should be event-driven, without weakening room recovery."""
import threading
import time
import unittest
from contextlib import nullcontext

from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime


class IdleSupervisors(unittest.TestCase):
    def runtime(self, provider, poll=0.015):
        return HostedRoomRuntime(
            db_path="/unused/only-the-fake-provider-is-read.db",
            rooms=provider,
            turn_lock=lambda _: nullcontext(),
            rpc=object(),
            poll_interval_seconds=poll,
        )

    def test_no_rooms_does_not_poll_forever(self):
        first = threading.Event()
        calls = []
        def provider():
            calls.append(time.monotonic())
            first.set()
            return ()
        runtime = self.runtime(provider)
        try:
            runtime.start()
            self.assertTrue(first.wait(1))
            time.sleep(.09)
            self.assertEqual(len(calls), 1, "empty profiles should not perform periodic SQLite reads")
            self.assertTrue(runtime.status()["running"], "available for future groups")
        finally:
            self.assertTrue(runtime.stop(timeout=2))

    def test_group_created_after_idle_wakes_worker_without_restart(self):
        first = threading.Event()
        processed = threading.Event()
        bindings = []
        def provider():
            first.set()
            return tuple(bindings)
        runtime = self.runtime(provider)
        runtime._run_room_once = lambda binding: processed.set()
        try:
            runtime.start()
            self.assertTrue(first.wait(1))
            bindings.append(HostedRoomBinding("new-room", "gateway-id", 1))
            runtime.wakeup()
            self.assertTrue(processed.wait(1), "room creation must wake idle worker")
        finally:
            self.assertTrue(runtime.stop(timeout=2))

    def test_existing_rooms_still_poll_for_recovery(self):
        second = threading.Event()
        calls = []
        room = HostedRoomBinding("existing-room", "gateway-id", 1)
        def provider():
            calls.append(1)
            if len(calls) >= 3:
                second.set()
            return (room,)
        runtime = self.runtime(provider)
        runtime._run_room_once = lambda binding: None
        try:
            runtime.start()
            self.assertTrue(second.wait(1), "active rooms must retain recovery polling")
        finally:
            self.assertTrue(runtime.stop(timeout=2))


if __name__ == "__main__":
    unittest.main()
