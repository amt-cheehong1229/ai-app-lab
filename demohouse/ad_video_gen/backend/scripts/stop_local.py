from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time

from start_local import SERVICES, STATE_FILE


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def remove_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def get_listening_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(
            f"Failed to inspect port {port}: {result.stderr.strip() or result.stdout.strip()}"
        )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def collect_status() -> list[dict]:
    return [
        {
            "name": service["name"],
            "port": service["port"],
            "pids": get_listening_pids(service["port"]),
        }
        for service in SERVICES
    ]


def print_status(status_rows: list[dict]) -> None:
    print("Local service status:")
    for row in status_rows:
        status = "OPEN" if row["pids"] else "CLOSED"
        pid_text = ",".join(str(pid) for pid in row["pids"]) if row["pids"] else "-"
        print(f"  {row['port']} {row['name']:<17} {status:<6} pid={pid_text}")


def collect_all_listener_pids() -> set[int]:
    pids: set[int] = set()
    for row in collect_status():
        pids.update(row["pids"])
    return pids


def send_signal_to_pids(pids: set[int], sig: int) -> None:
    for pid in sorted(pids):
        if process_exists(pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue


def wait_until_gone(pids: set[int], timeout_seconds: float) -> set[int]:
    deadline = time.time() + timeout_seconds
    remaining = {pid for pid in pids if process_exists(pid)}
    while remaining and time.time() < deadline:
        time.sleep(0.2)
        remaining = {pid for pid in remaining if process_exists(pid)}
    return remaining


def stop_services(timeout_seconds: float) -> int:
    status_before = collect_status()
    print_status(status_before)

    state = read_state()
    launcher_pid = state.get("launcher_pid")
    if isinstance(launcher_pid, int) and process_exists(launcher_pid):
        print(f"Sending SIGINT to launcher pid={launcher_pid}")
        send_signal_to_pids({launcher_pid}, signal.SIGINT)
        wait_until_gone({launcher_pid}, timeout_seconds)

    remaining_listeners = collect_all_listener_pids()
    if remaining_listeners:
        print(f"Sending SIGINT to listener pids={sorted(remaining_listeners)}")
        send_signal_to_pids(remaining_listeners, signal.SIGINT)
        wait_until_gone(remaining_listeners, timeout_seconds)

    remaining_listeners = collect_all_listener_pids()
    if remaining_listeners:
        print(f"Sending SIGTERM to listener pids={sorted(remaining_listeners)}")
        send_signal_to_pids(remaining_listeners, signal.SIGTERM)
        wait_until_gone(remaining_listeners, 3.0)

    remaining_listeners = collect_all_listener_pids()
    if remaining_listeners:
        print(f"Sending SIGKILL to listener pids={sorted(remaining_listeners)}")
        send_signal_to_pids(remaining_listeners, signal.SIGKILL)
        wait_until_gone(remaining_listeners, 1.0)

    status_after = collect_status()
    print_status(status_after)
    remove_state()

    still_open = [row for row in status_after if row["pids"]]
    if still_open:
        print("Some services are still running.")
        return 1

    print("Stopped local services.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop local ad_video_gen services.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only print the current status of ports 8000-8005.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait after sending SIGINT before escalating.",
    )
    args = parser.parse_args()

    if args.status:
        print_status(collect_status())
        return 0

    return stop_services(timeout_seconds=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
