from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

from dotenv import dotenv_values

from bootstrap_local import BACKEND_ROOT, main as bootstrap_main

STATE_DIR = BACKEND_ROOT / ".local_state"
STATE_FILE = STATE_DIR / "local_services.json"

SERVICES = [
    {
        "name": "short-link",
        "port": 8005,
        "cwd": BACKEND_ROOT / "app" / "short_link",
        "module": "app:app",
        "readiness_path": "/openapi.json",
    },
    {
        "name": "market-agent",
        "port": 8000,
        "cwd": BACKEND_ROOT / "app" / "market-agent" / "src",
        "module": "app:app",
        "readiness_path": "/.well-known/agent-card.json",
    },
    {
        "name": "director-agent",
        "port": 8001,
        "cwd": BACKEND_ROOT / "app" / "director-agent" / "src",
        "module": "app:app",
        "readiness_path": "/.well-known/agent-card.json",
    },
    {
        "name": "evaluate-agent",
        "port": 8002,
        "cwd": BACKEND_ROOT / "app" / "evaluate-agent" / "src",
        "module": "app:app",
        "readiness_path": "/.well-known/agent-card.json",
    },
    {
        "name": "release-agent",
        "port": 8003,
        "cwd": BACKEND_ROOT / "app" / "release-agent" / "src",
        "module": "app:app",
        "readiness_path": "/.well-known/agent-card.json",
    },
    {
        "name": "multimedia-agent",
        "port": 8004,
        "cwd": BACKEND_ROOT / "app" / "multimedia-agent" / "src",
        "module": "server:app",
        "readiness_path": "/list-apps",
    },
]


def write_runtime_state(processes: list[subprocess.Popen]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "launcher_pid": os.getpid(),
        "services": [
            {
                "name": service["name"],
                "port": service["port"],
                "pid": process.pid,
            }
            for process, service in zip(processes, SERVICES)
        ],
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_runtime_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    dotenv_path = BACKEND_ROOT / ".env"
    if dotenv_path.exists():
        env.update(
            {
                key: str(value)
                for key, value in dotenv_values(dotenv_path).items()
                if value is not None
            }
        )
    env.setdefault("AD_VIDEO_GEN_BACKEND_ROOT", str(BACKEND_ROOT))
    env.setdefault("LOCAL_MEDIA_DIR", str(BACKEND_ROOT / ".local_media"))
    env.setdefault("LOCAL_MEDIA_BASE_URL", "http://127.0.0.1:8005")
    env.setdefault("SHORTEN_URL_SERVICE_URL", "http://127.0.0.1:8005")
    env.setdefault("SHORT_LINK_DOMAIN", "http://127.0.0.1:8005")
    env.setdefault("REMOTE_AGENT_MARKET_AGENT_URL", "http://127.0.0.1:8000")
    env.setdefault("REMOTE_AGENT_DIRECTOR_AGENT_URL", "http://127.0.0.1:8001")
    env.setdefault("REMOTE_AGENT_EVALUATE_AGENT_URL", "http://127.0.0.1:8002")
    env.setdefault("REMOTE_AGENT_RELEASE_AGENT_URL", "http://127.0.0.1:8003")
    env.setdefault("MODEL_AGENT_PROVIDER", "openai")
    env.setdefault("MODEL_AGENT_NAME", env.get("OPENAI_MODEL_TEXT", "gpt-4.1-mini"))
    env.setdefault("MODEL_AGENT_API_BASE", env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    env.setdefault("MODEL_AGENT_API_KEY", env.get("OPENAI_API_KEY", ""))
    env.setdefault("MODEL_FORMAT_NAME", env.get("OPENAI_MODEL_FORMAT", env.get("OPENAI_MODEL_EVAL", env.get("OPENAI_MODEL_TEXT", "gpt-4.1-mini"))))
    env.setdefault("MODEL_IMAGE_NAME", env.get("OPENAI_IMAGE_MODEL", "gpt-image-1"))
    env.setdefault("MODEL_IMAGE_API_BASE", env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    env.setdefault("MODEL_IMAGE_API_KEY", env.get("OPENAI_API_KEY", ""))
    env.setdefault("MODEL_VIDEO_NAME", env.get("OPENAI_VIDEO_MODEL", "sora-2"))
    env.setdefault("MODEL_VIDEO_API_BASE", env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    env.setdefault("MODEL_VIDEO_API_KEY", env.get("OPENAI_API_KEY", ""))
    env.setdefault("MODEL_EVALUATE_ITEM", env.get("OPENAI_MODEL_EVAL", env.get("OPENAI_MODEL_VISION", "gpt-4.1-mini")))
    common_path = str(BACKEND_ROOT / "common")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        common_path if not existing_pythonpath else f"{common_path}:{existing_pythonpath}"
    )
    return env


def wait_for_http_ready(
    *,
    service_name: str,
    port: int,
    path: str,
    process: subprocess.Popen,
    timeout_seconds: float = 60.0,
) -> None:
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}{path}"
    last_error = "service did not respond"

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{service_name} exited before readiness check completed "
                f"(code {process.returncode})"
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                status = getattr(response, "status", 200)
                if 200 <= status < 500:
                    print(f"{service_name} is ready at {url}")
                    return
                last_error = f"unexpected status {status}"
        except urllib.error.HTTPError as exc:
            # Some routes may answer with auth/validation errors and still prove the app is live.
            if 200 <= exc.code < 500:
                print(f"{service_name} is ready at {url} (status {exc.code})")
                return
            last_error = f"http {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.5)

    raise RuntimeError(f"{service_name} did not become ready at {url}: {last_error}")


def main() -> None:
    bootstrap_main()
    env = build_env()
    python_bin = BACKEND_ROOT / ".venv" / "bin" / "python"
    if not python_bin.exists():
        raise SystemExit("Missing backend/.venv. Run `UV_CACHE_DIR=.uv-cache uv sync` first.")

    processes: list[subprocess.Popen] = []
    try:
        for service in SERVICES:
            cmd = [
                str(python_bin),
                "-m",
                "uvicorn",
                service["module"],
                "--host",
                "127.0.0.1",
                "--port",
                str(service["port"]),
                "--loop",
                "asyncio",
            ]
            process = subprocess.Popen(cmd, cwd=service["cwd"], env=env)
            processes.append(process)
            write_runtime_state(processes)
            print(f"Started {service['name']} on http://127.0.0.1:{service['port']}")
            wait_for_http_ready(
                service_name=service["name"],
                port=service["port"],
                path=service["readiness_path"],
                process=process,
            )

        print("All local services are running. Press Ctrl+C to stop them.")
        while True:
            for process, service in zip(processes, SERVICES):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{service['name']} exited early with code {process.returncode}"
                    )
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        deadline = time.time() + 10
        for process in reversed(processes):
            if process.poll() is None:
                timeout = max(deadline - time.time(), 0.1)
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
        remove_runtime_state()
        print("Stopped local services.")


if __name__ == "__main__":
    main()
