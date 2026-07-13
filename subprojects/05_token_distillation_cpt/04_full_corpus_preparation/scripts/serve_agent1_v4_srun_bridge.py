#!/usr/bin/env python3
"""Expose a CSCS compute-node localhost site through a login-node localhost port.

CSCS compute nodes do not accept direct user SSH connections.  This deliberately
small bridge binds *only* to the login host's loopback address; each browser
connection is transported with ``srun --overlap`` into an already allocated
site-serving job, then to that job's own loopback HTTP server.  It neither
writes nor copies review documents.
"""

from __future__ import annotations

import argparse
import signal
import socket
import socketserver
import subprocess
import threading
from typing import Sequence


class LoopbackSrunBridge(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, job_id: str, target_port: int, srun_bin: str) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("bridge must bind only to 127.0.0.1")
        super().__init__(address, BridgeHandler)
        self.job_id = job_id
        self.target_port = target_port
        self.srun_bin = srun_bin


class BridgeHandler(socketserver.BaseRequestHandler):
    server: LoopbackSrunBridge

    def handle(self) -> None:
        command = [
            self.server.srun_bin,
            f"--jobid={self.server.job_id}",
            "--overlap",
            "--ntasks=1",
            "--cpus-per-task=1",
            "nc",
            "127.0.0.1",
            str(self.server.target_port),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        def receive_client() -> None:
            assert process.stdin is not None
            try:
                while True:
                    chunk = self.request.recv(65536)
                    if not chunk:
                        break
                    process.stdin.write(chunk)
                    process.stdin.flush()
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass

        receiver = threading.Thread(target=receive_client, daemon=True)
        receiver.start()
        assert process.stdout is not None
        try:
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                self.request.sendall(chunk)
        except ConnectionError:
            pass
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            receiver.join(timeout=1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, help="running Slurm job that serves the private site")
    parser.add_argument("--port", type=int, required=True, help="localhost port exposed on the Clariden login host")
    parser.add_argument("--target-port", type=int, default=18765, help="site's compute-node loopback port")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--srun-bin", default="srun")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.target_port <= 65535:
        parser.error("ports must be in 1..65535")
    server = LoopbackSrunBridge((args.bind, args.port), job_id=args.job_id, target_port=args.target_port, srun_bin=args.srun_bin)
    stop = lambda *_: threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
