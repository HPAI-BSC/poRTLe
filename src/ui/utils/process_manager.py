"""
Background Process Manager for poRTLe

Provides persistent command execution that survives UI restarts.
Processes run in background with output logged to files.
"""

import os
import sqlite3
import subprocess
import threading
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json


class ProcessManager:
    """
    Manages background processes with persistent tracking.

    Features:
    - Processes run independently of UI
    - SQLite registry for process tracking
    - Output logged to files
    - Process lifecycle management (start, stop, query)
    """

    def __init__(self, db_path: str = None, log_dir: str = None):
        """
        Initialize the process manager.

        Args:
            db_path: Path to SQLite database for process registry
            log_dir: Directory for process output logs
        """
        # Default paths relative to repo root
        repo_root = Path(__file__).parent.parent.parent.parent

        if db_path is None:
            db_path = repo_root / "results" / "process_registry.db"

        if log_dir is None:
            log_dir = repo_root / "results" / "logs" / "commands"

        self.db_path = Path(db_path)
        self.log_dir = Path(log_dir)

        # Ensure directories and database exist on startup
        self._ensure_storage_ready()

    def _ensure_storage_ready(self):
        """Ensure directories exist and the process table is initialized."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the process registry database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT UNIQUE NOT NULL,
                pid INTEGER,
                command TEXT NOT NULL,
                cwd TEXT,
                description TEXT,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                exit_code INTEGER,
                log_file TEXT,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    def start_process(
        self,
        command: List[str],
        cwd: Path = None,
        description: str = "Background command",
        metadata: Dict = None
    ) -> str:
        """
        Start a background process.

        Args:
            command: Command list (e.g., ['python', 'script.py'])
            cwd: Working directory
            description: Human-readable description
            metadata: Additional metadata (stored as JSON)

        Returns:
            process_id: Unique identifier for this process
        """
        self._ensure_storage_ready()

        # Generate unique process ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        process_id = f"proc_{timestamp}"

        # Create log file path
        log_file = self.log_dir / f"{process_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Register process in database (status=starting)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO processes (
                process_id, command, cwd, description, status,
                start_time, log_file, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            process_id,
            json.dumps(command),
            str(cwd) if cwd else None,
            description,
            "starting",
            datetime.now().isoformat(),
            str(log_file),
            json.dumps(metadata) if metadata else None
        ))

        conn.commit()
        conn.close()

        # Start process in background thread
        thread = threading.Thread(
            target=self._run_process,
            args=(process_id, command, cwd, log_file),
            daemon=True
        )
        thread.start()

        return process_id

    def _run_process(
        self,
        process_id: str,
        command: List[str],
        cwd: Path,
        log_file: Path
    ):
        """
        Run a process in the background and log output.

        This runs in a separate thread.

        IMPORTANT: Output is redirected directly to the log file, not piped through Python.
        This ensures the process survives even if the parent process (Streamlit) dies.
        """
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # Write header to log file
            with open(log_file, 'w') as log:
                log.write(f"Command: {' '.join(command)}\n")
                log.write(f"Working directory: {cwd}\n")
                log.write(f"Start time: {datetime.now().isoformat()}\n")
                log.write("-" * 80 + "\n\n")
                log.flush()

            # Open log file for subprocess output (append mode)
            # The subprocess gets its own file descriptor, so it can continue writing
            # even if this parent process dies
            with open(log_file, 'a') as log:
                # Start process with output redirected DIRECTLY to file
                # No PIPE means no broken pipe issues when parent dies
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd) if cwd else None,
                    stdout=log,  # Direct file redirection (not PIPE!)
                    stderr=subprocess.STDOUT,
                    start_new_session=True  # Start in new process group
                )

                # Update database with PID and running status
                self._update_process(
                    process_id,
                    status="running",
                    pid=process.pid
                )

                # Wait for completion
                # No need to read output - it goes directly to the file
                # Process will continue running even if this thread dies
                process.wait()

            # Log completion (append to file)
            with open(log_file, 'a') as log:
                log.write("\n" + "-" * 80 + "\n")
                log.write(f"End time: {datetime.now().isoformat()}\n")
                log.write(f"Exit code: {process.returncode}\n")

            # Update database with completion status
            self._update_process(
                process_id,
                status="completed" if process.returncode == 0 else "failed",
                end_time=datetime.now().isoformat(),
                exit_code=process.returncode
            )

        except Exception as e:
            # Log error
            with open(log_file, 'a') as log:
                log.write(f"\n\nERROR: {str(e)}\n")

            # Update database with error status
            self._update_process(
                process_id,
                status="error",
                end_time=datetime.now().isoformat(),
                exit_code=-1
            )

    def _update_process(
        self,
        process_id: str,
        status: str = None,
        pid: int = None,
        end_time: str = None,
        exit_code: int = None
    ):
        """Update process information in database."""
        self._ensure_storage_ready()
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if pid is not None:
            updates.append("pid = ?")
            params.append(pid)

        if end_time is not None:
            updates.append("end_time = ?")
            params.append(end_time)

        if exit_code is not None:
            updates.append("exit_code = ?")
            params.append(exit_code)

        if updates:
            params.append(process_id)
            cursor.execute(
                f"UPDATE processes SET {', '.join(updates)} WHERE process_id = ?",
                params
            )

        conn.commit()
        conn.close()

    def get_process(self, process_id: str) -> Optional[Dict]:
        """
        Get information about a process.

        Returns:
            Dictionary with process information, or None if not found
        """
        self._ensure_storage_ready()
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM processes WHERE process_id = ?",
            (process_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def list_processes(
        self,
        status: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        List processes.

        Args:
            status: Filter by status (running, completed, failed, error)
            limit: Maximum number of results

        Returns:
            List of process dictionaries
        """
        self._ensure_storage_ready()
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status:
            cursor.execute(
                """
                SELECT * FROM processes
                WHERE status = ?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (status, limit)
            )
        else:
            cursor.execute(
                """
                SELECT * FROM processes
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (limit,)
            )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_log(self, process_id: str, tail: int = None) -> Tuple[bool, str]:
        """
        Get log output for a process.

        Args:
            process_id: Process identifier
            tail: If specified, return only last N lines

        Returns:
            Tuple of (success, log_content)
        """
        process = self.get_process(process_id)
        if not process:
            return False, f"Process {process_id} not found"

        log_file = Path(process['log_file'])
        if not log_file.exists():
            return False, f"Log file not found: {log_file}"

        try:
            with open(log_file, 'r') as f:
                if tail:
                    lines = f.readlines()
                    content = ''.join(lines[-tail:])
                else:
                    content = f.read()

            return True, content

        except Exception as e:
            return False, f"Error reading log: {str(e)}"

    def kill_process(self, process_id: str) -> Tuple[bool, str]:
        """
        Kill a running process.

        Returns:
            Tuple of (success, message)
        """
        process = self.get_process(process_id)
        if not process:
            return False, f"Process {process_id} not found"

        if process['status'] not in ['starting', 'running']:
            return False, f"Process is not running (status: {process['status']})"

        pid = process['pid']
        if not pid:
            return False, "No PID available"

        try:
            # Try to kill the process
            os.kill(pid, signal.SIGTERM)

            # Wait a bit and check if it's still alive
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if process exists
                # Still alive, force kill
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                # Process is dead
                pass

            # Update database
            self._update_process(
                process_id,
                status="killed",
                end_time=datetime.now().isoformat(),
                exit_code=-9
            )

            return True, f"Process {process_id} killed"

        except ProcessLookupError:
            # Process already dead
            self._update_process(
                process_id,
                status="completed",
                end_time=datetime.now().isoformat()
            )
            return True, f"Process {process_id} was already terminated"

        except Exception as e:
            return False, f"Error killing process: {str(e)}"

    def cleanup_orphaned_processes(self) -> int:
        """
        Clean up processes that are marked as running but are actually dead.

        Returns:
            Number of processes cleaned up
        """
        processes = self.list_processes(status="running", limit=1000)
        cleaned = 0

        for process in processes:
            pid = process['pid']
            if not pid:
                continue

            try:
                # Check if process is still alive
                os.kill(pid, 0)
            except ProcessLookupError:
                # Process is dead, update database
                self._update_process(
                    process['process_id'],
                    status="orphaned",
                    end_time=datetime.now().isoformat()
                )
                cleaned += 1

        return cleaned


# Global instance
_manager = None


def get_manager() -> ProcessManager:
    """Get the global process manager instance."""
    global _manager
    if _manager is None:
        _manager = ProcessManager()
    return _manager
