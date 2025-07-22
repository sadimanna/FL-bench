import warnings
from pathlib import Path
from typing import Union
import wandb
from rich.console import Console


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str], wandb_project: str = None
    ):
        """This class is for solving the incompatibility between the progress
        bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
            wandb_project (str): The wandb project name (optional).
        """
        self.stdout = stdout
        self.logfile_output_stream = None
        self.enable_log = enable_log
        self.wandb_project = wandb_project
        self.wandb_run = None
        if self.enable_log:
            self.logfile_output_stream = open(logfile_path, "w")
            self.logfile_logger = Console(
                file=self.logfile_output_stream,
                record=True,
                log_path=False,
                log_time=False,
                soft_wrap=True,
                tab_size=4,
            )
        if wandb_project:
            self.wandb_run = wandb.init(project=wandb_project, reinit=True, mode='online', name="FL-Bench")

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logfile_logger.log(*args, **kwargs)
        if self.wandb_run:
            wandb.log({"message": " ".join(str(a) for a in args)})

    def warn(self, *args, **kwargs):
        """Wrapper of `warnings.warn` to print the warning message onto
        `stdout` and log file (if enabled)."""
        msg = ""
        with warnings.catch_warnings(record=True) as w:
            warnings.warn(*args, **kwargs)
            msg = str(w[-1].message)
        self.stdout.log(msg)
        if self.enable_log:
            self.logfile_logger.log(msg)
        if self.wandb_run:
            wandb.log({"warning": msg})

    def close(self):
        if self.logfile_output_stream:
            self.logfile_output_stream.close()
        if self.wandb_run:
            wandb.finish()
