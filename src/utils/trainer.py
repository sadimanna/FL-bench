from collections import OrderedDict, deque
from typing import Any, Callable, Optional
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.constants import MODE

 

class FLbenchTrainer:
    def __init__(
        self, server, client_cls, mode: str, num_workers: int, init_args: dict
    ):
        self.server = server
        self.client_cls = client_cls
        self.mode = mode
        self.num_workers = num_workers
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(
            f"cuda:{self.rank % torch.cuda.device_count()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        if self.mode == MODE.PARALLEL and self.world_size > 1:
            if not dist.is_initialized():
                self.setup()
            self.worker = client_cls(**init_args)
            self.worker.model = DDP(
                self.worker.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
        else:
            self.worker = client_cls(**init_args)
        self.train = self._ddp_train
        self.test = self._ddp_test
        self.exec = self._ddp_exec

    def setup(self): 
        # Initialize the process group 
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank

    def cleanup(self): 
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def _ddp_train(self):
        client_packages = OrderedDict()
        for client_id in self.server.selected_clients:
            server_package = self.server.package(client_id)
            client_package = self.worker.train(server_package)
            client_packages[client_id] = client_package

            if self.server.verbose:
                self.server.logger.log(
                    *client_package["eval_results"]["message"], sep="\n"
                )
            self.server.client_metrics[client_id][self.server.current_epoch] = (
                client_package["eval_results"]
            )
            self.server.clients_personal_model_params[client_id].update(
                client_package["personal_model_params"]
            )
            self.server.client_optimizer_states[client_id].update(
                client_package["optimizer_state"]
            )
            self.server.client_lr_scheduler_states[client_id].update(
                client_package["lr_scheduler_state"]
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return client_packages

    def _ddp_test(self, clients, results):
        for client_id in clients:
            server_package = self.server.package(client_id)
            metrics = self.worker.test(server_package)
            for stage in ["before", "after"]:
                for split in ["train", "val", "test"]:
                    results[stage][split].update(metrics[stage][split])
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _ddp_exec(
        self,
        func_name, #: str,
        clients, #: list[int],
        package_func, #: Optional[Callable[[int], dict[str, Any]]] = None,
    ):
        if package_func is None:
            package_func = getattr(self.server, "package")
        client_packages = OrderedDict()
        for client_id in clients:
            server_package = package_func(client_id)
            package = getattr(self.worker, func_name)(server_package)
            client_packages[client_id] = package
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return client_packages
