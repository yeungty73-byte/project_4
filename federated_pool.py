import os
import glob
import copy
import torch
import numpy as np
from loguru import logger
from typing import List, Optional, Dict


class FederatedPool:
# REF: Mnih, V. et al. (2016). Asynchronous methods for deep reinforcement learning. ICML, 1928-1937.
    """Checkpoint pool for federated-style PPO learning.

    Supports FedAvg parameter averaging, best-model selection,
    checkpoint history tracking, and merging from sibling runs.
    """

    def __init__(self, pool_dir: str, max_pool_size: int = 5, device: str = "cpu"):
        self.pool_dir = pool_dir
        self.max_pool_size = max_pool_size
        self.device = device
        self.pool: List[Dict] = []
        os.makedirs(pool_dir, exist_ok=True)
        logger.info("[FederatedPool] Initialized pool at {} (max_size={})".format(pool_dir, max_pool_size))

    # --- Pool management ---

    def add_checkpoint(self, agent, score: float, episode: int, metadata: Optional[Dict] = None) -> str:
        """Save a checkpoint to pool and evict the lowest-scoring entry if over limit."""
        fname = "pool_ep{:06d}_score{:.4f}.torch".format(episode, score)
        path = os.path.join(self.pool_dir, fname)
        torch.save(agent.state_dict(), path)
        entry = {"path": path, "score": score, "episode": episode, "metadata": metadata or {}}
        self.pool.append(entry)
        self.pool.sort(key=lambda x: x["score"], reverse=True)
        while len(self.pool) > self.max_pool_size:
            evicted = self.pool.pop()
            ename = os.path.basename(evicted["path"])
            escore = evicted["score"]
            if os.path.exists(evicted["path"]):
                os.remove(evicted["path"])
                logger.debug("[FederatedPool] Evicted {} (score={:.4f})".format(ename, escore))
        logger.info("[FederatedPool] Added ep={} score={:.4f} | pool_size={}".format(episode, score, len(self.pool)))
        return path

    def best_entry(self) -> Optional[Dict]:
        return self.pool[0] if self.pool else None

    def load_best(self, map_location=None):
        entry = self.best_entry()
        if entry is None:
            logger.warning("[FederatedPool] Pool is empty - nothing to load")
            return None
        agent = torch.load(entry["path"], map_location=map_location or self.device, weights_only=False)
        logger.info("[FederatedPool] Loaded best checkpoint (score={:.4f}, ep={})".format(entry["score"], entry["episode"]))
        return agent

    def load_best_into(self, agent, map_location=None):
        entry = self.best_entry()
        if entry is None:
            return False
        sd = torch.load(entry["path"], map_location=map_location or self.device)
        agent.load_state_dict(sd)
        return True
    
    # --- FedAvg ---

    def fedavg(self, target_agent, weights: Optional[List[float]] = None, top_k: Optional[int] = None):
        """Apply FedAvg: average pool parameters into target_agent."""
        pool = self.pool[:top_k] if top_k else self.pool
        if not pool:
            logger.warning("[FederatedPool] fedavg called on empty pool - skipping")
            return target_agent
        if weights is None:
            scores = np.array([e["score"] for e in pool], dtype=np.float64)
            scores = np.clip(scores, 0, None)
            total = scores.sum()
            weights = (scores / total).tolist() if total > 0 else [1.0 / len(pool)] * len(pool)
        assert len(weights) == len(pool)
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
        agents = []
        for entry in pool:
            try:
                a = torch.load(entry["path"], map_location=self.device, weights_only=False)
                agents.append(a)
            except Exception as exc:
                logger.warning("[FederatedPool] Failed to load {}: {}".format(entry["path"], exc))
        if not agents:
            return target_agent
        avg_sd = copy.deepcopy(agents[0].state_dict())
        for key in avg_sd:
            avg_sd[key] = avg_sd[key].float() * weights[0]
            for i, ag in enumerate(agents[1:], start=1):
                param = ag.state_dict()[key].float()
                if param.shape == avg_sd[key].shape:
                    avg_sd[key] += param * weights[i]
        target_agent.load_state_dict(avg_sd, strict=False)
        logger.info("[FederatedPool] FedAvg applied from {} agents".format(len(agents)))
        return target_agent

    # --- External pool merge ---

    @classmethod
    def discover_external_checkpoints(cls, search_dirs: List[str], pattern: str = "*_best.torch") -> List[str]:
        found = []
        for d in search_dirs:
            found.extend(glob.glob(os.path.join(d, pattern)))
        found = list(set(found))
        logger.info("[FederatedPool] Discovered {} external checkpoints".format(len(found)))
        return found

    def merge_external(self, target_agent, external_paths: List[str], own_weight: float = 0.5):
        """Merge external .torch checkpoints into target_agent via FedAvg."""
        if not external_paths:
            return target_agent
        ext_weight = (1.0 - own_weight) / len(external_paths)
        agents = [target_agent]
        weights = [own_weight]
        for p in external_paths:
            try:
                a = torch.load(p, map_location=self.device, weights_only=False)
                agents.append(a)
                weights.append(ext_weight)
            except Exception as exc:
                logger.warning("[FederatedPool] Cannot load external {}: {}".format(p, exc))
        if len(agents) == 1:
            return target_agent
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
        avg_sd = copy.deepcopy(agents[0].state_dict())
        for key in avg_sd:
            avg_sd[key] = avg_sd[key].float() * weights[0]
            for i, ag in enumerate(agents[1:], start=1):
                try:
                    param = ag.state_dict()[key].float()
                    if param.shape == avg_sd[key].shape:
                        avg_sd[key] += param * weights[i]
                except KeyError:
                    pass
        target_agent.load_state_dict(avg_sd, strict=False)
        logger.info("[FederatedPool] Merged {} external checkpoints (own_weight={})".format(len(agents)-1, own_weight))
        return target_agent

    # --- Persistence ---

    def save_manifest(self) -> str:
        import json
        manifest_path = os.path.join(self.pool_dir, "pool_manifest.json")
        data = [{"path": e["path"], "score": e["score"], "episode": e["episode"]} for e in self.pool]
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        return manifest_path

    def load_manifest(self):
        import json
        manifest_path = os.path.join(self.pool_dir, "pool_manifest.json")
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path) as f:
            data = json.load(f)
        for entry in data:
            if os.path.exists(entry["path"]):
                self.pool.append({"path": entry["path"], "score": entry["score"], "episode": entry["episode"], "metadata": {}})
        self.pool.sort(key=lambda x: x["score"], reverse=True)
        logger.info("[FederatedPool] Restored {} entries from manifest".format(len(self.pool)))


# --- Standalone helpers ---

def fedavg_from_files(target_agent, checkpoint_files: List[str], device: str = "cpu", own_weight: float = 0.7):
    """One-shot FedAvg from checkpoint files into target_agent."""
    if not checkpoint_files:
        return target_agent
    ext_w = (1.0 - own_weight) / len(checkpoint_files)
    agents = [target_agent]
    weights = [own_weight]
    for p in checkpoint_files:
        if not os.path.exists(p):
            continue
        try:
            a = torch.load(p, map_location=device, weights_only=False)
            agents.append(a)
            weights.append(ext_w)
        except Exception as exc:
            logger.warning("[fedavg_from_files] Skipping {}: {}".format(p, exc))
    if len(agents) == 1:
        return target_agent
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]
    avg_sd = copy.deepcopy(agents[0].state_dict())
    for key in avg_sd:
        avg_sd[key] = avg_sd[key].float() * weights[0]
        for i, ag in enumerate(agents[1:], start=1):
            try:
                param = ag.state_dict()[key].float()
                if param.shape == avg_sd[key].shape:
                    avg_sd[key] += param * weights[i]
            except KeyError:
                pass
    target_agent.load_state_dict(avg_sd, strict=False)
    logger.info("[fedavg_from_files] Merged {} agents".format(len(agents)))
    return target_agent


def select_best_checkpoint(checkpoint_dir: str, pattern: str = "*.torch") -> Optional[str]:
    """Pick best checkpoint: prefer files with "best" in name, else most recent."""
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not files:
        return None
    best_files = [f for f in files if "best" in os.path.basename(f).lower()]
    if best_files:
        return max(best_files, key=os.path.getmtime)
    return max(files, key=os.path.getmtime)
