"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import bisect
import queue
import threading
from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


class _BestFitDocs:
    """Documents bucketed by length.

    ``pop_best_fit`` returns the oldest document of the greatest length that
    still fits, matching a linear scan that keeps the first maximum. ``pop_shortest``
    is the crop source: the oldest document of the smallest length.
    """

    def __init__(self):
        self._buckets: dict[int, deque] = {}
        self._lengths: list[int] = []
        self.n = 0

    def append(self, tokens: torch.Tensor):
        length = tokens.numel()
        bucket = self._buckets.get(length)
        if bucket is None:
            bucket = deque()
            self._buckets[length] = bucket
            bisect.insort(self._lengths, length)
        bucket.append(tokens)
        self.n += 1

    def _pop_len(self, length: int) -> torch.Tensor:
        bucket = self._buckets[length]
        doc = bucket.popleft()
        if not bucket:
            del self._buckets[length]
            del self._lengths[bisect.bisect_left(self._lengths, length)]
        self.n -= 1
        return doc

    def pop_best_fit(self, remaining: int):
        # Length 0 never wins the strict ``> best_len`` scan (best_len starts at 0).
        idx = bisect.bisect_right(self._lengths, remaining) - 1
        if idx < 0 or self._lengths[idx] == 0:
            return None
        return self._pop_len(self._lengths[idx])

    def pop_shortest(self) -> torch.Tensor:
        if not self._lengths:
            raise RuntimeError("best-fit document buffer is empty")
        return self._pop_len(self._lengths[0])


def _prefetch_iterator(iterable, depth=3):
    """Run ``iterable`` on a daemon thread and yield its items.

    The training step calls ``next()`` inside the timed loop. Packing documents
    is pure Python; doing it on this thread overlaps it with the TPU step.
    ``depth`` is how many finished batches may sit in memory ahead of the consumer.
    """
    filled = queue.Queue(maxsize=depth)

    def _produce():
        try:
            for item in iterable:
                filled.put((True, item))
        except Exception as exc:
            filled.put((False, exc))
        else:
            filled.put((None, None))

    threading.Thread(target=_produce, name="nanochat-data", daemon=True).start()
    while True:
        ok, item = filled.get()
        if ok is None:
            return
        if not ok:
            raise item
        yield item


def _allocate_token_buffers(B, T, device, pin_memory):
    """Allocate pinned CPU staging and owning device tensors for a (B, T) batch.

    Device inputs/targets must own their (B, T) storage rather than slicing a
    single packed 2*B*T device buffer: TorchTPU's compiler can record the base
    2*B*T buffer on a slice view during graph lowering and then fail with
    RuntimeProgramInputMismatch when the materialized B*T tensor is passed to
    embedding_dense_backward.
    """
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=pin_memory)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = torch.empty((B, T), dtype=torch.long, device=device)
    targets = torch.empty((B, T), dtype=torch.long, device=device)
    return cpu_inputs, cpu_targets, inputs, targets


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="tpu", resume_state_dict=None,
    buffer_size=1000, prefetch=False,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    docs = _BestFitDocs()
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            docs.append(torch.tensor(tokens, dtype=torch.long))

    # Device ring matches the longest TPU queue (4). CPU ring is the prefetch
    # depth plus two, so a slot can be packed and copied without page-locking
    # a fresh buffer every microbatch. A CPU slot is reused only after four
    # later copies have been issued, which is after the training loop has synced.
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    use_accel = device_type == "tpu"
    device_ring = 4
    prefetch_depth = 3
    cpu_ring = prefetch_depth + 2
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_inputs = []
    cpu_targets = []
    for _ in range(cpu_ring):
        ci, ct, _, _ = _allocate_token_buffers(B, T, device="cpu", pin_memory=use_accel)
        cpu_inputs.append(ci)
        cpu_targets.append(ct)
    device_inputs = []
    device_targets = []
    for _ in range(device_ring):
        _, _, di, dt = _allocate_token_buffers(B, T, device=device, pin_memory=False)
        device_inputs.append(di)
        device_targets.append(dt)

    def fill_slot(slot):
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while docs.n < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                doc = docs.pop_best_fit(remaining)
                if doc is None:
                    # Nothing fits. Crop the shortest document and drop its tail.
                    doc = docs.pop_shortest()
                    row_buffer[row_idx, pos:pos + remaining].copy_(doc[:remaining])
                    pos += remaining
                else:
                    n = doc.numel()
                    row_buffer[row_idx, pos:pos + n].copy_(doc)
                    pos += n
        cpu_inputs[slot].copy_(row_buffer[:, :-1])
        cpu_targets[slot].copy_(row_buffer[:, 1:])
        return {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

    free_slots = queue.Queue()
    for slot in range(cpu_ring):
        free_slots.put(slot)
    ready = queue.Queue(maxsize=prefetch_depth)

    def _produce():
        try:
            while True:
                slot = free_slots.get()
                state = fill_slot(slot)
                ready.put((True, slot, state))
        except Exception as exc:
            ready.put((False, exc, None))

    if prefetch:
        threading.Thread(target=_produce, name="nanochat-data", daemon=True).start()

    in_flight = deque()
    dev_idx = 0
    while True:
        if prefetch:
            ok, slot, state_dict = ready.get()
            if not ok:
                raise slot
        else:
            slot = free_slots.get()
            state_dict = fill_slot(slot)
        inputs = device_inputs[dev_idx]
        targets = device_targets[dev_idx]
        dev_idx = (dev_idx + 1) % device_ring
        inputs.copy_(cpu_inputs[slot], non_blocking=use_accel)
        targets.copy_(cpu_targets[slot], non_blocking=use_accel)
        in_flight.append(slot)
        if len(in_flight) > device_ring:
            free_slots.put(in_flight.popleft())
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
