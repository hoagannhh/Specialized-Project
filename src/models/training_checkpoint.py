"""Checkpoint huấn luyện giữa epoch, dùng chung cho bước 0 và tầng 3/4.

`last` để tiếp tục train; `best` vẫn là bundle suy luận đã chọn trên validation.
Không dựa vào bắt SIGKILL: luôn lưu trước, chỉ mất phần sau lần lưu gần nhất.
"""
import csv
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class TrainingPaused(Exception):
    """Dừng có chủ đích khi hết ngân sách thời gian của lần chạy."""


def pause_cleanly(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TrainingPaused as exc:
            print(str(exc), flush=True)
            return None
    return wrapped


def atomic_save(payload, path):
    """Ghi tệp tạm cùng ổ đĩa rồi thay thế; bản cũ còn nguyên nếu ghi bị ngắt."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rng_state():
    np_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda'])


class EpochBatchSampler(Sampler):
    """Tái tạo thứ tự theo seed/epoch, bỏ chỉ số đã học trước khi token hóa."""
    def __init__(self, size, batch_size, seed, epoch, start_batch=0):
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.start_batch = seed, epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        for start in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield indices[start:start + self.batch_size]

    def __len__(self):
        return max(0, (self.size + self.batch_size - 1) // self.batch_size - self.start_batch)


class TrainingRun:
    def __init__(self, output_dir, tier, train_csv, val_csv, config,
                 resume_from=None, save_every_steps=500, save_every_minutes=10,
                 max_runtime_minutes=None):
        # Đồng hồ tính cả đọc dữ liệu, khởi tạo model và validation.
        self.started = time.monotonic()
        if save_every_steps < 1 or save_every_minutes <= 0:
            raise ValueError('Chu kỳ lưu checkpoint phải lớn hơn 0')
        if max_runtime_minutes is not None and max_runtime_minutes <= 0:
            raise ValueError('max_runtime_minutes phải lớn hơn 0 hoặc None')
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f'tang{tier}_last.pt'
        self.tier = tier
        self.config = dict(config, train_sha256=file_hash(train_csv), val_sha256=file_hash(val_csv))
        self.save_steps, self.save_seconds = save_every_steps, save_every_minutes * 60
        self.deadline = None if max_runtime_minutes is None else self.started + max_runtime_minutes * 60
        self.last_saved = time.monotonic()
        self.state = {'epoch': 1, 'next_batch': 0, 'global_step': 0,
                      'total_loss': 0.0, 'best_metric': float('-inf'),
                      'bad_epochs': 0, 'finished': False}
        self.history = {'steps': [], 'epochs': []}
        self.payload = None
        self.source = None
        if resume_from:
            self.source = Path(resume_from)
            if self.source.is_dir():
                self.source = self.source / f'tang{tier}_last.pt'
            if (self.source.resolve() != self.path.resolve()
                    and (self.path.exists() or (self.directory / f'tang{tier}_best.pt').exists())):
                raise FileExistsError('Đích resume đã có model khác; dùng output-dir mới')
            self.payload = torch.load(self.source, map_location='cpu', weights_only=True)
            if self.payload.get('training_checkpoint_version') != 1 or self.payload.get('tier') != tier:
                raise ValueError('Cần checkpoint last đúng tầng, không dùng best để resume optimizer')
            if self.payload['config'] != self.config:
                raise ValueError('Dữ liệu hoặc cấu hình train khác checkpoint; không thể resume cùng thí nghiệm')
            self.state = dict(self.payload['progress'])
            self.history = self.payload.get('history', {'steps': [], 'epochs': []})
        elif self.path.exists() or (self.directory / f'tang{tier}_best.pt').exists():
            raise FileExistsError('Đã có checkpoint: truyền --resume-from hoặc dùng output-dir mới')

    @property
    def encoder_config(self):
        return None if self.payload is None else self.payload['encoder_config']

    @property
    def tokenizer_source(self):
        return self.config['pretrained_name'] if self.source is None else str(self.source.parent)

    def attach(self, model, tokenizer, optimizer):
        self.model, self.optimizer = model, optimizer
        tokenizer.save_pretrained(self.directory)
        if self.payload is not None:
            model.load_state_dict(self.payload['model'])
            optimizer.load_state_dict(self.payload['optimizer'])
            # best đi kèm last để resume từ Kaggle Input sang Working không mất best cũ.
            best = self.payload['best_bundle']
            if best is not None:
                atomic_save(best, self.directory / f'tang{self.tier}_best.pt')
                (self.directory / 'model_config.json').write_text(
                    json.dumps(best['metadata'], ensure_ascii=False, indent=2), encoding='utf-8')
            else:
                # Có thể bị ngắt sau ghi best nhưng trước khi commit tiến độ epoch.
                # last là mốc khôi phục; không dùng best chưa thuộc mốc đó.
                best_path = self.directory / f'tang{self.tier}_best.pt'
                if best_path.exists():
                    best_path.replace(self.directory / f'tang{self.tier}_uncommitted.pt')
                    (self.directory / 'model_config.json').unlink(missing_ok=True)
            restore_rng(self.payload['rng'])
            self.payload = None
            print(f"Resume tầng {self.tier}: epoch={self.state['epoch']}, "
                  f"next_batch={self.state['next_batch']}, step={self.state['global_step']}", flush=True)
        # Có checkpoint hợp lệ ngay trước batch đầu tiên, kể cả chưa xong epoch 1.
        self.save()
        self.check_time()

    def loader(self, dataset, batch_size):
        sampler = EpochBatchSampler(len(dataset), batch_size, self.config['seed'],
                                    self.state['epoch'], self.state['next_batch'])
        return DataLoader(dataset, batch_sampler=sampler,
                          generator=torch.Generator().manual_seed(self.config['seed']))

    def save(self):
        best_path = self.directory / f'tang{self.tier}_best.pt'
        best = torch.load(best_path, map_location='cpu', weights_only=True) if best_path.exists() else None
        atomic_save({'training_checkpoint_version': 1, 'tier': self.tier,
                     'config': self.config, 'encoder_config': self.model.encoder.config.to_dict(),
                     'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                     'progress': self.state, 'history': self.history, 'rng': rng_state(), 'best_bundle': best}, self.path)
        status = dict(self.state, tier=self.tier, checkpoint=str(self.path),
                      has_validated_best=best is not None)
        status_path = self.directory / 'training_status.json'
        temp = status_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(status_path)
        self.export_history()
        self.last_saved = time.monotonic()
        print(f"[checkpoint] {self.path.name}: epoch={self.state['epoch']}, "
              f"next_batch={self.state['next_batch']}, step={self.state['global_step']}", flush=True)

    def check_time(self, save=True):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            if save:
                self.save()
            raise TrainingPaused(f'Tạm dừng theo ngân sách thời gian. Resume từ {self.path}')

    def after_step(self, batch_index, loss):
        self.state['next_batch'] = batch_index + 1
        self.state['global_step'] += 1
        self.state['total_loss'] += float(loss)
        self.history['steps'].append({
            'epoch': self.state['epoch'], 'batch': batch_index + 1,
            'global_step': self.state['global_step'], 'train_loss': float(loss),
            'learning_rate': float(self.optimizer.param_groups[0]['lr'])})
        if (self.state['global_step'] % self.save_steps == 0
                or time.monotonic() - self.last_saved >= self.save_seconds):
            self.save()
        if self.state['global_step'] % 50 == 0:
            print(f"[train] epoch={self.state['epoch']} batch={self.state['next_batch']} "
                  f"step={self.state['global_step']} loss={float(loss):.4f}", flush=True)
        self.check_time()

    def export_history(self):
        # CSV được tái tạo từ cùng checkpoint: resume không giữ/trùng các bước chưa commit.
        columns = {
            'steps': ['epoch', 'batch', 'global_step', 'train_loss', 'learning_rate'],
            'epochs': ['epoch', 'global_step', 'train_loss', 'val_loss', 'val_f1',
                       'val_pr_auc', 'val_accuracy', 'threshold', 'learning_rate']}
        for kind, fields in columns.items():
            path = self.directory / f'{kind}_history.csv'
            temp = path.with_suffix('.csv.tmp')
            with temp.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.history[kind])
            temp.replace(path)

    def finish_epoch(self, best_metric, bad_epochs, finished, metrics=None):
        if metrics is not None:
            self.history['epochs'].append(dict(
                epoch=self.state['epoch'], global_step=self.state['global_step'],
                learning_rate=float(self.optimizer.param_groups[0]['lr']), **metrics))
        self.state.update(epoch=self.state['epoch'] + 1, next_batch=0, total_loss=0.0,
                          best_metric=float(best_metric), bad_epochs=bad_epochs, finished=finished)
        self.save()
        self.check_time()


def add_checkpoint_arguments(parser):
    parser.add_argument('--resume-from', default=None, help='Thư mục hoặc file tangN_last.pt')
    parser.add_argument('--save-every-steps', type=int, default=500)
    parser.add_argument('--save-every-minutes', type=float, default=10)
    parser.add_argument('--max-runtime-minutes', type=float, default=None,
                        help='Dừng chủ động và lưu last; tính cho mỗi lần gọi train')
