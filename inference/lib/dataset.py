from math import nan
import math
import os
import random
import time

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import librosa

import hashlib

try:
    from lib import spec_utils
except ModuleNotFoundError:
    import spec_utils

class VocalAutoregressiveDataset(torch.utils.data.Dataset):
    def __init__(self, path, extra_path=None, pair_path=None, mix_path=None, vocal_path="", is_validation=False, mul=1, downsamples=0, epoch_size=None, pair_mul=1, slide=True, cropsize=256, mixup_rate=0, mixup_alpha=1):
        self.epoch_size = epoch_size
        self.slide = slide
        self.mul = mul
        patch_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.cropsize = cropsize
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.token_size = 8
        pair_list = []

        if pair_path is not None and pair_mul > 0:
            pairs = [os.path.join(pair_path, f) for f in os.listdir(pair_path) if os.path.isfile(os.path.join(pair_path, f))]

            for p in pairs:
                if pair_mul > 1:
                    for _ in range(pair_mul):
                        pair_list.append(p)
                else:
                    if np.random.uniform() < pair_mul:
                        pair_list.append(p)

        if mix_path is not None:
            mixes = [os.path.join(mix_path, f) for f in os.listdir(mix_path) if os.path.isfile(os.path.join(mix_path, f))]

            for m in mixes:
                if pair_mul > 1:
                    for _ in range(pair_mul):
                        pair_list.append(m)
                else:
                    if np.random.uniform() < pair_mul:
                        pair_list.append(m)

        if extra_path is not None:
            extra_list = [os.path.join(extra_path, f) for f in os.listdir(extra_path) if os.path.isfile(os.path.join(extra_path, f))]

            for f in extra_list:
                patch_list.append(f) 
        
        if not is_validation and vocal_path != "":
            self.vocal_list = [os.path.join(vocal_path, f) for f in os.listdir(vocal_path) if os.path.isfile(os.path.join(vocal_path, f))]

        self.is_validation = is_validation

        self.curr_list = []
        for p in patch_list:
            self.curr_list.append(p)

        for p in pair_list:
            self.curr_list.append(p)

        self.downsamples = downsamples
        self.full_list = self.curr_list

        if not is_validation and self.epoch_size is not None:
            end = math.ceil(len(self.full_list) * self.epoch_size)
            self.curr_list = self.full_list[:end]            
            self.rebuild()

    def rebuild(self):
        self.vidx = 0
        random.shuffle(self.vocal_list)

        if self.epoch_size is not None:
            end = math.ceil(len(self.full_list) * self.epoch_size)
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:end]

    def __len__(self):
        return len(self.curr_list) * self.mul

    def _get_vocals(self, root=True):
        vidx = np.random.randint(len(self.vocal_list))                
        vpath = self.vocal_list[vidx]
        vdata = np.load(str(vpath))
        V, Vc = vdata['X'], vdata['c']

        if np.random.uniform() < 0.5:
            V = V[::-1]

        if np.random.uniform() < 0.025:
            if np.random.uniform() < 0.5:
                V[0] = V[0] * 0
            else:
                V[1] = V[1] * 0

        if self.slide:
            start = np.random.randint(0, V.shape[2] - self.cropsize - 2)
            stop = start + self.cropsize + 1
            V = V[:,:,start:stop]

        if np.random.uniform() < 0.5 and root:
            V2, Vc2 = self._get_vocals(root=False)
            a = np.random.beta(1, 1)
            inv = 1 - a

            Vc = (Vc * a) + (Vc2 * inv)
            V = (V * a) + (V2 * inv)

        return V, Vc

    def __getitem__(self, idx, root=True):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        aug = 'Y' not in data.files
        X, Xc = data['X'], data['c']
        Y = X if aug else data['Y']
        vocals = 'vocals' in data.files
        
        if not self.is_validation:
            start = np.random.randint(0, X.shape[2] - self.cropsize - 1)
            stop = start + self.cropsize

            Y = X[:,:,start+1:stop+1]
            X = X[:,:,start:stop]

            if aug and np.random.uniform() < 0.05 and not vocals:
                V, Vc = self._get_vocals()
                X = X + V[:,:,:-1]
                Y = Y + V[:,:,1:]
                c = np.max([Xc, Vc])
            else:
                c = Xc

            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]

            if np.random.uniform() < 0.025:
                X = Y
                c = Xc
        else:
            c = Xc
            start = np.random.randint(0, X.shape[2] - self.cropsize - 1)
            stop = start + self.cropsize

            if np.random.uniform() < 0.5:
                Y = X[:,:,start+1:stop+1]
                X = X[:,:,start:stop]
            else:
                X = Y[:,:,start:stop]
                Y = Y[:,:,start+1:stop+1]

        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)

        if np.random.uniform() < self.mixup_rate and root and not self.is_validation:
            MX, MY = self.__getitem__(np.random.randint(len(self)), root=False)
            a = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            X = X * a + (1 - a) * MX
            Y = Y * a + (1 - a) * MY

        return X, Y

class DenoisingPretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, path, gamma=0.3, epoch_size=None, cropsize=256):
        self.epoch_size = epoch_size
        self.cropsize = cropsize
        self.gamma = gamma

        self.curr_list = []
        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                self.curr_list.append(m)
        
        self.full_list = self.curr_list

        self.rebuild()

    def rebuild(self):
        if self.epoch_size is not None:
            end = math.ceil(len(self.full_list) * self.epoch_size)
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:end]

    def __len__(self):
        return len(self.curr_list)

    def __getitem__(self, idx, root=True):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        aug = 'Y' not in data.files
        X, Xc = data['X'], data['c']
        Y = X if aug else data['Y']
        
        if not self.is_validation:
            c = Xc

            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]
        else:
            c = Xc

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]
            Y = Y[:,:,start:stop]
            
        X = (np.abs(X) / c) * 2 - 1.0
        E = np.random.normal(scale=self.gamma, size=X.shape)
        
        return X + E, E

class MaskedPretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, path, is_validation=False, mul=1, downsamples=0, epoch_size=None, pair_mul=1, slide=True, cropsize=256, mixup_rate=0, mixup_alpha=1, mask_rate=0.15, next_frame_chunk_size=16, token_size=16, target_token_size=32, current_step=0, num_steps=16000, start_step=0, mask_indices=[]):
        self.epoch_size = epoch_size
        self.slide = slide
        self.mul = mul
        self.cropsize = cropsize
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.mask_rate = mask_rate
        self.next_frame_chunk_size = next_frame_chunk_size
        self.random_ending_rate = 0
        self.mask_indices = []
        self.starting_token_size = token_size
        self.token_size = token_size
        self.target_token_size = target_token_size
        self.warmup_steps = num_steps
        self.separator_size = target_token_size // 4
        self.current_step = current_step
        self.start_step = start_step
        self.is_validation = is_validation
        self.separator_token = None

        self.curr_list = []
        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                self.curr_list.append(m)
        
        self.downsamples = downsamples
        self.full_list = self.curr_list

        if not is_validation and self.epoch_size is not None:
            end = math.ceil(len(self.full_list) * self.epoch_size)
            self.curr_list = self.full_list[:end]            
            self.rebuild()

    def rebuild(self):
        if self.epoch_size is not None:
            end = math.ceil(len(self.full_list) * self.epoch_size)
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:end]

    def __len__(self):
        return len(self.curr_list) * self.mul

    def __getitem__(self, idx, root=True):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        aug = 'Y' not in data.files
        X, Xc = data['X'], data['c']
        Y = X if aug else data['Y']
        
        if not self.is_validation:
            c = Xc

            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]
        else:
            c = Xc

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]
            Y = Y[:,:,start:stop]

        if np.random.uniform() < self.mixup_rate and root and not self.is_validation:
            MX, _, _, _ = self.__getitem__(np.random.randint(len(self)), root=False)
            a = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            X = X * a + (1 - a) * MX

        self.current_step = self.current_step + 1
        
        starts = []
        index_count = None
        indices = None
        if root:
            self.current_step = self.current_step + 1
            noise = np.random.uniform(0, 1, X.shape)
            num_tokens = self.cropsize // self.token_size
                                    
            X = np.clip(np.abs(X) / c, 0, 1)
            Y = X.copy()

            for token in range(num_tokens):
                if (np.random.uniform() < self.mask_rate and len(self.mask_indices) == 0) or token in self.mask_indices:
                    start = token * self.token_size
                    stop = start + self.token_size
                    starts.append(start)
            
                    X[:, :, start:stop] = 1.0

                    if np.random.uniform() < 0.2:
                        if np.random.uniform() < 0.5:
                            X[:, :, start:stop] = np.clip(Y[:, :, start:stop] + noise[:, :, start:stop], 0, 1)
                        else:
                            X[:, :, start:stop] = Y[:, :, start:stop]

            if len(starts) == 0:
                num_tokens = self.cropsize // self.token_size
                token = np.random.randint(0, num_tokens)
                start = token * self.token_size
                stop = start + self.token_size
                starts.append(start)
        
                X[:, :, start:stop] = 1.0

                if np.random.uniform() < 0.2:
                    if np.random.uniform() < 0.5:
                        X[:, :, start:stop] = np.clip(Y[:, :, start:stop] + noise[:, :, start:stop], 0, 1)
                    else:
                        X[:, :, start:stop] = Y[:, :, start:stop]

            index_count = len(starts)
            indices = np.pad(np.array(starts), (0, num_tokens - len(starts)))

        return X, Y, index_count, indices

class VocalAugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path="", is_validation=False, mul=1, downsamples=0, epoch_size=None, cropsize=256, vocal_mixup_rate=0, mixup_rate=0, mixup_alpha=1, include_phase=False, force_voxaug=False):
        self.epoch_size = epoch_size
        self.mul = mul
        patch_list = []
        self.cropsize = cropsize
        self.vocal_mixup_rate = vocal_mixup_rate
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.include_phase = include_phase
        self.force_voxaug = force_voxaug

        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                patch_list.append(m)
        
        if not is_validation and vocal_path != "":
            self.vocal_list = [os.path.join(vocal_path, f) for f in os.listdir(vocal_path) if os.path.isfile(os.path.join(vocal_path, f))]

        self.is_validation = is_validation

        self.curr_list = []
        for p in patch_list:
            self.curr_list.append(p)

        self.downsamples = downsamples
        self.full_list = self.curr_list

        if not is_validation and self.epoch_size is not None:
            self.curr_list = self.full_list[:self.epoch_size]            
            self.rebuild()

    def rebuild(self):
        self.vidx = 0
        random.shuffle(self.vocal_list)

        if self.epoch_size is not None:
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:self.epoch_size]

    def __len__(self):
        return len(self.curr_list) * self.mul

    def _get_vocals(self, root=True):
        vidx = np.random.randint(len(self.vocal_list))                
        vpath = self.vocal_list[vidx]
        vdata = np.load(str(vpath))
        V = vdata['X']

        if np.random.uniform() < 0.5:
            V = V[::-1]

        if np.random.uniform() < 0.025:
            if np.random.uniform() < 0.5:
                V[0] = V[0] * 0
            else:
                V[1] = V[1] * 0

        if np.random.uniform() < self.vocal_mixup_rate and root:
            V2 = self._get_vocals(root=False)
            V = V + V2

        return V

    def __getitem__(self, idx, root=True):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        aug = 'Y' not in data.files
        X, Xc = data['X'], data['c']
        Y = X if aug else data['Y']

        if self.force_voxaug:
            if not aug:
                X = Y
                aug = True

        if not self.is_validation:
            if aug and np.random.uniform() > 0.02:
                V = self._get_vocals()
                X = Y + V
                c = np.max([Xc, np.abs(X).max()])
            else:
                c = Xc

            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]

            if np.random.uniform() < 0.025:
                X = Y
                c = Xc
        else:
            c = Xc

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]
            Y = Y[:,:,start:stop]

        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)

        if np.random.uniform() < self.mixup_rate and root and not self.is_validation:
            MX, MY = self.__getitem__(np.random.randint(len(self)), root=False)
            a = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            X = X * a + (1 - a) * MX
            Y = Y * a + (1 - a) * MY

        return X, Y

class VocalRemoverTrainingSet(torch.utils.data.Dataset):
    def __init__(self, training_set, cropsize, reduction_rate, reduction_weight, mixup_rate, mixup_alpha):
        self.training_set = training_set
        self.cropsize = cropsize
        self.reduction_rate = reduction_rate
        self.reduction_weight = reduction_weight
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.training_set)

    def do_crop(self, X_path, y_path):
        X_mmap = np.load(X_path, mmap_mode='r')
        y_mmap = np.load(y_path, mmap_mode='r')

        start = np.random.randint(0, X_mmap.shape[2] - self.cropsize)
        end = start + self.cropsize

        X_crop = np.array(X_mmap[:, :, start:end], copy=True)
        y_crop = np.array(y_mmap[:, :, start:end], copy=True)

        return X_crop, y_crop

    def do_aug(self, X, y):
        if np.random.uniform() < self.reduction_rate:
            y = spec_utils.aggressively_remove_vocal(X, y, self.reduction_weight)

        if np.random.uniform() < 0.5:
            # swap channel
            X = X[::-1].copy()
            y = y[::-1].copy()

        if np.random.uniform() < 0.01:
            # inst
            X = y.copy()

        # if np.random.uniform() < 0.01:
        #     # mono
        #     X[:] = X.mean(axis=0, keepdims=True)
        #     y[:] = y.mean(axis=0, keepdims=True)

        return X, y

    def do_mixup(self, X, y):
        idx = np.random.randint(0, len(self))
        X_path, y_path, coef = self.training_set[idx]

        X_i, y_i = self.do_crop(X_path, y_path)
        X_i /= coef
        y_i /= coef

        X_i, y_i = self.do_aug(X_i, y_i)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        X = lam * X + (1 - lam) * X_i
        y = lam * y + (1 - lam) * y_i

        return X, y

    def __getitem__(self, idx):
        X_path, y_path, coef = self.training_set[idx]

        X, y = self.do_crop(X_path, y_path)
        X /= coef
        y /= coef

        X, y = self.do_aug(X, y)

        if np.random.uniform() < self.mixup_rate:
            X, y = self.do_mixup(X, y)

        X_mag = np.abs(X)
        y_mag = np.abs(y)

        return X_mag, y_mag


class VocalRemoverValidationSet(torch.utils.data.Dataset):

    def __init__(self, patch_list):
        self.patch_list = patch_list

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        path = self.patch_list[idx]
        data = np.load(path)

        X, y = data['X'], data['y']

        X_mag = np.abs(X)
        y_mag = np.abs(y)

        return X_mag, y_mag


def make_pair(mix_dir, inst_dir, voxaug=False, pretraining=False):
    input_exts = ['.wav', '.m4a', '.mp3', '.mp4', '.flac']

    inst_dir = mix_dir if pretraining else inst_dir

    y_list = sorted([
        os.path.join(inst_dir, fname)
        for fname in os.listdir(inst_dir)
        if os.path.splitext(fname)[1] in input_exts])

    if not voxaug and not pretraining:
        X_list = sorted([
        os.path.join(mix_dir, fname)
        for fname in os.listdir(mix_dir)
        if os.path.splitext(fname)[1] in input_exts])
    else:
        X_list = y_list

    filelist = list(zip(X_list, y_list))

    return filelist


def train_val_split(dataset_dir, val_filelist, selected_validation=[], voxaug=False):
    filelist = make_pair(
        os.path.join(dataset_dir, 'mixtures'),
        os.path.join(dataset_dir, 'instruments'),
        voxaug=voxaug)

    train_filelist = list()
    val_filelist = list()

    for i, entry in enumerate(filelist):
        if len(selected_validation) > 0:
            validation_file = False
            for v in selected_validation:
                if entry[0].find(v) != -1:
                    validation_file = True
                    break
            
            if validation_file:
                val_filelist.append(entry)
            else:
                train_filelist.append(entry)
        else:
            train_filelist.append(entry)

    return train_filelist, val_filelist

def train_val_split(dataset_dir, val_filelist, val_size=-1, train_size=-1, selected_validation=[], voxaug=False, pretraining=False):
    filelist = make_pair(
        os.path.join(dataset_dir, 'mixtures'),
        os.path.join(dataset_dir, 'instruments'),
        voxaug=voxaug,
        pretraining=pretraining)

    train_filelist = list()
    val_filelist = list()

    for i, entry in enumerate(filelist):
        if len(selected_validation) > 0:
            validation_file = False

            for v in selected_validation:
                if entry[0].find(v) != -1:
                    validation_file = True
                    break
            
            if validation_file:
                val_filelist.append(entry)
            else:
                train_filelist.append(entry)
        else:
            train_filelist.append(entry)

    if len(selected_validation) == 0:
        if val_size != -1:
            val_filelist = train_filelist[-val_size:]

        if val_size == 0:
            train_filelist = train_filelist
        elif train_size != -1:
            if val_size != -1:
                train_filelist = train_filelist[-(train_size+val_size):-val_size]
            else:
                train_filelist = train_filelist[:train_size]
        elif val_size != -1:
            train_filelist = train_filelist[:-val_size]

    return train_filelist, val_filelist

def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def make_training_set(filelist, sr, hop_length, n_fft):
    ret = []
    for X_path, y_path in tqdm(filelist):
        X, y, X_cache_path, y_cache_path = spec_utils.cache_or_load(
            X_path, y_path, sr, hop_length, n_fft
        )
        coef = np.max([np.abs(X).max(), np.abs(y).max()])
        ret.append([X_cache_path, y_cache_path, coef])

    return ret

def get_oracle_data(X, y, oracle_loss, oracle_rate, oracle_drop_rate):
    k = int(len(X) * oracle_rate * (1 / (1 - oracle_drop_rate)))
    n = int(len(X) * oracle_rate)
    indices = np.argsort(oracle_loss)[::-1][:k]
    indices = np.random.choice(indices, n, replace=False)
    oracle_X = X[indices].copy()
    oracle_y = y[indices].copy()

    return oracle_X, oracle_y, indices

def make_vocal_stems(dataset, cropsize=1024, sr=44100, hop_length=512, n_fft=1024, offset=0, root=''):
    input_exts = ['.wav', '.m4a', '.mp3', '.mp4', '.flac']

    filelist = sorted([
        os.path.join(dataset, fname)
        for fname in os.listdir(dataset)
        if os.path.splitext(fname)[1] in input_exts])

    for i, X_path in enumerate(tqdm(filelist)):
        basename = os.path.splitext(os.path.basename(X_path))[0]
        xw, _ = spec_utils.load_wave(X_path, X_path, sr)

        patch_dir = '{}cs{}_sr{}_hl{}_nf{}_of{}{}'.format(root, cropsize, sr, hop_length, n_fft, 0, "_VOCALS")
        os.makedirs(patch_dir, exist_ok=True)

        X, _ = spec_utils.to_spec(xw, xw, hop_length=hop_length, n_fft=n_fft)
        coef = np.abs(X).max()

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(patch_dir, '{}_p{}.npz'.format(basename, j))

            start = j * roi_size
            if coef != 0:
                xp = X_pad[:, :, start:start + cropsize]
                xc = np.abs(xp).mean()

                if xc > 0:
                    np.savez(
                        outpath,
                        X=xp,
                        c=coef)

def make_dataset(filelist, cropsize, sr, hop_length, n_fft, offset=0, is_validation=False, suffix='', root=''):
    patch_list = []    
    patch_dir = f'{root}cs{cropsize}_sr{sr}_hl{hop_length}_nf{n_fft}_of{offset}{suffix}'
    os.makedirs(patch_dir, exist_ok=True)

    for X_path, Y_path in tqdm(filelist):
        basename = os.path.splitext(os.path.basename(X_path))[0]

        voxaug = X_path == Y_path

        X, Y = spec_utils.load(X_path, Y_path, sr, hop_length, n_fft)

        coef = np.max([np.abs(X).max(), np.abs(Y).max()])

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
        Y_pad = np.pad(Y, ((0, 0), (0, 0), (l, r)), mode='constant')

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(patch_dir, '{}_p{}.npz'.format(basename, j))

            found, i = not os.path.exists(outpath), 0
            while not found:
                outpath = os.path.join(patch_dir, '{}_p{}.{}.npz'.format(basename, j, i))
                
                if os.path.exists(outpath):
                    i = i + 1
                else:
                    found = True
                    break

            patch_list.append(outpath)

            start = j * roi_size
            if not os.path.exists(outpath):
                if voxaug:
                    np.savez(
                        outpath,
                        X=X_pad[:, :, start:start + cropsize],
                        c=coef.item())
                else:
                    np.savez(
                        outpath,
                        X=X_pad[:, :, start:start + cropsize],
                        Y=Y_pad[:, :, start:start + cropsize],
                        c=coef.item())

def make_validation_set(filelist, sr, hop_length, n_fft, offset=0, root=''):
    patch_list = []    
    patch_dir = f'{root}_sr{sr}_hl{hop_length}_nf{n_fft}_of{offset}_VALIDATION'
    os.makedirs(patch_dir, exist_ok=True)

    for X_path, Y_path in tqdm(filelist):
        basename = os.path.splitext(os.path.basename(X_path))[0]

        X, Y = spec_utils.load(X_path, Y_path, sr, hop_length, n_fft)

        coef = np.max([np.abs(X).max(), np.abs(Y).max()])

        outpath = os.path.join(patch_dir, '{}.npz'.format(basename))
        patch_list.append(outpath)

        if not os.path.exists(outpath):
            np.savez(
                outpath,
                X=X,
                Y=Y,
                c=coef.item())

if __name__ == "__main__":
    import sys
    import utils

    mix_dir = sys.argv[1]
    inst_dir = sys.argv[2]
    outdir = sys.argv[3]

    os.makedirs(outdir, exist_ok=True)

    filelist = make_pair(mix_dir, inst_dir)
    for mix_path, inst_path in tqdm(filelist):
        mix_basename = os.path.splitext(os.path.basename(mix_path))[0]

        X_spec, y_spec, _, _ = spec_utils.cache_or_load(
            mix_path, inst_path, 44100, 1024, 2048
        )

        X_mag = np.abs(X_spec)
        y_mag = np.abs(y_spec)
        v_mag = X_mag - y_mag
        v_mag *= v_mag > y_mag

        outpath = '{}/{}_Vocal.jpg'.format(outdir, mix_basename)
        v_image = spec_utils.spectrogram_to_image(v_mag)
        utils.imwrite(outpath, v_image)