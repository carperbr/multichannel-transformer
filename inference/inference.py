import argparse
import os
import shutil
import librosa
import numpy as np
import soundfile as sf
import music_tag
import torch
from tqdm import tqdm

from v7.libft2gan.frame_transformer4 import FrameTransformerGenerator as FrameTransformerV7
from v8.libft2gan.frame_transformer5 import FrameTransformer as FrameTransformerV8
from v9r.libft2gan.frame_transformer12 import FrameTransformer as FrameTransformerV9
from v10.libft2gan.frame_transformer13 import FrameTransformer as FrameTransformerV10

from lib import dataset
from lib import spec_utils
from lib import utils

class Separator(object):
    def __init__(self, model, device, batchsize, cropsize, n_fft, postprocess=False, autoregressive=False):
        self.model = model
        self.autoregressive = autoregressive
        self.offset = 0
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess
        self.n_fft = n_fft

    def _separate(self, X_mag_pad, cropsize=None, padding=None):
        X_dataset = []
        cropsize = self.cropsize if cropsize is None else cropsize
        padding = cropsize // 2 if padding is None else padding

        # autoregressive model doesn't work as well with the padding method; perhaps make this configurable.
        if self.autoregressive:
            cropsize = cropsize + padding * 2
            padding = 0

        patches = X_mag_pad.shape[2] // cropsize
        X_mag_pad = np.pad(X_mag_pad, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        for i in range(patches):
            start = (i * cropsize) + padding
            X_mag_crop = X_mag_pad[:, :, (start - padding):(start + cropsize + padding)]
            X_dataset.append(X_mag_crop)

        self.model.eval()

        with torch.no_grad():
            mask = []

            if self.autoregressive:
                PX = torch.zeros(1, 2, self.n_fft // 2, cropsize + padding * 2).to(self.device)
                PY = torch.zeros(1, 2, self.n_fft // 2, cropsize + padding * 2).to(self.device)
                PX2 = torch.zeros(1, 2, self.n_fft // 2, cropsize + padding * 2).to(self.device)
                PY2 = torch.zeros(1, 2, self.n_fft // 2, cropsize + padding * 2).to(self.device)

                for i in tqdm(range(0, patches)):
                    X_batch = X_dataset[i: i + 1]
                    X_batch = torch.from_numpy(np.asarray(X_batch)).to(self.device)[:, :, :(self.n_fft // 2)]
                    X_batch_1 = torch.cat((X_batch, PX, PY, PX2, PY2), dim=1)
                    
                    with torch.cuda.amp.autocast_mode.autocast(enabled=True):
                        mask_pred = torch.sigmoid(self.model(X_batch_1))

                    PX2 = PX
                    PY2 = PY
                    PX = X_batch_1[:, :2]
                    PY = X_batch_1[:, :2] * mask_pred
                        
                    if padding > 0:
                        pred = mask_pred[:, :, :, (padding):-(padding)]
                    else:
                        pred = mask_pred

                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
            else:
                for i in tqdm(range(0, patches, self.batchsize)):
                    X_batch = X_dataset[i: i + self.batchsize]
                    X_batch = torch.from_numpy(np.asarray(X_batch)).to(self.device)[:, :, :(self.n_fft // 2)]

                    with torch.cuda.amp.autocast_mode.autocast(enabled=True):
                        pred = torch.sigmoid(self.model(X_batch))

                    if padding > 0:
                        pred = pred[:, :, :, (padding):-(padding)]

                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        mask = np.pad(mask, ((0,0), (0,1), (0, 0)))

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        m_spec = mask * 255

        return y_spec, v_spec, m_spec

    def separate(self, X_spec, padding=None):
        X_mag, X_phase = self._preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, _ = dataset.make_padding(n_frame, self.cropsize, 0)
        xm = X_mag / X_mag.max()
        X_mag_pad = np.pad(xm, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        mask = self._separate(X_mag_pad, self.cropsize, padding)

        mask = mask[:, :, :n_frame]
        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec, m_spec

    def separate_tta(self, X_spec, cropsizes=[64, 128, 256, 512, 1024], paddings=[128, 256, 512, 1024, 2048]):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        X_mag_pad1 = X_mag / X_mag.max()

        mask = np.zeros_like(X_mag)

        for idx in range(len(paddings)):
            pad_l, pad_r, _ = dataset.make_padding(n_frame, paddings[idx], 0)
            X_mag_pad2 = np.pad(X_mag_pad1, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            mask += self._separate(X_mag_pad2, cropsizes[idx], paddings[idx])[:, :, :n_frame]

        mask = mask / len(paddings)

        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec, m_spec

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)

    p.add_argument('--model_v7', type=str, default='model.v7.pth')
    p.add_argument('--model_v8', type=str, default='model.v8.pth')
    p.add_argument('--model_v9', type=str, default='model.v9.pth')
    p.add_argument('--model_v10', type=str, default='model.v10.pth')
    p.add_argument('--models', type=str, default='v7,v8,v10')
    
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', type=str, default="")
    p.add_argument('--output_format', type=str, default="flac")
    p.add_argument('--num_res_encoders', type=int, default=4)
    p.add_argument('--num_res_decoders', type=int, default=4)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=1)
    p.add_argument('--cropsize', '-c', type=int, default=4096)
    p.add_argument('--padding', type=int, default=0)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--copy_source_images', action='store_true') # copies images from input into output
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--create_webm', action='store_true')
    p.add_argument('--create_vocals', action='store_true')
    p.add_argument('--num_encoders', type=int, default=2)
    p.add_argument('--num_decoders', type=int, default=13)
    # p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--cropsizes', type=str, default='128,256,512,1024')
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--num_transformer_blocks', type=int, default=2)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--rename_dir', type=str, default='false')
    
    p.add_argument('--num_attention_maps', type=int, default=1)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_bridge_layers', type=int, default=4)
    p.add_argument('--latent_expansion', type=int, default=4)
    p.add_argument('--expansion', type=float, default=2.2)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--weight_decay', type=float, default=1e-2)

    p.add_argument('--seed', type=int, default=1)

    p.add_argument('--num_res_blocks', type=int, default=1)
    p.add_argument('--feedforward_expansion', type=int, default=24)

    args = p.parse_args()
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    args.models = [model for model in args.models.split(',')]
    args.rename_dir = str.lower(args.rename_dir) == 'true'

    print('loading model...', end=' ')
    device = torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = []

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))

    if 'v7' in args.models:
        print('loading v7')
        model_v7 = FrameTransformerV7(in_channels=2, out_channels=2, channels=8, n_fft=2048, dropout=0, num_heads=8, num_attention_maps=2)
        model_v7.load_state_dict(torch.load(args.model_v7, map_location=device))
        model_v7.to(device)
        models.append(model_v7)

    if 'v8' in args.models:
        print('loading v8')
        model_v8 = FrameTransformerV8(in_channels=2, out_channels=2, channels=8, expansion=2.2, n_fft=2048, dropout=0, num_heads=8, num_attention_maps=1)
        model_v8.load_state_dict(torch.load(args.model_v8, map_location=device))
        model_v8.to(device)
        models.append(model_v8)

    if 'v9' in args.models:
        print('loading v9')
        model_v9 = FrameTransformerV9(in_channels=10, out_channels=2, channels=16, expansion=512, n_fft=2048, dropout=0, num_heads=8, num_attention_maps=1, num_layers=17)
        model_v9.load_state_dict(torch.load(args.model_v9, map_location=device))
        model_v9.to(device)
        models.append(model_v9)

    if 'v10' in args.models:
        print('loading v10')
        model_v10 = FrameTransformerV10(in_channels=2, out_channels=2, embedding=8, expansion=4, n_fft=2048, dropout=0, num_heads=8, num_attention_maps=1)
        model_v10.load_state_dict(torch.load(args.model_v10, map_location=device))
        model_v10.to(device)
        models.append(model_v10)
        
    print('done')

    output_folder = args.output
    if output_folder != '' and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_format = args.output_format.lower()
    if output_format not in ["flac", "wav", "mp3"]:
        output_format = "flac"

    cover = ""
    files = []

    if not os.path.isdir(args.input):
        files.append(args.input)
    else:        
        args.input = os.path.join(args.input, '')
        d = os.path.basename(os.path.dirname(args.input))
        print(output_folder)
        extensions = ['wav', 'm4a', 'mp3', 'mp4', 'flac', 'ogg']
        cover_ext = ["jpg", "png", "bmp"]

        
        if output_folder != '' and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for f in os.listdir(args.input):
            ext = f[::-1].split('.')[0][::-1]

            if ext in extensions:
                files.append(os.path.join(args.input, f))

            if ext in cover_ext:
                cover = os.path.join(args.input, f)
                if args.copy_source_images:
                    shutil.copy(cover, output_folder)

    for file in tqdm(files):
        print('\nloading wave source...', end=' ')
        X, sr = librosa.load(
            file, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
        basename = os.path.splitext(os.path.basename(file))[0]
        print('done')

        if X.ndim == 1:
            X = np.asarray([X, X])

        print('stft of wave source...', end=' ')
        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        print('done')

        y_specs = []
        v_specs = []

        for model in models:
            sp = Separator(model, device, args.batchsize, args.cropsize, args.n_fft, args.postprocess, model.autoregressive)
            y_spec, v_spec, _ = sp.separate(X_spec, padding=args.padding)
            y_specs.append(y_spec)
            v_specs.append(v_spec)
    
        y_mag = None
        v_mag = None
        y_phase = np.angle(y_specs[-1])
        v_phase = np.angle(v_specs[-1])

        for y_spec in y_specs:
            y_mag = np.abs(y_spec) if y_mag is None else np.minimum(y_mag, np.abs(y_spec))
            v_mag = np.abs(v_spec) if v_mag is None else np.maximum(v_mag, np.abs(v_spec))

        y_spec = y_mag * np.exp(1.j * y_phase)
        v_spec = v_mag * np.exp(1.j * v_phase)

        print('\ninverse stft of instruments...', end=' ')
        wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
        print('done')
        inst_file = f'{output_folder}/{basename}_Instruments.{output_format}'
        sf.write(inst_file, wave.T, sr)

        filetags = music_tag.load_file(file)
        inst_tags = music_tag.load_file(inst_file)
        inst_tags['tracktitle'] = filetags['tracktitle']
        inst_tags['album'] = filetags['album']
        inst_tags['artist'] = filetags['artist']
        inst_tags['tracknumber'] = filetags['tracknumber']
        inst_tags['totaltracks'] = filetags['totaltracks']
        inst_tags['year'] = filetags['year']
        inst_tags.save()

        if args.create_webm:
            vid_file = f'{output_folder}/{basename}.mp4'
            os.system(f'ffmpeg -y -framerate 1 -loop 1 -i "{cover}" -i "{inst_file}" -t {librosa.get_duration(wave, sr=args.sr)} "{vid_file}"')

        if args.create_vocals:
            print('\ninverse stft of vocals...', end=' ')
            wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
            print('done')
            sf.write(f'{output_folder}/{basename}_Vocals.{output_format}', wave.T, sr)

            vocal_tags = music_tag.load_file(f'{output_folder}/{basename}_Vocals.{output_format}')
            vocal_tags['tracktitle'] = f'{filetags["tracktitle"]} (Vocals)'
            vocal_tags['album'] = f'{filetags["album"]} (Vocals)'
            vocal_tags['artist'] = filetags['artist']
            vocal_tags['tracknumber'] = filetags['tracknumber']
            vocal_tags['totaltracks'] = filetags['totaltracks']
            vocal_tags['year'] = filetags['year']
            vocal_tags.save()

        if args.rename_dir:
            os.system(f'python song-renamer.py --dir "{output_folder}"')

if __name__ == '__main__':
    main()
