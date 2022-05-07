# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is a variation of tsurumeso's vocal remover that I've been tinkering with for a while that I'm calling a frame transformer. The goal of this fork is to find a meaningful way to use the evolved transformer architecture for track separation. This isn't a very user-friendly version just yet, has a lot of stuff that is specific to my dataset and environment; will take at least 12gb of VRAM to train this given the settings in the code. Will pretty this up over time. There is a broken autoregressive version in the wip folder, will probably tinker with that more eventually. cloud training example is in the cloud folder, but it won't be as easy as just copy/pasting; haven't had a chance to test it in this project so there is probably a small issue somewhere, will finish up cloud stuff in the coming days and make it more user friendly. DO NOT RUN USING THE A100S UNLESS YOU KNOW WHAT YOU ARE DOING, THEY ARE EXPENSIVE. Be sure you want to use them before you do; I suggest testing on more reasonable machines first.

This version consists of a single u-net; the u-net uses convolutions with Nx1 kernels with a stride of Mx1 to create embeddings for each frame of the audio data; no information is shared between frames in the encoding process. After encoding using the Nx1 convolutions, the decoding process begins. This makes use of a post-norm variant of the evolved transformer that has been modified to make use of the Primer architecture and relative positional encoding as seen in the music transformer; I call this architecture a frame transformer in the code. The transformer modules make use of a bottleneck for the modules input as well as a bottleneck for the skip connection for that layer. This allows each transformer block to query against the skip connections for global information using the multiband frame attention module. The multiband frame attention module is just a normal multihead attention module with relative positional encoding, however in the context of a spectrogram it makes more sense to call it multiband frame attention. I have compared the architecture with and without the transformer modules and the difference in validation loss was quite significant, so the strength appears to be in the meshing of the two architectures; replacing the convolutional portion with linear layers proved to yield poor results as well. This architecture does have larger checkpoints, however the new default settings recently checked in have smaller checkpoints than more recent versions that yield far lower validation loss. I'm open to PRs to this repo, the goal is to push this as far as I can with audio. I intend to make an autoregressive variant soon, however preliminary tests didn't really work too well so I have to do some more tinkering.

New version's default settings has a quadrupled cropsize; made sense to give a longer context window. In testing this very clearly helps and speeds up training quite significantly. As an example, this new version has surpassed my previous best that trained for something like 10 epochs and its only at the 3rd epoch. This seems to imply that the longer you can make the cropsize the more accurate it will be, which makes sense given the non-local nature of self attention. I'm starting to consider implementing a form of relative positional encoding that is capable to expand to new sequence lengths. As a test, I randomly injected noise into the attention matrix resulting in the neural network no longer successfully removing vocals. It does remove a little bit but not at audible levels. This coupled with the large increase in cropsize which was followed by a massive drop in validation loss around the 2nd epoch seems to point to the attention mechanism being critical in this architecture.

Edit: I'm leaving the checkpoint below around, however I have added significantly more data to my dataset and will be retraining it from scratch as it has some more examples of fretless bass which can throw my model off (although it throws basically every other model off way more since I already have fretless bass in my dataset, just adding more). The more recent version was the lowest first epoch validation loss, however it was also used 4 instead of 8 channels for the u-nets output while increasing feedforward dimensions in the transformer modules. This appears to be even more evidence that the transformer modules are lending quite a bit of power to this architecture. NOTE: you will have to rename the skip_attn modules to enc_attn (so enc_attn1 and enc_attn2 in FrameTransformerDecoder).

Edit 2: I got an even lower validation loss by compressing the convolutional channels even further and expanding the feedforward dimensions in the architecture, so I will likely be going this route as it seems most promising . I am going to be uploading a new version of a checkpoint soon with slight changes in architecture, however if you wish to use the below checkpoint you can work off this commit specifically: https://github.com/carperbr/vocal-remover-frame-transformer/commit/0f774208d7d0502133d6f624f4de9beb1d56f263

A checkpoint at the 11th epoch (though still handles metal better than any previous version I've tried) on my dataset can be found here; will be taking it to at least 20 so this is incomplete: https://mega.nz/file/7sxxyQQa#1fsEanHAN2fh8d692hIhEx7T4NtRGHxVjImH9GcwpIg. This uses the same settings as the default settings in inference.py, so you should be able to test it out that way.

There are currently two versions: a standard single pass network that works on chunks of the main spectrogram, and an autoregressive version that slides a window across the spectrogram to smoothly capture transitions. The autoregressive one is currently not working, clearly there is an issue in the code that I have to hunt down. Will hopefully be getting that updated soon.

This repo also contains a custom lr scheduler to warmup the learning rate gradually.

Will update this repo with checkpoints once I've converged on a more final architecture (which I believe I have now); I have an RTX 3090 Ti sitting in the corner so I should soon be able to train an even larger version that people can use for inference. My personal dataset consists of 5,249 instrumental songs from many genres, 488 vocal tracks from many genres, and 345 instrumental + mix pairs mainly from metal but with some rap and tracks from MUSDB18 all of which comes out to around 1TB. An example of this architecture after 4 epochs on my dataset is here: https://www.youtube.com/watch?v=bAJ_zUlUcAA, a vocal extraction (warning, the vocals here are screams) is here: https://www.youtube.com/watch?v=Wny0gBz_3Og with instrumental counterpart here: https://www.youtube.com/watch?v=jMVcX9RQCbg

This fork makes use of vocal augmentations. The dataset takes an instrumental path, an optional second instrumental path if you have a dataset split across multiple locations, a vocal path, and a pair path. The dataset's __getitem__ method will check to see if a Y value exists in the npz file; instrumental npzs will be expected to only have an 'X' and a 'c' key defined. If a 'Y' key does not exist, it will treat it as a file that should be augmented with vocals. This will cause the dataset to randomly select a vocal file, augment the vocal spectrogram, and then add the vocal spectrogram to the instrumental track and select the largest of either the instrumental max magnitude, vocal max magnitude, or the max magnitude of that slice. There is also a chance that vocals will be randomly added to pair tracks, so the vocal version will get an extra layer of vocals added to it and a new divisor selected if need be for normalization.

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] So et al., "The Evolved Transformer", https://arxiv.org/pdf/1901.11117.pdf
- [6] Huang et al., "Music Transformer: Generating Music with Long-Term Structure", https://arxiv.org/pdf/1809.04281.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/abs/2109.08668v2