# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is a variation of tsurumeso's vocal remover that I've been tinkering with for a while that I'm calling a frame transformer. The goal of this fork is to find a meaningful way to use the evolved transformer architecture for track separation. This isn't a very user-friendly version just yet, has a lot of stuff that is specific to my dataset and environment; will take at least 12gb of VRAM to train this given the settings in the code. Will pretty this up over time.

This version consists of a single u-net; the u-net uses convolutions with Nx1 kernels with a stride of Mx1 to create embeddings for each frame of the audio data; no information is shared between frames in the encoding process. After encoding using the Nx1 convolutions, the decoding process begins. This makes use of a post-norm variant of the evolved transformer that has been modified to make use of the Primer architecture and relative positional encoding as seen in the music transformer; I call this architecture a frame transformer in the code. The transformer modules make use of a bottleneck for the modules input as well as a bottleneck for the skip connection for that layer. This allows each transformer block to query against the skip connections for global information using the multiband frame attention module. The multiband frame attention module is just a normal multihead attention module with relative positional encoding, however in the context of a spectrogram it makes more sense to call it multiband frame attention. I have compared the architecture with and without the transformer modules and the difference in validation loss was quite significant, so the strength appears to be in the meshing of the two architectures; replacing the convolutional portion with linear layers proved to yield poor results as well. This architecture does have larger checkpoints, however the new default settings recently checked in have smaller checkpoints than more recent versions that yield far lower validation loss. I'm open to PRs to this repo, the goal is to push this as far as I can with audio. I intend to make an autoregressive variant soon, however preliminary tests didn't really work too well so I have to do some more tinkering. 

The attention is what really seems to make this architecture work. As a test, if you randomly insert noise into the attention matrix the neural network will no longer successfully remove vocals fully. It does remove a little bit, but listening to the mix its not noticable levels. This seems to point to the attention being critical in making the frame transformer work, however the convolutional layers DO contribute as well (as would be expected). It currently seems that a hybrid between the Primer architecture and the Evolved Transformer yields the best results.

There are currently two versions: a standard single pass network that works on chunks of the main spectrogram, and an autoregressive version that slides a window across the spectrogram to smoothly capture transitions. The autoregressive one is currently not working, clearly there is an issue in the code that I have to hunt down. Will hopefully be getting that updated soon.

Note on checkpoints: I think the current repo is fairly final, currently training the frame transformer on my dataset and validation loss is actually dropping faster and faster with each epoch, so that's pretty exciting lol. Will let this model train until tomorrow and will upload it for people to start toying around with; building a new computer for a local compute cluster and after that will use model paralleism to expand this model quite a bit.

This repo also contains a custom lr scheduler to warmup the learning rate gradually.

I will likely restructure this repo soon and include my Google Cloud dataloader and training script, dockerfile etc etc. My current cloud version streams the data from google cloud storage so is meant to be used with a cluster via distributed data parallel.

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