# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

This is a variation of tsurumeso's vocal remover that I've been tinkering with for a while that I'm calling a frame transformer. The goal of this fork is to find a meaningful way to use the evolved transformer architecture for track separation. This isn't a very user-friendly version just yet, has a lot of stuff that is specific to my dataset and environment; will take at least 12gb of VRAM to train this given the settings in the code. Will pretty this up over time.

This version consists of only a single u-net. This u-net includes modified evolved transformer encoders after each downsampling as well as modified evolved transformer decoders before each upsampling.

Will update this repo in the coming days with checkpoints with listed settings. My personal dataset consists of 5,249 instrumental songs from many genres, 488 vocal tracks from many genres, and 345 instrumental + mix pairs mainly from metal but with some rap and tracks from MUSDB18 all of which comes out to around 1TB. An example of this architecture after 4 epochs on my dataset is here: https://www.youtube.com/watch?v=bAJ_zUlUcAA, a vocal extraction (warning, the vocals here are screams) is here: https://www.youtube.com/watch?v=Wny0gBz_3Og with instrumental counterpart here: https://www.youtube.com/watch?v=jMVcX9RQCbg

The main difference with this architecture is that it makes use of frame convolutions in order to only encode features from the same frame without respect to time. Each encoder is followed by a sequence of frame transformer encoders, and every decoder is preceded by a sequence of frame transformer decoders; frame transformer decoders utilize the skip connection fed through a bottleneck as memory for the encoder attention. Each frame transformer encoders output is concatentated with the input and fed into subsequent layers in a dense-net style. There is one final set of frame transformer decoders at the end; these decoders are responsible for decoding into the output left and right channels which are then sent through a sigmoid to produce the mask. All frame convolutions use 3x1 kernels so as to avoid transferring information between frames outside of the transformer blocks. Downsampling occurs only on the frequency axis; the temporal axis is preserved and remains set to cropsize for the entirety of the network. The frame transformer decoders are stacked and used at every resolution unlike the original LSTM implementation due to transformers typically requiring more layers and for the fact that all cross-frame communication occurs in the transformer modules in this variant. Due to memory I tend to use 4 per decoding step.

Current architecture is testing out usage of residual attention layers within decoding steps as described in https://arxiv.org/pdf/2012.11747.pdf. Each attention module, self and encoder attention, has its own residual connection between attention layers. 

Currently working on a branch for using TPUs and after that will modify it for GPUs. Haven't been able to test the TPU version because my jobs keep failing after starting (and billing me) with the error that no TPU resources are available (feels a little scammy...). GPU version is working well with DistributedDataParallel, will get that added in another branch after work.

This fork also makes use of vocal augmentations. This takes a random vocal spectrogram and adds it to an isntrumental spectrogram for training (and sometimes a mix for another layer of vocals); the dataset takes a path to a directory with vocal spectrogram npz files (need to update the dataset generator code still, the new vocal augmentation dataset expects unnormalied magnitudes with the normalization coefficient set in its own property in the npz file; old dataset class is still in dataset py). Mixup is also applied to vocals in the dataset to mix various vocals together.

A test has been carried out between the original stacked u-net implementation that was modified to use frame convolution encoders and this to verify that the boost in performance was not from the frame convolutions and rather from the transformer architecture. The 3x1 encoders used along with the vanilla architecture causes training time to literally double from 4 hours to 8 hours on my dataset while also causing the loss to be significantly higher. I need to do a test between the original architecture with its 3x3 encoders and this, however vocal extractions seem to be subjectively higher quality.

Will rewrite the rest of the readme for this specific fork soon.

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] So, Liang, and Le, "The Evolved Transformer", https://arxiv.org/pdf/1901.11117.pdf
