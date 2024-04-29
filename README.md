# FrameTransformer / AsymUNetMCT

The YT channel was terminated due to a coordinated barrage of copyright strikes from Sony. Discord community is located here: https://discord.gg/8V5t9ZXRqS

Update on project: I have updated the project with a new neural network inspired by hierarchical transformers and my older frame transformer. The new architecture is a temporal u-net rather than a frequency u-net. This has the added benefit of condensing the sequence length akin to hierarchical transformers which allows for more processing power in the transformer. This appears to capture temporal dynamics quite well as well, and with an updated data synthesis technique it appears the quality has surpassed the frequency u-net despite only being at 280k optimization steps thus far. V9 was a bit of a failed experiment though it did teach me some new stuff, specifically that the data synthesis changes were indeed beneficial and that the patch transformer setup isn't very great at capturing temporal dynamics in audio given that you are breaking apart the true temporal dynamics in that process. V10 will continue training until 2 million optimization steps, so it has quite a bit to go until fully trained which is a very promising sign given that it surpassed v9 when v9 was at 1.5mil optimization steps while it was only at 280k optimization steps.

You can find the new temporal u-net transformer in inference/v10/frame_transformer13.py

I will be working on a react app and backend moving forward for inferencing that will allow users to convert entire songs with ensembles and then reconvert sections with specific models to perfect instrumentals with a human in the loop to an extent.

10/7/23: Updated snapshots folder with a new version which is smaller but high quality. Should be more accessible for people.

Snapshots folder has been updated with a new version. I have a third new version finishing up now that is doing even better, and a fourth model training on my second workstation which is doing even better than these. All checkpoints are found in the inference.py file in the corresponding snapshot folder, left as a comment with a mega.nz link that leads to a model.pth. If anyone has any issues with the checkpoints please let me know.

Installation tips from the community with thanks to HunterThompson, PsychoticFrog, mesk and others for helping converge on these, will clean up requirements.txt soon:
https://discord.com/channels/1143212618006405170/1143212618492936292/1148050453645508658

TypeError: init() got an unexpected keyword argument 'dtype'
pytorch is too old, update to 2.0.0+cu118

ModuleNotFoundError: No module named 'einops'
pip install einops

module 'numpy' has no attribute 'complex'.
pip install numpy==1.23.5

librosa 0.8.1 works
librosa 0.9.2 works, but gave me warnings

edit inference.py like this for model version in filenames
https://cdn.discordapp.com/attachments/531122660244062236/1157139846947672115/image.png?ex=651ad1b6&is=65198036&hm=f529e7dcbcb268afa02f41937486ca86e9fa85cd4d0c571cf6e0b0b7d294466e&

weird python issues? Uninstall python, reinstall python and reboot. This wipes all your pip modules so you have to install pytorch, numpy 1.23.5 and einops again

Right now all snapshots use the old method of librosa's STFT for creating spectrograms. I have a new version training now using the combined linear and mel scale L1 loss terms as well as using PyTorch's STFT. This will allow for a pretty large speedup during inferencing in the next version, should have it finished this weekend.

Two new versions are currently training, both of which are outperforming current snapshots (I have another snapshot I'll upload before these two in the coming days as well). Most versions have been using the old method of preprocessed spectrograms as I haven't been training on my main machine which has my regenerated dataset of just waveforms and utilizes TorchAudio to convert to spectrograms on the GPU. The newest version I'm training includes a loss term for reconstruction of the spectrogram in the Mel scale as well to encourage more overlap with human perception but I'm not really sure if that will work. It's performing quite well but who knows, might not make a big difference.

I'm shifting work into the app folder for docker related reasons, other folders will no longer be updated and will eventually be deleted. The app folder contains scripts that are able to run using DistributedDataParallel, the dockerfile for this is located in the root directory with a build.sh script to build it for convenience. You can run these directly without docker, however.

I have converged on an architecture that I am happy with after a great deal of experimentation, architecture is not likely to change much from this point on. The architecture in general I'm referring to as an AsymUNetMCT, or **Asym**metrically down-sampled **UNet** **M**ulti-**C**hannel **T**ransformer. It consists of a residual u-net that downsamples using a stride of (2,1) coupled with a multichannel transformer encoder and decoder after each u-net encoder and decoder respectively. Multichannel transformers are an adaptation of transformers that I created for dealing with audio. Multichannel transformers utilize multichannel linear layers which are layers I created in order to efficiently utilize parallel position-wise linear layers. Instead of the typical approach I had found online of using grouped 1d convolutions, I opted to use batched matrix multiplication. This allows for far more efficient parallel linear layers that scale far more effectively. Multichannel transformers simply utilize this mulitchannel linear layer as well as multichannel layer normalization; in the case of this project, the frames of each channel (H dimension) constitute the 'embedding' vectors of the 'tokens' for the transformer. As in MMDENSELSTM, a representation is extracted from the full feature map volume however for this architecture it extracts a set of feature maps which are then processed with the multichannel transformer layers. In testing, I find that having more channels is beneficial to a degree but at a certain point more attention maps seem to be needed for that to really help. As an exmaple, setting the number of attention maps to 1 and the number of channels to 50 produces a neural network that is outperformed by one that uses 32 channels and 4 attention maps. However, a frame transformer with 8 channels and 16 attention maps seems to do worse than the others, so there appears to be a degree of balancing that is required. So far, it appears that scaling helps quite a bit as is usually the case with transformers. Training (and inferencing within reason) on longer sequence lengths is also incredibly beneficial to validation loss, and quality increases quite a bit when that is done. I intend on training current checkpoints on longer sequence lengths for a period of time, typically it only requires a short finetuning period rather than training for a large number of epochs.

I have created a snapshots folder which I'm beginning to fill up with pretrained models I've had thus far. I will be storing code snapshots within that folder with links to the checkpoints on mega.nz. I have one snapshot in the folder which is the first GAN variant that has been trained on a cropsize of 256 using the below training approach. It only uses 1 attention map per transformer layer, however I am currently pretraining a model with more attention maps and more parameters. The GAN in the snapshots folder did not have any pretraining and was trained on magnitude only by generating a multiplicative mask. New networks are instead treating the real and imaginary parts of the spectrograms as separate channels and predicting the complex spectrogram directly which is part of why the supervised training stage was added.

**ConvolutionalEmbedding** - This module was inspired by [11] below. This makes use of an encoding branch that consists of residual blocks with symmetrical stride, as well as residual blocks for extracting positional information from that scale. The down-sampling blocks utilize a kernel size of 3 while the position extraction layers utilize kernel sizes of 11, as more padding is shown to help with positional information in the paper. This also includes sinusoidal positional encoding in the form of an extra channel that is appended to the input before being down-sampled.

**FrameConv** - This module is focused on being a layer that utilizes locality information along with the fully connected parallel linear layers in order to contextualize the fully connected features. It makes use of a symmetrical kernel convolution followed by batched-matrix multiplication to carry out parallel linear layers on the GPU in an efficient manner.

**MultichannelLinear** - This module is focused on being a layer that processes parallel linear layers and then shares information between those layers. It makes use of batched-matrix multiplication to process parallel linear layers on the GPU and then utilizes matrix multiplication again to carry out a depth-wise transformation that is equivalent to a 1x1 convolution in order to share information between frames and to modify number of channels. This module is similar to the FrameConv module, however the depth-wise transform occurs in the form of matrix multiplication after the parallel position-wise transform and thus is equivalent to a 1x1 convolution - its purpose is not necessarily to encode spatial information as in the FrameConv module.

**MultichannelLayerNorm** - This module applies LayerNorm in parallel across multiple channels, allowing each channel to learn its own element-wise affine parameters.

**ChannelNorm** - This module applies LayerNorm across the channel dimension, module is included for ease of use.

**ResBlock** - This is a res block that is more inspired by the pre-norm transformer architecture than resnet. It uses less computational resources while performing similarly which allowed me to scale the transformer portion of the network a bit more. This utilizes MultichannelLayerNorm for normalization.

**FrameEncoder** - This is the encoder for the asymmetrical u-net; it consists of a single residual block with a symmetrical kernel but asymmetrical stride when downsampling so as to preserve the temporal dimension.

**FrameDecoder** - This is the decoder for the asymmetrical u-net; it consists of a single residual block that operates on the bilinearly interpolated input with the skip connection concatenated to the channel dimension.

**ConvolutionalMultiheadAttention** - This module is a variant of multihead attention that I created to use within the latent space of the U-Net. This is similar to other approaches that are in use elsewhere, I opted to include a symmetrical kernel size > 1 and use multi-head attention. This includes an optional residual attention connection as well.

**MultichannelMultiheadAttention** - This module was created to utilize parallel attention mechanisms. It uses rotary positional embedding as seen in the RoFormer architecture to better contextualize positional information. After this, the query, key, and value projections utilize MultichannelLinear layers with no depth-wise component which is then followed by 1xN kernel convolutions that utilize only a single group so as to share information across the attention mechanisms. This includes a residual attention connection and is ended with a final output projection that utilizes a multichannel linear layer.

**FrameTransformerEncoder** - Frame transformer encoders work by optionally extracting a set of feature maps from the overall channel volume (similar manner to MMDENSELSTM when it extracts a single channel) and then computing attention between the frames of the extracted channels in parallel. The architecture for this transformer encoder is modeled off of a few transformer variants. The main structure is taken from the Evolved Transformer and then extended into the channel dimension using the above multichannel layers and FrameConv, while also making use of changes from the Primer architecture, the RealFormer architecture, and the RoFormer architecture. Where the Evolved Transformer utilizes depth-wise separable 1d convolutions, this architecture utilizes the above FrameConv modules. This is similar to how the separable convolutions work in the original Evolved Transformer paper, however instead of the frequency dimension being fully connected it too uses locality followed by the fully connected parallel linear layers. This concatenates the output to the input, returns the output separately for use later in the network, and returns a residual attention connection for use in the next layer as well as later on in the network.

**FrameTransformerDecoder** - Frame transformer decoders work by optionally extracting a set of feature maps from the overall channel volume (similar manner to MMDENSELSTM when it extracts a single channel) and then computeing attention between the frames of the extracted channels in parallel. As above, this is based on a series of transformer architectures that have been adapted for multichannel audio data. The main difference between this and the encoder is that this follows the Evolved Transformer Decoder architecture. For memory, this module utilizes the returned attention maps from the corresponding FrameTransformerEncoder from that level. For the residual attention connection for skip attention, there are two residual connections. To deal with this, I introduced a gating mechanism which uses 3d convolutions to process the attention scores from the residual attention connections and outputs a weight for each frequency that allows the network to choose which source to use information from.

**ConvolutionalTransformerEncoder** - This is a transformer encoder that is uses the channel dimension as the embedding dimensions. This too utilizes the evolved transformer architecture, however it has been adapted for use with multichannel data. This utilizes ChannelNorm to normalize features across the channel dimension so as to contextualize the channels of each feature and allow to compare them with one another in a more sensible manner. This makes use of ConvolutionalMultiheadAttention.

### These will be moved into the snapshots folder shortly with the expedcted code.

V1 - This neural network has been trained to about 1.2 million optimization steps using progressively larger context lengths throughout training. It makes use of rotary positional embeddings only, and consists of only a single attention map per u-net layer. It uses far less vram than the other variants, however. I need to update the script to play nicely with this checkpoint, will fix that shortly but shouldn't be too hard for people if they are familiar with PyTorch. https://mega.nz/file/C5pGXYYR#ndHuj-tYWtttoj8Y4QqAkAruvZDYcQONkTHfZoOyFaQ

V2b - This neural network is similar to the new frame transformer, however it uses the conventional transformer architecture and does not include a convolutional transformer branch. https://mega.nz/file/a843wbTQ#E319mlp5qjsyky6PRp-zikou-YtWb9TVpbtHGaVh2AA

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
- [5] Vaswani et al., "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
- [6] So et al., "Primer: Searching for Efficient Transformers for Language Modeling", https://arxiv.org/pdf/2109.08668v2.pdf
- [7] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", https://arxiv.org/abs/2104.09864
- [9] Asiedu et all., "Decoder Denoising Pretraining for Semantic Segmentation", https://arxiv.org/abs/2205.11423
- [10] He et al., "RealFormer: Transformer Likes Residual Attention", https://arxiv.org/abs/2012.11747
- [11] Islam et al., "How Much Position Information Do Convolutional Neural Networks Encode?", https://arxiv.org/abs/2001.08248
- [12] Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", https://arxiv.org/pdf/1611.07004.pdf
