# Style_Transfer
The idea of style transfer is based on the paper of Leon A. Gatys, et al (https://arxiv.org/abs/1508.06576) is aimed to capture a style of a painter and transfer it to a real image. In this repository I build the model from scratch based on pre-trained VggNet using TensorFlow.

Three nearal networks named vgg_c, vgg_s, vgg_t are builed based on VggNet layers:
  * vgg_c and vgg_s takes as input content and style images. These nets are not trainable
  * vgg_t takes as input random noise. Since its layers are trainbale, eventually it should be trained to output an image so that it containes a content      image decorates by a style image
Trianing algorithm is as follows, on every epoch:
  * error_c containt an error accomulated from difference between corresponding layers of the nets vgg_c and vgg_t
  * error_s containt an error accomulated from difference between corresponding layers of the nets vgg_s and vgg_t primerly converted by the Gram's matrix
  * total error is a liniear combination of error_c and error_s
  * then vgg_t weits are oprimized so to minimize the total error
