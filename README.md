# UDA_Multisource_Remote_Sensing_HMDA-DANet
Unsupervised Domain Adaptation with Hierarchical Masked Dual-adversarial Network for End-to-end Classification of Multisource Remote Sensing Data

Paper web page: [Unsupervised Domain Adaptation with Hierarchical Masked Dual-adversarial Network for End-to-end Classification of Multisource Remote Sensing Data](https://xplorestaging.ieee.org/document/10981764).

Related code will be released gradually.

Abstract:
Although unsupervised domain adaptation (UDA) has been successfully applied for cross-scene classification of multisource remote sensing (MSRS) data, there are still some tough issues: 1) The vast majority of them are patch-based, requiring pixel by pixel processing at high complexity and ignoring the roles of unlabeled data between different domains. 2) Traditional masked autoencoder (MAE)-based methods lack effective multiscale analysis and require pre-training, ignoring the roles of low-level representations. As such, a hierarchical masked dual-adversarial DA network (HMDA-DANet) is proposed for cross-domain end-to-end classification of MSRS data. Firstly, a hierarchical asymmetric MAE (HAMAE) without pre-training is designed, containing a frequency dynamic large-scale convolutional (FDLConv) block to enhance important structural information in the frequency domain, and an intramodality enhancement and intermodality interaction (IAEIEI) block to embed some additional information beyond the domain distribution by expanding the cross-modal reconstruction space. Representative multimodal multiscale features can be extracted, while to some extent improving their generalization to the target domain. Then, a multimodal multiscale feature fusion (MMFF) block is built to model the spatial and scale dependencies for feature fusion and reduce the layer by layer transmission of redundancy or interference information. Finally, a dual-discriminator-based DA (DDA) block is designed for class-specific semantic feature and global structural alignments in both spatial and prediction spaces. It will enable HAMAE to model the cross-modal, cross-scale, and cross-domain associations, yielding more representative domain-invariant multimodal fusion features. Extensive experiments on five cross-domain MSRS datasets verify the superiority of the proposed HMDA-DANet over other state-of-the-art methods.

Paper
Please cite our paper if you find the code or dataset useful for your research.

@ARTICLE{10981764,
  author={Hu, Wen-Shuai and Li, Wei and Li, Heng-Chao and Zhao, Xudong and Zhang, Mengmeng and Tao, Ran},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Unsupervised Domain Adaptation with Hierarchical Masked Dual-adversarial Network for End-to-end Classification of Multisource Remote Sensing Data}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Adaptation models;Semantics;Laser radar;Frequency-domain analysis;Decoding;Data mining;Complexity theory;Autoencoders;Accuracy;Cross scene;multisource remote sensing data;end-to-end classification;multimodal masking learning;masked autoencoder;multi-adversarial learning;frequency attention},
  doi={10.1109/TGRS.2025.3566305}}

Requirements
Pytorch 2.0.0 
python 3.10.12

Note
