# Unsupervised Domain Adaptation with Hierarchical Masked Dual-adversarial Network for End-to-end Classification of Multisource Remote Sensing Data

Wen-Shuai Hu, Wei Li, Heng-Chao Li, Xudong Zhao, Mengmeng Zhang, and Ran Tao

Paper web page: [Unsupervised Domain Adaptation with Hierarchical Masked Dual-adversarial Network for End-to-end Classification of Multisource Remote Sensing Data](https://xplorestaging.ieee.org/document/10981764).
___________

# Abstract:
Although unsupervised domain adaptation (UDA) has been successfully applied for cross-scene classification of multisource remote sensing (MSRS) data, there are still some tough issues: 1) The vast majority of them are patch-based, requiring pixel by pixel processing at high complexity and ignoring the roles of unlabeled data between different domains. 2) Traditional masked autoencoder (MAE)-based methods lack effective multiscale analysis and require pre-training, ignoring the roles of low-level representations. As such, a hierarchical masked dual-adversarial DA network (HMDA-DANet) is proposed for cross-domain end-to-end classification of MSRS data. Firstly, a hierarchical asymmetric MAE (HAMAE) without pre-training is designed, containing a frequency dynamic large-scale convolutional (FDLConv) block to enhance important structural information in the frequency domain, and an intramodality enhancement and intermodality interaction (IAEIEI) block to embed some additional information beyond the domain distribution by expanding the cross-modal reconstruction space. Representative multimodal multiscale features can be extracted, while to some extent improving their generalization to the target domain. Then, a multimodal multiscale feature fusion (MMFF) block is built to model the spatial and scale dependencies for feature fusion and reduce the layer by layer transmission of redundancy or interference information. Finally, a dual-discriminator-based DA (DDA) block is designed for class-specific semantic feature and global structural alignments in both spatial and prediction spaces. It will enable HAMAE to model the cross-modal, cross-scale, and cross-domain associations, yielding more representative domain-invariant multimodal fusion features. Extensive experiments on five cross-domain MSRS datasets verify the superiority of the proposed HMDA-DANet over other state-of-the-art methods.

Citation
---------------------
**Please kindly cite the papers if this code is useful and helpful for your research.**

Wen-Shuai Hu, Wei Li, Heng-Chao Li, Xudong Zhao, Mengmeng Zhang, and Ran Tao, "Unsupervised Domain Adaptation With Hierarchical Masked Dual-Adversarial Network for End-to-End Classification of Multisource Remote Sensing Data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1-17, 2025, Art no. 4409917, doi: 10.1109/TGRS.2025.3566305. <br>

@ARTICLE{10981764,
  author={Hu, Wen-Shuai and Li, Wei and Li, Heng-Chao and Zhao, Xudong and Zhang, Mengmeng and Tao, Ran},<br>
  journal={IEEE Transactions on Geoscience and Remote Sensing}, <br>
  title={Unsupervised Domain Adaptation With Hierarchical Masked Dual-Adversarial Network for End-to-End Classification of Multisource Remote Sensing Data}, <br>
  year={2025},<br>
  volume={63},<br>
  number={},<br>
  pages={1-17},<br>
  keywords={Feature extraction;Adaptation models;Semantics;Laser radar;Frequency-domain analysis;Decoding;Data mining;Complexity theory;Autoencoders;Accuracy;Cross-scene;end-to-end classification;frequency attention;masked autoencoder (MAE);multiadversarial learning;multimodal masking learning;multisource remote sensing (MSRS) data},<br>
  doi={10.1109/TGRS.2025.3566305}}


# Requirements

CUDA Version: 11.8 <br>
Pytorch Version: 2.0 <br>
Python Version: 3.10 <br>

# Note
System-specific notes
---------------------
Please refer to the file `requirements.txt` for the running environment of this code.

Contact Information:
--------------------

Wen-Shuai Hu: wshuswjtu@163.com<br>
Wen-Shuai Hu is currently a Post-Doctoral Researcher of the School of Information and Electronics with Beijing Institute of Technology, Beijing, China. 
