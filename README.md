This repository contains the implementation of ABC-CapsNet (https://openaccess.thecvf.com/content/CVPR2024W/WiCV/papers/Wani_ABC-CapsNet_Attention_based_Cascaded_Capsule_Network_for_Audio_Deepfake_Detection_CVPRW_2024_paper.pdf), a novel architecture designed to detect audio deepfakes using Mel spectrograms, VGG18 for feature extraction, an attention mechanism, and Cascaded Capsule Networks (CapsNet). The architecture has been validated on the ASVspoof 2019 and Fake or Real (FoR) datasets, achieving state-of-the-art results in audio deepfake detection.
Introduction
ABC-CapsNet (Attention-Based Cascaded Capsule Network) is an advanced model designed to detect sophisticated audio deepfakes. By utilizing a combination of Mel spectrograms, a pre-trained VGG18 model for feature extraction, an attention mechanism to focus on key features, and cascaded capsule networks, ABC-CapsNet achieves high precision in distinguishing between real and fake audio samples.

The paper introduces a method that addresses the limitations of traditional CNN-based models by employing Capsule Networks (CapsNet), which are better at preserving spatial hierarchies and identifying subtle deformations. The cascaded architecture, where Capsule Network 1 (CN1) focuses on primary feature extraction and Capsule Network 2 (CN2) refines these features further, allows for a robust audio classification pipeline.

Model Architecture
Mel Spectrograms
The input audio is transformed into Mel spectrograms using a standard resampling rate of 16kHz. Mel spectrograms represent the time-frequency content of the audio signal, making it easier to analyze key features like timbre and pitch.

VGG18 Feature Extraction
A pre-trained VGG18 model is used to extract high-level features from the Mel spectrograms. VGG18 is widely used in image classification tasks and proves effective for feature extraction when treating Mel spectrograms as 2D images.

Attention Mechanism
An Attention Mechanism is applied after feature extraction to prioritize the most relevant features, improving the focus on specific parts of the audio signal that might indicate a deepfake.

Cascaded Capsule Networks
The cascaded capsule networks (CapsNet) consist of two networks:

Capsule Network 1 (CN1): Extracts primary, lower-level features and maintains spatial hierarchies.
Capsule Network 2 (CN2): Further refines and processes the extracted features to identify more complex patterns typical of deepfakes.
Both capsule networks use dynamic routing to pass important information between capsules.

