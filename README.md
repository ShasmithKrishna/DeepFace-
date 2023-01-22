# DeepFace Recognition
Name - Akkapally Shasmith Krishna <br>
Roll No- 210050008

# Deep Learning
Deep learning is a branch of machine learning that is concerned with the development of artificial neural networks that are capable of learning from data. The key idea behind deep learning is to use multiple layers of artificial neurons, known as a deep neural network, to learn increasingly abstract representations of the data.
Applications of Deep learning include image recognition, speech recognition, natural language processing, and game playing. The success of deep learning is largely due to the ability of deep neural networks to learn hierarchical representations of data.

An innovation of Deep learning is the use of convolutional neural networks (CNNs) for image recognition. CNNs are a type of deep neural network that are designed to take advantage of the 2D structure of images. They have been shown to be highly effective at recognizing objects in images, and they are now widely used in computer vision tasks such as object detection and semantic segmentation.

Another innovation of Deep learning is the use of recurrent neural networks (RNNs) for sequential data such as speech and text. RNNs are a type of deep neural network that are designed to handle sequential data by maintaining an internal state that can be used to remember information from previous time steps. This makes RNNs well-suited for tasks such as speech recognition, natural language processing and machine translation.

Deep learning has been widely adopted in industry and has led to many practical applications such as self-driving cars, virtual assistants, and intelligent video surveillance systems

# Convoultion Neural Networks
Convolutional neural networks (CNNs) are a type of deep learning architecture that are designed to take advantage of the 2D structure of images. They are composed of multiple layers, with each layer consisting of a set of filters that are used to detect specific features in the input image.

The key idea behind CNNs is the use of convolutional layers, which apply a set of filters to the input image in a way that preserves the spatial relationship between pixels. This allows the network to learn increasingly complex features of the image, starting with simple edges and textures in the lower layers, and progressing to more abstract concepts in the higher layers.

CNNs also include pooling layers, which are used to reduce the spatial dimensions of the feature maps produced by the convolutional layers. This helps to reduce the computational requirements of the network and also helps to make the network more robust to small translations of the input image.
CNNs have been shown to be highly effective at recognizing objects in images. They have also been used to improve the performance of other computer vision tasks, such as object detection and semantic segmentation.

Below are some breif summaries of some papers on Neural Networks:

# ResNet
This introduced a new architecture for convolutional neural networks called Residual Networks (ResNets). The main idea behind ResNets is the use of "residual connections," which are connections between the input and output of a layer that bypass one or more layers(adding the residual function to the raw output itself instead of trying to leanr itself). This allows the network to learn residual functions(such as identity function), or the differences between the desired output and the output of the current network. This architecture allows for much deeper networks (up to 152 layers) to be trained without the problem of vanishing gradients.
It also demonstrate that ResNets can be used to improve the performance of other computer vision tasks, such as object detection and semantic segmentation.

# FaceNet
This introduced a new method for face recognition using a neural network architecture called FaceNet. 
This propose a method for learning a "face embedding"(neural network) that maps an image of a face to a high-dimensional feature vector in a way that preserves the similarity between faces. The embedding is learned using a triplet loss function, which encourages the network to produce similar embeddings for faces of the same person and dissimilar embeddings for faces of different people.

They evaluated the performance of the FaceNet model on a benchmark dataset of faces, and show that it outperforms state-of-the-art methods for face recognition, achieving a lower error rate than previous methods. They also show that the FaceNet embedding can be used to perform face clustering, which is the task of grouping faces of the same person, even if they have different poses and lighting conditions.

One of the key contributions of the FaceNet paper is the use of the triplet loss function, which has since been widely adopted in other face recognition systems. The paper is also notable for showing that a deep CNN can be trained to perform face recognition using a relatively small dataset (around 200,000 images), which has important implications for real-world applications where large amounts of labeled data may not be available

# InceptionNet
The key innovation of InceptionNet is the use of "Inception modules", which are building blocks that can be stacked to form a deep network. The Inception module is designed to be computationally efficient while also allowing the network to learn a wide range of feature representations at different scales.

The authors evaluate the performance of the InceptionNet model on the ImageNet dataset, which is a large dataset of images with 1000 different classes. They show that InceptionNet outperforms state-of-the-art CNN architectures and achieves a top-5 error rate of 6.67% on the test set, which is a significant improvement over previous methods.
The InceptionNet paper also introduced a few other important innovations, such as the use of global average pooling, which replaces the traditional fully connected layers and reduces the number of parameters, and the use of a technique called "batch normalization", which helps to improve the stability of the training process.

InceptionNet has been widely adopted in the field of computer vision, and it has served as the basis for many other CNN architectures, such as Google's MobileNet and Inception-ResNet. The Inception module is a popular building block in many computer vision architectures such as deep learning models for object detection and semantic segmentation.

# MobileNetv2
MobileNetV2 uses depth-wise separable convolutions, which are a type of convolution that reduces the number of computations required to process an image. This makes the network more computationally efficient, which is important for running deep learning models on mobile devices with limited resources.

The authors introduce a new building block called "inverted residuals with linear bottlenecks", which allows to increase the resolution and number of channels of the feature maps while keeping the computational cost constant. This increases the accuracy of the network.

MobileNetV2 has been widely adopted in the field of mobile and embedded devices, and it has served as the basis for many other efficient CNN architectures, such as MobileNetV3 and EfficientNet.

# DeepFace
Proposed a new neural network architecture called DeepFace, which was designed to improve the performance of face recognition tasks. The architecture is similar to the architecture of AlexNet, but it uses a more complex pipeline of preprocessing and data augmentation to improve the performance of the network.
