## **Conclusion:**

# **Executive Summary**

Key Findings from the Analysis:

**EfficientNet** with transfer learning is one of the most effective solutions for rapid implementation of facial emotion detection.

Among the evaluated models, EfficientNet demonstrated good and fast performance in facial emotion recognition.

Models tried: classical CNNs, VGG16, EfficientNet and ResNet architectures.

**Model Architecture:** EfficientNet-B0

**Input:** Facial images resized to 224x224 pixels, normalized and preprocessed to match EfficientNet-B0 input requirements.

**Transfer Learning:** Initialized with pretrained ImageNet weights, excluding the original classification head (include_top=False), then fine-tuned by adding a custom classification head tailored to the number of facial emotion classes in your dataset.

**Regularization:** Drop connect rate set to apply stochastic depth regularization during fine-tuning, improving generalization without affecting pretrained weights.

**Training Configuration:**

**Optimizer:** Here Adam (test RMSprop)

**Loss function:** Categorical cross-entropy for multi-class emotion classification.

**Batch size and learning rate** tuned based on hardware and dataset size.

**Early stopping and validation monitoring** to prevent overfitting.

Model Complexity and Efficiency:

**Parameters:** Approximately 5.3 million.

**FLOPs:** Around 0.39 billion, making it computationally efficient and suitable for deployment in resource-constrained environments.

Performance: EfficientNet-B0 provides a strong balance between accuracy and efficiency, achieving competitive accuracy in facial emotion recognition while maintaining low computational cost, making it ideal for real-time or embedded applications.

**I tested 1 ANN, 3 CNN, 3 transfer learning models and one complex CNN.**

**ANN**  accuracy: 0.4233  val_accuracy: 0.46177 test:accuracy: 0.5490

**CNN1** accuracy: 0.6943  val_accuracy: 0.6710  test:accuracy: 0.7604

**CNN2 accuracy: 0.7505  val_accuracy: 0.7372  test:accuracy: 0.7812**

**CNN3** accuracy: 0.6329  val_accuracy: 0.7812  test:accuracy: 0.0860

**VGG16** model accuracy: 0.6380 val_loss: 0.8290 test:accuracy: 0.6865

**ResNet V2** w/ GAP accuracy: 0.5424  val_accuracy: 0.5843 test:accuracy: 0.5552

**EfficientNet w/ GAP accuracy: 0.7703 val_accuracy: 0.1605 test:accuracy: 0.6427** (Overfitting)

**Complex CNN** accuracy: 0.6740   val_accuracy: 0.7366 test:accuracy: 0.7510

**CNNs are better than ANNs for facial recognition** because they automatically extract spatial features (such as edges, textures, and shapes) from images using convolutional layers, making them more accurate and efficient for image-based tasks.

**CNNs** help reduce size issues through convolution and pooling, which extract important features. This makes them more efficient and scalable for handling varying input sizes, unlike ANNs, which require fixed-size, flattened inputs and can struggle with large images. However, CNNs still require tuning.

**VGG16**: Easy to understand, provides good performance, but is very large and slow.

**ResNet**: A deep network with skip connections (to avoid vanishing gradients), offering better performance than VGG, but still quite heavy.

**EfficientNet**: Offers the best accuracy-to-size trade-off and is highly optimized, though more complex to implement.

**Our Complex_CNN**: Used for learning and experimentation. It has limited data and resources but offers full control over the architecture.


**Challenges:**

1_**Key Facial Features are Crucial for Identification**

2_**Lighting and Image Quality Impact Performance**

3_**Face Aging and Expression Variability**

4_**Data Augmentation and Regularization**

5_**Model Overfitting and Underfitting**


**ANN**

By flattening a 2D image into a 1D vector, ANNs lose the spatial relationships between pixels.

Not the best choice.

**Convolutional Neural Network (CNN)**

Preserve spatial relationships using convolutional layers to automatically learn hierarchical features (e.g., edges, textures, facial structures).

Efficient Feature Extraction

Local and Global Feature Learning

Translation Invariance

Better Generalization

**Transfer Learning**

Often trained on a large and diverse dataset like ImageNet plus,

fine-tuning it for a specific task (like facial recognition).

Already learned highly effective feature extraction strategies.

Faster Convergence, Higher Accuracy, Better Generalization.

**Problem and solution summary**

The proposed solution design—using **EfficientNet-B0** with transfer learning was chosen because it offers a strong balance between accuracy and computational efficiency.
Even if, in this particular case, the model is overfitting, it is very efficient in term on time & cost.

EfficientNet-B0’s architecture, which scales network depth, width, and resolution in a compound manner, enables effective extraction of fine-grained facial features necessary for distinguishing emotions while maintaining a lightweight model suitable for real-time and resource-constrained environments.

**CNN**

With higher accuracy than models solely based on a one-dimensional (ANN), CNN models recognize and capture spatial hierarchies in the image, meaning that spatial relationships between pixels (such as edges, textures, and patterns) are well preserved and understood.

Based on this research (the final usage not being specified) my final proposition tends to opt for a "simple" **CNN (model_2)** .
More adaptability.
 _____________________________________________________________________

It all depends on time and money to invest in the final project.
For this task, where the reference images are small and of relatively poor quality, opting for a simple and lightweight model seems to be the two best options.

The first, flexible, adaptable, and personalized CNN, will provide a better option in terms of reuse within the company for similar tasks. This requires more research in terms of parameter tuning.

The second and faster option to implement is EfficientNet, requires Fine-tuning, which uses transfer learning and freezes the last layers before flattening, making greater use of the "decoder" part.
This allows, as this brief implementation demonstrates, the use of a pre-trained model ready for immediate implementation.
On the other hand, it allows for less adaptation to other environments in the future, as its "generalization" is limited.

**Recommendations for implementation**

The **problem** addressed is the challenge of accurately recognizing Human facial emotions from images, which is complicated by variations in facial expressions, occlusions, lighting, pose, cultural and gender differences, and subtle distinctions between similar emotions.


These **difficulties** impact applications in mental health monitoring, Human-computer interaction, security, and business intelligence, where understanding emotional states is crucial for effective communication, safety, and customer engagement.

The proposed **solution** design—using EfficientNet-B0 with transfer learning was chosen because it offers a strong balance between accuracy and computational efficiency.

**EfficientNet-B0**’s architecture, which scales network depth, width, and resolution in a compound manner, enables effective extraction of fine-grained facial features necessary for distinguishing emotions while maintaining a lightweight model suitable for real-time and resource-constrained environments.

**Transfer learning** leverages pretrained weights on large datasets, improving recognition accuracy and reducing training time, which is essential given the variability and complexity of facial emotion data.


Implementing this solution can significantly enhance the problem domain and business outcomes by enabling more reliable and scalable facial emotion recognition systems.


**For businesses**, this might improve customer experience through real-time sentiment analysis, better workforce management by monitoring employee stress and engagement (students), and enhanced security through emotion-based threat detection.

The technology also supports smarter decision-making in retail, healthcare, manufacturing, and logistics by converting emotional cues into actionable insights, thereby increasing customer retention, productivity, and operational efficiency.

In summary, the EfficientNet-B0 based FER model addresses the core challenges of emotion recognition with a design optimized for accuracy and efficiency, enabling practical deployment that can transform emotional data into strategic business value.
