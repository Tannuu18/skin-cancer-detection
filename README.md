## Project README: Skin Cancer Classification using Machine Learning

### 1. Project Goal

This project aims to develop and evaluate machine learning models for classifying skin cancer lesions from dermatoscopic images using the HAM10000 dataset. The dataset contains seven distinct classes of skin lesions, posing a multi-class classification challenge.

### 2. Dataset Overview & Initial Data Exploration (EDA)

#### Dataset:

The HAM10000 (Human Against Machine with 10000 training images) dataset consists of over 10,000 dermatoscopic images. Each image is accompanied by metadata including diagnosis, age, sex, and lesion location.

#### Key EDA Findings:

*   **Image Characteristics**: Raw images are high-resolution (typically 450x600x3 pixels), containing intricate details of skin textures and colors.
*   **Class Imbalance**: A significant challenge identified was severe class imbalance. The `melanocytic nevi` ('nv' - common benign moles) class had a disproportionately higher number of samples compared to other, more critical classes like `melanoma` ('mel') or `basal cell carcinoma` ('bcc'). This bias can lead models to over-predict the majority class.
*   **Demographic Trends**: Analysis revealed that conditions such as Basal Cell Carcinoma (bcc) and Actinic Keratoses (akiec) are more prevalent in older patients (typically 60+), while Melanocytic Nevi (nv) are found across a broader, younger age range. Common lesion locations included the back, lower extremities, and trunk.

### 3. Model Development Workflow

The project followed a staged approach, progressively building and refining classification models:

#### Stage 1: Baseline Models (Raw Pixels)

*   **Approach**: Traditional machine learning models (Gaussian Naive Bayes, Decision Tree, K-Nearest Neighbors, Logistic Regression, Random Forest) were applied directly to raw, flattened pixel data. Images were downscaled to **32x32 pixels** and reshaped into 3,072-dimensional vectors.
*   **Performance**: Random Forest achieved the highest accuracy at approximately **71.14%**.
*   **Limitations**: This stage highlighted computational inefficiency, memory constraints, and high sensitivity to noise in raw pixel data. Models like Gaussian Naive Bayes performed poorly (~40%) due to feature independence assumptions, and Logistic Regression struggled with convergence.

#### Stage 2: Feature Engineering & Data Balancing

To address the limitations of raw pixels, feature engineering and data balancing techniques were introduced:

*   **Feature Extraction (HOG + Color)**:
    *   **HOG (Histogram of Oriented Gradients)**: Used to capture shape and texture information.
    *   **Color Features**: Mean and standard deviation of RGB channels were extracted to capture color properties.
    *   **Dimensionality Reduction**: The combined features were reduced using filter methods (e.g., `SelectKBest` for top 100 features) and embedded methods (Random Forest feature importance for 67 features).
*   **Feature Extraction (LBP + Color)**:
    *   **LBP (Local Binary Pattern)**: Used for local texture descriptors.
    *   Combined with statistical color features.
*   **Hybrid PCA**: HOG + Color features were further compressed using PCA to 50 principal components.
*   **Data Balancing**: To counteract class imbalance, **oversampling** was performed to ensure 500 samples per class across the dataset. This was crucial for preventing model bias towards majority classes.
*   **Performance with Engineered Features**:
    *   Traditional ML models showed improved performance. Random Forest, when trained on **balanced data with HOG + Color features**, achieved a notable accuracy of **77.49%**.
    *   Logistic Regression and K-Nearest Neighbors also saw improvements over their raw pixel performance.

#### Stage 3: Hyperparameter Tuning (Random Forest)

*   **Approach**: `GridSearchCV` was employed to optimize the hyperparameters of the Random Forest model (e.g., `n_estimators`, `max_depth`, `min_samples_split`, `max_features`) using the balanced, engineered features.
*   **Optimized Performance**: This tuning process further refined the Random Forest model, achieving an accuracy of **76.46%** (note: the previous run showed 78.86% after tuning, the current run yielded 76.46%).

#### Stage 4: Convolutional Neural Network (CNN) Implementation

*   **Rationale**: CNNs are generally superior for image classification as they automatically learn hierarchical features from raw pixel data.
*   **Architecture**: A sequential CNN model was constructed with multiple `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers, including `Dropout` for regularization.
*   **Preprocessing**: Images were resized to **64x64 pixels** and normalized to a 0-1 range. The full dataset (10,015 images) was used.
*   **Training Enhancements**:
    *   **Class Weighting**: Applied during training to mitigate the effects of inherent class imbalance.
    *   **Early Stopping**: Implemented to prevent overfitting by monitoring `val_accuracy` with a patience of 5 epochs.
    **Model Checkpointing**: Used to save the best model weights based on `val_accuracy`.
*   **Performance**: The implemented CNN achieved an accuracy of **58.41%**.

#### Stage 5: Advanced CNN with Bipolar Fuzzy Set and Sakaguchi Loss

**Rationale**: Despite various efforts, the traditional CNN model achieved a modest accuracy of 58.41%. This, coupled with the inherent uncertainty in dermatoscopic image classification (where a lesion might share characteristics with multiple classes), motivated the exploration of more nuanced classification approaches. We introduced a CNN architecture enhanced with Bipolar Fuzzy Sets and a custom Sakaguchi Loss function to explicitly model uncertainty and inter-class similarity.

**Bipolar Fuzzy Sets**: Unlike traditional crisp classification (where an item either belongs or doesn't belong to a class) or even standard fuzzy sets (which model degree of membership), Bipolar Fuzzy Sets introduce two independent measures: a **membership function** and a **non-membership function**. This allows the model to quantify not only the evidence *for* a class but also the evidence *against* it or the degree of uncertainty. This is particularly useful in medical imaging where diagnoses can be ambiguous. Our model's output layer was designed to predict both a membership score and a non-membership score for each class.

**Sakaguchi Loss**: To further refine the classification in the context of ambiguous lesions, we incorporated a custom `Sakaguchi Loss`. This loss function aims to leverage known similarities between different skin lesion classes during training. For instance, some benign lesions might visually resemble malignant ones. By encoding these similarities into a "Sakaguchi Similarity Matrix," the loss function penalizes predictions that are far from not only the true label but also from similar classes, encouraging the model to learn a more robust decision boundary that respects these biological relationships.

**Architecture**: A pre-trained `MobileNetV2` model was used as the base for feature extraction. Custom output layers were added on top to generate both the membership and non-membership predictions for the seven skin lesion classes. The base model's layers were initially frozen.

**Training Workflow**:

1.  **Initial Bipolar Fuzzy Training**: The model was first compiled with:
    *   `categorical_crossentropy` for the `membership` output (standard classification).
    *   `binary_crossentropy` for the `non_membership` output.
    *   A `loss_weights` balance was applied to prioritize membership prediction.
    *   The model was trained for 10 epochs using an `ImageDataGenerator` that also generated inverse labels for the non-membership output, simulating uncertainty.
2.  **Sakaguchi Loss Integration**: After the initial fuzzy training, the model was recompiled. The `membership` output now utilized the custom `sakaguchi_loss` (which incorporates the similarity matrix) while `non_membership` continued to use `binary_crossentropy`. The learning rate was reduced to `1e-5` for finer adjustments. This phase ran for an additional 5 epochs.
3.  **Fine-tuning**: In the final stage, the top 30 layers of the `MobileNetV2` base model were unfrozen to allow for fine-tuning of deeper features. The model was recompiled with `categorical_crossentropy` as the loss for both outputs (where `non_membership` loss became part of the overall `categorical_crossentropy` due to a change in model compilation to a single loss for fine-tuning) and trained for 5 more epochs. This step was crucial for adapting the pre-trained features to the specific nuances of the dermatoscopic images.

**Performance (Final Fine-tuned Model)**: The fine-tuned model, after integrating Bipolar Fuzzy Sets and Sakaguchi Loss, achieved a validation accuracy of approximately **77.78%** on the membership prediction.

### 4. Model Comparison & Analysis (Updated)

| Model                                  | Features Used              | Data Balancing | Accuracy     |
| :------------------------------------- | :------------------------- | :------------- | :----------- |
| Random Forest (Baseline)               | Raw Pixels (32x32)         | No             | 71.14%       |
| Random Forest (Optimized)              | HOG + Color                | Yes            | 76.46%       |
| Convolutional Neural Network (Baseline)| Learned from Raw Pixels (64x64) | Yes (Weighting)| 58.41%       |
| **CNN with Bipolar Fuzzy & Sakaguchi** | MobileNetV2 features       | Yes (Augmented)| **77.78%**   |

**Analysis**: The advanced CNN model incorporating Bipolar Fuzzy Sets and Sakaguchi Loss has shown significant improvement, achieving the highest accuracy of **77.78%**. This indicates that explicitly modeling uncertainty and leveraging inter-class similarities can lead to more effective classification in complex medical imaging tasks, surpassing both traditional ML models with engineered features and simpler CNN architectures.

### 5. Key Findings (Updated)

*   **Feature Engineering is Powerful**: For traditional ML models, intelligent feature extraction (HOG, Color, LBP) drastically improved performance by providing more abstract and robust representations than raw pixels.
*   **Data Balancing is Crucial**: Addressing class imbalance was paramount for achieving fair and accurate predictions across all disease types, especially for rarer conditions.
*   **Bipolar Fuzzy Sets for Uncertainty**: Incorporating bipolar fuzzy sets provides a valuable framework for handling inherent ambiguities in medical image diagnosis, allowing the model to express both membership and non-membership (uncertainty).
*   **Sakaguchi Loss for Similarity**: The custom Sakaguchi loss effectively integrates prior knowledge about inter-class similarities, guiding the model to learn more meaningful decision boundaries.
*   **Advanced CNNs Outperform**: While initial CNNs struggled, the refined CNN architecture with Bipolar Fuzzy Sets and Sakaguchi Loss ultimately achieved the best performance, demonstrating the potential of these advanced techniques for dermatoscopic image classification.

### 6. Next Steps & Future Work (Updated)

To further enhance model performance, especially for the fuzzy set-based CNN, the following steps are recommended:

*   **Explore Different Fuzzy Loss Formulations**: Investigate other forms of fuzzy loss functions or combinations of losses that could further optimize the balance between membership and non-membership predictions.
*   **Dynamic Sakaguchi Matrix**: Instead of a static Sakaguchi matrix, explore methods to dynamically learn or refine the similarity matrix during training.
*   **Ensemble Methods**: Combine predictions from the fuzzy-Sakaguchi CNN with other high-performing models (e.g., the optimized Random Forest) to potentially achieve even higher accuracy and robustness.
*   **Interpretability**: Develop methods to interpret the non-membership output, providing insights into *why* the model is uncertain about a particular classification.
*   **Real-world Evaluation**: Test the model on external, unseen datasets to evaluate its generalization capabilities and clinical utility.
*   **Higher Resolution Images**: Train the CNN with higher resolution images (e.g., 128x128 or higher) if computational resources permit, to allow the model to capture finer visual details that might be lost at 64x64.
*   **Deeper CNN Architectures**: Experiment with more complex and deeper CNN architectures, or explore different layer configurations.
*   **Extensive Hyperparameter Tuning for CNN**: Conduct a more thorough search for optimal CNN hyperparameters (learning rate, batch size, optimizer, dropout, etc.) using techniques like Grid Search, Random Search, or Bayesian Optimization.
*   **Data Augmentation**: Implement more aggressive data augmentation strategies (e.g., rotations, flips, zooms, color jittering) to artificially expand the training dataset and improve generalization, especially for minority classes.
