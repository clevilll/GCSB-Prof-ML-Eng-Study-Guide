# Machine Learning Architecture Questions and Solutions

## 1. Setting up an ML Pipeline for Marketing Team

**Question:**
You need to set up a machine learning pipeline to run a marketing campaign for a retail company. The marketing team needs to periodically update the pipeline and schedule parameters without requiring significant code changes. The solution must be cost-effective and secure. What should you do?

- A. Use Vertex AI Pipelines with Cloud Storage to store and manage the data, and allow the marketing team to update the pipeline parameters through the Google Cloud console.
- B. Use Cloud Composer with BigQuery for the pipeline, and allow the marketing team to manage the pipeline from the Composer interface.
- C. Use Vertex AI Workbench to develop the pipeline and allow the marketing team to update parameters by interacting directly with the Workbench interface.
- D. Use Cloud Composer with email alerts to notify the marketing team of pipeline updates and allow them to manually trigger the pipeline.

### Correct Answer: A. Use Vertex AI Pipelines with Cloud Storage to store and manage the data, and allow the marketing team to update the pipeline parameters through the Google Cloud console.

#### Explanation:
- **A.** Correct: **Vertex AI Pipelines** and **Cloud Storage** offer a cost-effective, secure solution. The marketing team can interact with the pipeline parameters directly through the **Google Cloud Console**, minimizing the need for coding.
- **B.** Not correct: **Cloud Composer** is more suitable for complex workflows and ongoing pipelines, but it's not cost-efficient for a single pipeline, as it requires an always-active environment. Additionally, **BigQuery** is not the most cost-effective for this use case.
- **C.** Not correct: **Vertex AI Workbench** requires manual code updates to change pipeline parameters, which does not minimize the need for interaction by the marketing team.
- **D.** Not correct: **Cloud Composer** is not cost-efficient for just one pipeline. Additionally, using **email** for handling personally identifiable information (PII) is not secure or recommended.

#### Resources:
- [Cloud Storage Encryption](https://cloud.google.com/storage/docs/encryption)
- [Running Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/run-pipeline)
- [Schedule Managed Notebooks with Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/managed/schedule-managed-notebooks-run-quickstart)
- [Setting up MLOps with Composer and MLflow](https://cloud.google.com/architecture/setting-up-mlops-with-composer-and-mlflow)

---

## 2. Optimizing Model Training Execution Time

**Question:**
You are training a deep learning model using TensorFlow on Google Cloud. You want to minimize execution time for model training and ensure the model is trained efficiently. What should you do?

- A. Create an instance group with multiple instances, each with a single GPU attached, and use MirroredStrategy for distributed training.
- B. Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharding.
- C. Create an instance group with multiple TPU nodes, and use the TPU as the accelerator for distributed training.
- D. Create an instance group with a single TPU node, and use it for high-precision distributed training.

### Correct Answer: B. Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharding.

#### Explanation:
- **A.** Not correct: Using **MirroredStrategy** on multiple GPUs in a single instance is suboptimal in minimizing execution time for model training. It only supports parallelism on a single machine and is not as performant as distributed training across multiple instances.
- **B.** Correct: **GPUs** are optimal for deep learning tasks requiring high-precision training. Scaling up with **MultiWorkerMirroredStrategy** will allow distributed training, providing flexibility for tuning accelerator selection to minimize execution time.
- **C.** Not correct: **TPUs** are not recommended for workloads requiring high-precision arithmetic or tasks with shorter training times. TPUs are better suited for long-running tasks and models that take weeks or months to train.
- **D.** Not correct: **TPUs** are not recommended for workloads requiring high-precision arithmetic and should be used only when necessary. Additionally, a single TPU node would not provide as much benefit for this task.

#### Resources:
- [When to Use TPUs](https://cloud.google.com/tpu/docs/intro-to-tpu#when_to_use_tpus)
- [Distributed Training Guide](https://www.tensorflow.org/guide/distributed_training)
- [MultiWorkerMirroredStrategy Tutorial](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl)

---

## 3. Anonymizing Sensitive Data

**Question:**
You have a dataset that contains sensitive information. You want to anonymize the data in a way that preserves referential integrity and prevents model overfitting due to sensitive features. Which transformation should you apply to the sensitive features?

- A. Remove the sensitive features from the model.
- B. Apply masking to the sensitive features in the model.
- C. Hash the sensitive features in the model.
- D. Apply deterministic encryption to the sensitive features in the model.

### Correct Answer: C. Hash the sensitive features in the model.

#### Explanation:
- **A.** Not correct: **Removing features** from the model does not maintain the original relationship between records (referential integrity), which could lead to performance degradation.
- **B.** Not correct: **Masking** does not guarantee referential integrity and could lead to performance drops. Additionally, tuning the existing model is not recommended because it may have memorized sensitive information.
- **C.** Correct: **Hashing** is an irreversible transformation that ensures **anonymization** while preserving referential integrity. It does not lead to a significant drop in model performance because it maintains the same feature set.
- **D.** Not correct: **Deterministic encryption** is reversible, and anonymization requires irreversibility. Additionally, tuning the existing model may have caused it to memorize sensitive information, which is not recommended.

#### Resources:
- [Data Loss Prevention (DLP) Transformations Reference](https://cloud.google.com/dlp/docs/transformations-reference#transformation_methods)
- [De-identify Sensitive Data](https://cloud.google.com/dlp/docs/deidentify-sensitive-data)
- [Security Week Session Guide](https://cloud.google.com/blog/products/identity-security/next-onair20-security-week-session-guide)
- [Creating DLP Job Triggers](https://cloud.google.com/dlp/docs/creating-job-triggers)


---

## 4. Object Detection with Cloud Vision API

**Question:**
You want to develop a custom solution for detecting sticky notes in images. The sticky notes are small and may appear in varying positions and orientations. What should you do?

- A. Use the Cloud Vision API for object detection to identify the sticky notes in the image.
- B. Use AutoML Vision to train a custom model for object detection on the sticky notes.
- C. Create a custom training job for object detection using TensorFlow to detect sticky notes.
- D. Create a custom training job for object detection using PyTorch to detect sticky notes.

### Correct Answer: B. Use AutoML Vision to train a custom model for object detection on the sticky notes.

#### Explanation:
- **A.** Not correct: **Cloud Vision API** provides general object detection but works best with larger objects. It may not be accurate enough for detecting small, variable objects like sticky notes.
- **B.** Correct: **AutoML Vision** is a codeless solution that can handle small objects (down to 8-32 pixels) and is capable of detecting bounding boxes for sticky notes. It minimizes development time and achieves high accuracy for detecting small objects.
- **C.** Not correct: **Custom training jobs** require more development time and extra flexibility, which isn't necessary since AutoML can achieve great results for small objects using transfer learning.
- **D.** Not correct: **Custom training jobs** using **PyTorch** would require a significant amount of development time and flexibility that is not needed. AutoML Vision is simpler and more efficient in this case.

#### Resources:
- [Vertex AI Training Methods](https://cloud.google.com/vertex-ai/docs/start/training-methods)
- [AutoML Vision - Is the Vision API or AutoML the Right Tool?](https://cloud.google.com/vision/automl/docs/beginners-guide#is_the_vision_api_or_automl_the_right_tool_for_me)
- [Prepare Image Datasets for AutoML Vision](https://cloud.google.com/vertex-ai/docs/datasets/prepare-image)
- [Cloud Vision AI Documentation](https://cloud.google.com/vision-ai/docs)


---

## 5. Automating and Orchestrating AI Pipelines

**Question:**
You are developing a machine learning pipeline in Vertex AI. You want to schedule and automate the execution of your pipeline, including triggering the pipeline on a regular schedule. What should you do?

- A. Use Vertex AI Workbench to manually schedule the pipeline to run weekly.
- B. Use Cloud Composer to schedule and automate the pipeline, and check the logs regularly.
- C. Use Kubeflow Pipelines SDK with Google Cloud executors to define your pipeline, and use Vertex AI pipelines to automate the pipeline to run.
- D. Use Vertex AI pipelines with Cloud Scheduler to automate the pipeline but manually manage scheduling tasks in your notebook.

### Correct Answer: C. Use Kubeflow Pipelines SDK with Google Cloud executors to define your pipeline, and use Vertex AI pipelines to automate the pipeline to run.

#### Explanation:
- **A.** Not correct: **Vertex AI Workbench** does not provide built-in scheduling and alerting features. You would need to manually check the pipeline status, which adds to the maintenance overhead.
- **B.** Not correct: **Cloud Composer** is a general-purpose workflow orchestration tool, but it lacks **ML-specific** monitoring capabilities. It also may not be the most cost-efficient for ML pipelines unless you're managing multiple pipelines.
- **C.** Correct: **Kubeflow Pipelines SDK** is the best practice for defining AI pipelines with modular steps, and **Vertex AI Pipelines** is a fully managed solution for automating pipeline execution, making it an optimal choice for orchestration and automation.
- **D.** Not correct: This approach would require more manual effort and is not the best practice. **Vertex AI pipelines** is the more suitable and managed product for automating AI pipeline execution.

#### Resources:
- [ML on GCP Best Practices - Machine Learning Workflow Orchestration](https://cloud.google.com/architecture/ml-on-gcp-best-practices#machine-learning-workflow-orchestration)
- [Vertex AI Workbench - Schedule Managed Notebooks](https://cloud.google.com/vertex-ai/docs/workbench/managed/schedule-managed-notebooks-run-quickstart)
- [Vertex AI Pipelines - Schedule Cloud Scheduler](https://cloud.google.com/vertex-ai/docs/pipelines/schedule-cloud-scheduler)

---

## 6. Configuring Feature Lookup for Low Latency and Scalability

**Question:**
You recently developed a custom ML model that was trained in Vertex AI on a post-processed training dataset stored in BigQuery. You used a Cloud Run container to deploy the prediction service. The service performs feature lookup and pre-processing and sends a prediction request to a model endpoint in Vertex AI. You want to configure a comprehensive monitoring solution for training-serving skew that requires minimal maintenance. What should you do?

- A. Use Vertex AI Feature Store with BigQuery to prioritize low latency, scalability, and minimal maintenance.
- B. Use Cloud Functions to perform feature lookup and inference on Google Kubernetes Engine for model serving.
- C. Use Vertex AI Feature Store with Google Kubernetes Engine for low-latency feature lookup and model inference.
- D. Use Cloud Functions to perform feature lookup and inference, with Memorystore as a low-latency feature store.

### Correct Answer: A. Use Vertex AI Feature Store with BigQuery to prioritize low latency, scalability, and minimal maintenance.

#### Explanation:
- **A.** Correct: Vertex AI **Feature Store** integrated with **BigQuery** is a fully managed solution that prioritizes low-latency and scalability for feature lookup. It requires minimal maintenance and seamlessly integrates with other **Vertex AI** services.
- **B.** Not correct: While Cloud Functions can perform feature lookup and model inference, using **Google Kubernetes Engine** introduces additional maintenance complexity compared to a fully managed solution like **Vertex AI Feature Store**.
- **C.** Not correct: Vertex AI Feature Store is more suitable for low-latency feature lookups than **Google Kubernetes Engine** (GKE). GKE may require more management and maintenance compared to the fully managed Vertex AI services.
- **D.** Not correct: While **Cloud Functions** and **Memorystore** can provide low-latency access, **Vertex AI Feature Store** is specifically optimized for managing features at scale, providing better long-term support with minimal maintenance.

#### Resources:
- [ML on GCP Best Practices - Model Deployment and Serving](https://cloud.google.com/architecture/ml-on-gcp-best-practices#model-deployment-and-serving)
- [Vertex AI Feature Store Overview](https://cloud.google.com/vertex-ai/docs/featurestore/overview#benefits)
- [Google Cloud Memorystore Overview](https://cloud.google.com/memorystore/docs/redis/redis-overview)
- [Vertex AI Feature Store: Data Source Preparation](https://cloud.google.com/vertex-ai/docs/featurestore/latest/overview#data_source_prep)


---

## 7. Diagnosing Model Performance with TensorBoard

**Question:**
You are logged into the Vertex AI Pipeline UI and noticed that an automated production TensorFlow training pipeline finished three hours earlier than a typical run. You do not have access to production data for security reasons, but you have verified that no alert was logged in any of the ML system’s monitoring systems and that the pipeline code has not been updated recently. You want to assure the quality of the pipeline results as quickly as possible so you can determine whether to deploy the trained model. What should you do?

- A. Use Vertex AI TensorBoard to check whether the training metrics converge to typical values. Verify pipeline input configuration and steps have the expected values.
- B. Upgrade to the latest version of the Vertex SDK and re-run the pipeline.
- C. Determine the trained model’s location from the pipeline’s metadata in Vertex ML Metadata, and compare the trained model’s size to the previous model.
- D. Request access to production systems. Get the training data’s location from the pipeline’s metadata in Vertex ML Metadata, and compare data volumes of the current run to the previous run.

### Correct Answer: A. Use Vertex AI TensorBoard to check whether the training metrics converge to typical values. Verify pipeline input configuration and steps have the expected values.

#### Explanation:
- **A.** Correct: **TensorBoard** provides a comprehensive overview of training metrics like loss and accuracy over time. By checking these metrics, you can quickly verify whether the model is performing as expected and if it converges to typical values. This is the most efficient way to assess whether the model is ready for deployment.
- **B.** Not correct: Upgrading the Vertex SDK and re-running the pipeline does not directly address the current issue. While updating the SDK may resolve bugs or incompatibilities, it is not the best approach to check the current model's performance.
- **C.** Not correct: While model size can be an indicator of health, it is not enough to fully assess if the model is performing well. TensorBoard offers a more complete view of training performance.
- **D.** Not correct: Although data could be the cause of the issue, TensorBoard provides a better overview of training and model performance. Accessing production systems is also not necessary for this analysis.

#### Resources:
- [Vertex AI TensorBoard Overview](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview)
- [Vertex AI ML Metadata Introduction](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction)
- [Vertex AI Pipelines: Visualize Pipeline](https://cloud.google.com/vertex-ai/docs/pipelines/visualize-pipeline)


---

## 8. Model Monitoring for Training-Serving Skew

**Question:**
You recently deployed a custom-trained model in Vertex AI Prediction. You want to set up a comprehensive monitoring solution to track training-serving skew while minimizing maintenance. What should you do?

- A. Use Vertex AI Model Monitoring to detect training-serving skew, and use the console to manually diagnose any issues.
- B. Use TensorFlow Extended (TFX) to define the pipeline and automatically handle training-serving skew detection.
- C. Use Cloud Monitoring to track training-serving skew and trigger model retraining on skew detection.
- D. Use a custom solution with Cloud Logging and Cloud Functions to detect and address training-serving skew.

### Correct Answer: A. Use Vertex AI Model Monitoring to detect training-serving skew, and use the console to manually diagnose any issues.

#### Explanation:
- **A.** Correct: **Vertex AI Model Monitoring** is a fully managed solution for detecting training-serving skew. It requires minimal maintenance and provides tools for diagnosing issues through the console. This approach ensures continuous monitoring with minimal intervention required.
- **B.** Not correct: While **TensorFlow Extended (TFX)** is useful for defining pipelines, it requires more manual setup and maintenance. TFX involves multiple components that need to be updated if the schema changes, increasing maintenance.
- **C.** Not correct: **Cloud Monitoring** can help with tracking skew, but a model retrain triggered by skew detection does not necessarily address the root causes of the issue, such as differences in preprocessing logic between training and prediction.
- **D.** Not correct: This custom solution with **Cloud Logging** and **Cloud Functions** also involves maintenance-heavy components and does not guarantee the detection of skew issues. A retrain triggered by skew detection might not be sufficient to address the problem.

#### Resources:
- [ML Modeling & Monitoring for Skew Detection](https://cloud.google.com/architecture/ml-modeling-monitoring-automating-server-data-skew-detection-in-ai-platform-prediction)
- [Vertex AI Model Monitoring Overview](https://cloud.google.com/vertex-ai/docs/model-monitoring/overview)


---

## 9. Removing Age Bias from Model Predictions

**Question:**
You recently developed a classification model that predicts which customers will be repeat customers. Before deploying the model, you perform post-training analysis on multiple data slices and discover that the model is under-predicting for users who are more than 60 years old. You want to remove age bias while maintaining similar offline performance. What should you do?

- A. Perform correlation analysis on the training feature set against the age column, and remove features that are highly correlated with age from the training and evaluation sets.
- B. Review the data distribution for each feature against the bucketized age column for the training and evaluation sets, and introduce preprocessing to even irregular feature distributions.
- C. Configure the model to support explainability, and modify the input-baselines to include min and max age ranges.
- D. Apply a calibration layer at post-processing that matches the prediction distributions of users below and above 60 years old.

### Correct Answer: B. Review the data distribution for each feature against the bucketized age column for the training and evaluation sets, and introduce preprocessing to even irregular feature distributions.

#### Explanation:
- **A.** Not correct: Removing features highly correlated with age could lead to significant drops in offline performance as this would reduce the ability of the model to learn relevant patterns in the data.
- **B.** Correct: **Preprocessing** the data, including analyzing and adjusting the data distributions for users above 60 years old, helps to correct any data imbalance or bias. This could involve techniques like sampling, bucketing, or adjusting the distribution of certain features.
- **C.** Not correct: Modifying **input-baselines** for explainability might improve interpretability but will not address the underlying model bias related to age. It will not directly help in removing the bias.
- **D.** Not correct: Applying a **calibration layer** might help adjust post-processing results, but it could lead to introducing implicit or unconscious bias in the predictions, and it doesn't address the root cause of the bias.

#### Resources:
- [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)
- [Inclusive Machine Learning](https://cloud.google.com/inclusive-ml)
- [Understanding Types of Bias in ML](https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias)
- [Reducing Prediction Bias](https://developers.google.com/machine-learning/crash-course/classification/prediction-bias)

---

## 10. Reducing Latency in a Pre-trained TensorFlow Model

**Question:**
You downloaded a TensorFlow language model pre-trained on a proprietary dataset by another company, and you tuned the model with Vertex AI Training by replacing the last layer with a custom dense layer. The model achieves the expected offline accuracy; however, it exceeds the required online prediction latency by 20ms. You want to reduce latency while minimizing the offline performance drop and modifications to the model before deploying the model to production. What should you do?

- A. Apply post-training quantization on the tuned model, and serve the quantized model.
- B. Apply knowledge distillation to train a new, smaller "student" model that mimics the behavior of the larger, fine-tuned model.
- C. Use pruning to tune the pre-trained model on your dataset, and serve the pruned model after stripping it of training variables.
- D. Use clustering to tune the pre-trained model on your dataset, and serve the clustered model after stripping it of training variables.

### Correct Answer: A. Apply post-training quantization on the tuned model, and serve the quantized model.

#### Explanation:
- **A.** Correct: **Post-training quantization** is the recommended approach for reducing model latency when retraining is not feasible. While it may cause a slight drop in performance, it minimizes latency without requiring extensive retraining.
- **B.** Not correct: **Knowledge distillation** focuses on training a smaller "student" model but will likely result in a drop in offline performance, which could negate the benefit of reducing latency.
- **C.** Not correct: **Pruning** reduces model size but is unlikely to provide as significant latency improvements as quantization, and it could also cause a drop in offline performance.
- **D.** Not correct: **Clustering** helps in compressing model size but doesn't effectively reduce latency in the way that post-training quantization does. It also risks causing a performance drop.

#### Resources:
- [Best Practices for ML Performance and Cost](https://cloud.google.com/architecture/best-practices-for-ml-performance-cost)
- [TensorFlow Lite Performance: Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)
- [TensorFlow Tutorial on Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Vertex AI Distill Text Models Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/distill-text-models)

---

## 11. Optimizing Model Convergence and Reducing Overfitting

**Question:**
You have a dataset that is split into training, validation, and test sets. All the sets have similar distributions. You have sub-selected the most relevant features and trained a neural network. TensorBoard plots show the training loss oscillating around 0.9, with the validation loss higher than the training loss by 0.3. You want to update the training regime to maximize the convergence of both losses and reduce overfitting. What should you do?

- A. Decrease the learning rate to fix the validation loss, and increase the number of training epochs to improve the convergence of both losses.
- B. Decrease the learning rate to fix the validation loss, and increase the number and dimension of the layers in the network to improve the convergence of both losses.
- C. Introduce L1 regularization to fix the validation loss, and increase the learning rate and the number of training epochs to improve the convergence of both losses.
- D. Introduce L2 regularization to fix the validation loss.

### Correct Answer: D. Introduce L2 regularization to fix the validation loss.

#### Explanation:

- **A.** Not correct: **Changing the learning rate** does not directly address overfitting, and **increasing the number of training epochs** is unlikely to improve the losses significantly.
- **B.** Not correct: **Changing the learning rate** doesn’t reduce overfitting. Increasing the **number and dimension of layers** could lead to an overly complex model, worsening overfitting.
- **C.** Not correct: **Increasing the learning rate** may destabilize the training, and **L1 regularization** might not be as effective, especially when only the most relevant features are used.
- **D.** Correct: **L2 regularization** helps prevent overfitting by penalizing large weights, and it can improve model performance by optimizing loss convergence, particularly when underfitting.

#### Resources:
- [Overview of Testing and Debugging ML Models](https://developers.google.com/machine-learning/testing-debugging/common/overview)
- [L2 Regularization for Simplicity](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization)
- [L1 Regularization for Sparsity](https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/l1-regularization)
- [Preventing Overfitting with BigQuery ML](https://cloud.google.com/bigquery-ml/docs/preventing-overfitting)
- [TensorFlow Tutorial on Overfitting and Underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard/get_started)
- [Guidelines for Developing High-Quality ML Solutions](https://cloud.google.com/architecture/guidelines-for-developing-high-quality-ml-solutions#guidelines_for_model_quality)


---

## 12. Defining a Rollout Strategy for a New Model Version

**Question:**
You recently deployed a custom-trained model in production with Vertex AI Prediction. The automated retraining pipeline has made available a new model version that passed all unit and infrastructure tests. You want to define a rollout strategy for the new model version that guarantees an optimal user experience with zero downtime. What should you do?

- A. Release the new model version in the same Vertex AI endpoint. Use traffic splitting in Vertex AI Prediction to route a small random subset of requests to the new version and, if the new version is successful, gradually route the remaining traffic to it.
- B. Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application.
- C. Deploy the current model version with an Istio resource in Google Kubernetes Engine, and route production traffic to it. Deploy the new model version, and use Istio to route a small random subset of traffic to it. If the new version is successful, gradually route the remaining traffic to it.
- D. Install Seldon Core and deploy an Istio resource in Google Kubernetes Engine. Deploy the current model version and the new model version using the multi-armed bandit algorithm in Seldon to dynamically route requests between the two versions before eventually routing all traffic over to the best-performing version.

### Correct Answer: B. Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application.

#### Explanation:

- **A.** Not correct: **Canary deployments** may still affect user experience, even if a small subset of users is impacted.
- **B.** Correct: **Shadow deployments** ensure zero downtime by testing the new version in parallel with the old one without affecting user experience. This allows for seamless updates without impacting users.
- **C.** Not correct: **Canary deployments** may still impact user experience and could lead to downtime when transitioning between services.
- **D.** Not correct: The **multi-armed bandit** approach can still affect user experience, even with small subsets of traffic. It may also cause downtime when switching between services.

#### Resources:
- [MLOps Continuous Delivery and Automation Pipelines in Machine Learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#data_and_model_validation)
- [Implementing Deployment and Testing Strategies on GKE](https://cloud.google.com/architecture/implementing-deployment-and-testing-strategies-on-gke)
- [Application Deployment and Testing Strategies](https://cloud.google.com/architecture/application-deployment-and-testing-strategies#choosing_the_right_strategy)
- [Vertex AI Deployment Documentation](https://cloud.google.com/vertex-ai/docs/general/deployment)
- [Seldon Core Multi-Armed Bandit](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/routers.html)


---

## 13. Developing a Robust, Scalable ML Pipeline for Regression and Classification Models

**Question:**
You are developing a robust, scalable ML pipeline to train several regression and classification models. Your primary focus for the pipeline is model interpretability. You want to productionize the pipeline as quickly as possible. What should you do?

- A. Use Tabular Workflow for Wide & Deep through Vertex AI Pipelines to jointly train wide linear models and deep neural networks.
- B. Use Cloud Composer to build the training pipelines for custom deep learning-based models.
- C. Use Google Kubernetes Engine to build a custom training pipeline for XGBoost-based models.
- D. Use Tabular Workflow for TabNet through Vertex AI Pipelines to train attention-based models.

### Correct Answer: D. Use Tabular Workflow for TabNet through Vertex AI Pipelines to train attention-based models.

#### Explanation:

- **A.** Not correct: **Tabular Workflows for Wide & Deep** is capable of handling classification and regression pipelines, but it’s optimized for memorization and generalization. Deep learning-based models are not preferred for interpretability.
- **B.** Not correct: **Cloud Composer** is not the best tool to quickly build ML pipelines. Deep learning-based models are also generally not preferred for interpretability.
- **C.** Not correct: Building a pipeline on **Google Kubernetes Engine** would take a long time and be more complex compared to Vertex AI Pipelines.
- **D.** Correct: **TabNet** uses sequential attention, which promotes model interpretability. **Tabular Workflows** is a set of integrated, fully managed, and scalable pipelines for end-to-end ML with tabular data for regression and classification.

#### Resources:
- [Tabular Workflows Overview](https://cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/overview#cr-tabnet)

---

## 14. Storing and Accessing Image Files for Custom Model Training

**Question:**
You are developing a custom image classification model in Python. You plan to run your training application on Vertex AI. Your input dataset contains several hundred thousand small images. You need to determine how to store and access the images for training. You want to maximize data throughput and minimize training time while reducing the amount of additional code. What should you do?

- A. Store image files in Cloud Storage, and access them directly.
- B. Store image files in Cloud Storage, and access them by using serialized records.
- C. Store image files in Cloud Filestore, and access them by using serialized records.
- D. Store image files in Cloud Filestore, and access them directly by using an NFS mount point.

### Correct Answer: C. Store image files in Cloud Filestore, and access them by using serialized records.

#### Explanation:

- **A.** Not correct: **Cloud Storage** is not optimized for accessing lots of small files. There is overhead in establishing connections to retrieve each individual file.
- **B.** Not correct: Although **serialized records** (TFRecords, WebDatasets) are faster than accessing individual small files, **Cloud Storage** is still slower compared to Cloud Filestore.
- **C.** Correct: **Cloud Filestore** is faster than Cloud Storage for accessing files, and serialized records (e.g., TFRecords) are faster for feeding training pipelines than individual files.
- **D.** Not correct: While **Filestore** is faster than Cloud Storage for file access, serialized records are still faster than accessing individual file I/O.

#### Resources:
- [WebDataset](https://github.com/webdataset/webdataset)
- [Efficient PyTorch Training with Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/efficient-pytorch-training-with-vertex-ai)
- [Scaling Deep Learning Workloads with PyTorch XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
- [Reading and Storing Data for Custom Model Training in Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/reading-and-storing-data-custom-model-training-vertex-ai)


---

## 15. Optimizing Model Deployment Pipeline

**Question:**
You recently deployed a model in production using Vertex AI. The model is updated frequently, and you want to configure a pipeline that will ensure the model is retrained and deployed automatically when a new version is available, while ensuring minimal downtime and optimal user experience. What should you do?

- A. Use Vertex AI Model Monitoring to track model performance and automatically retrain the model if performance drops below a defined threshold.
- B. Use Vertex AI Pipelines to define an automated pipeline for model retraining and deployment, and set up traffic splitting to ensure zero downtime during model rollout.
- C. Set up a custom retraining pipeline in Cloud Composer that triggers a Cloud Function to deploy a new model version whenever a performance drop is detected.
- D. Use a Kubernetes CronJob to periodically check for model performance, retrain the model if needed, and deploy the new model to the existing endpoint.

### Correct Answer: C. Set up a custom retraining pipeline in Cloud Composer that triggers a Cloud Function to deploy a new model version whenever a performance drop is detected.

#### Explanation:

- **A.** Not correct: Cloud Functions may run into limitations based on request rate and model size, which could impact performance.
- **B.** Not correct: Exposing the model as an endpoint adds to the total latency, which may not meet the low-latency requirement.
- **C.** Correct: The **RunInference API** with a locally loaded model minimizes the prediction latency and makes model updates seamless.
- **D.** Not correct: Provisioning **Vertex AI Pipelines** adds to the total latency, making it unsuitable for low-latency requirements.

#### Resources:
- [RunInference API with WatchFilePattern in Dataflow](https://cloud.google.com/dataflow/docs/notebooks/run_custom_inference)
- [Cloud Functions and Pub/Sub Tutorial](https://cloud.google.com/functions/docs/tutorials/pubsub)
- [Vertex AI Pipelines Triggering from Pub/Sub](https://cloud.google.com/vertex-ai/docs/pipelines/trigger-pubsub)
- [Cloud Functions Quotas](https://cloud.google.com/functions/quotas)
- [Minimizing Predictive Serving Latency in Machine Learning](https://cloud.google.com/architecture/minimizing-predictive-serving-latency-in-machine-learning)









