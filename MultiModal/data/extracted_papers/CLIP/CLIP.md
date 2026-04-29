
# Fully Fine-tuned CLIP Models are Efficient Few-Shot Learners


# Mushui Liu, Bozheng Li, Yunlong Yu Zhejiang University


arXiv:2407.04003v1  [cs.CV]  4 Jul 2024


Abstract.


Prompt tuning, which involves training a small set of parameters, effectively enhances the pre-trained Vision-Language Models (VLMs) to downstream tasks. However, they often come at the cost of flexibility and adaptability when the tuned models are applied to different datasets or domains. In this paper, we explore capturing the task-specific information via meticulous refinement of entire VLMs, with minimal parameter adjustments. When fine-tuning the entire VLMs for specific tasks under limited supervision, overfitting and catastrophic forgetting become the defacto factors. To mitigate these issues, we propose a framework named CLIP-CITE via designing a discriminative visual-text task, further aligning the visual-text semantics in a supervision manner, and integrating knowledge distillation techniques to preserve the gained knowledge. Extensive experimental results under few-shot learning, base-to-new generalization, domain generalization, and cross-domain generalization settings, demonstrate that our method effectively enhances the performance on specific tasks under limited supervision while preserving the versatility of the VLMs on other datasets.


# 1 Introduction


Recently, the pre-trained Vision-Language Models (VLMs) such as CLIP [23] and ALIGN [13] have demonstrated impressive generalization capabilities across various downstream tasks, including image recognition [35, 34], object detection [7], image segmentation [24], and action recognition [25]. Though versatile, the performance of the VLMs on specific domains shows considerable potential for improvement, especially under limited supervision [35].


The existing methods attempt to equip VLMs with domain-specific knowledge by employing various tuning techniques. Prompt-based Tuning [35, 34, 15, 14] (as shown in Fig. 1 (a)), which refines the pre-trained models with specific prompts while keeping the parameters of VLMs fixed, has gained popularity due to its efficient parameter utilization and capability of quickly adapting VLMs to domain-specific information.


While the prompt-based tuning strategies enable VLMs to effectively capture domain-specific information with limited supervision [35, 34, 15, 14], there is a risk that these strategies may compromise the versatility of VLMs. In other words, the prompts trained on domain-specific data may struggle to generalize to other domains, limiting their versatility. A piece of evidence is provided in Fig. 1(c), which shows a transfer experiment under the cross-domain generalization setting. We employ a few-shot setting to train the model using the


![](figures/fileoutpart0.png)

Image / Text Backbone Task B 🔥 ✔ ✔ Task A Image / Text Backbone Task B ✔ Task A ❄ × 🔥 🔥 : Prompt

(a) Prompt-based Tuning


(b) Fine-tuning


![](tables/fileoutpart1.png)


Method


EuroSAT ⇒ ImageNet


Base ⇒ New


Base


New


CLIP [23]


56.48


64.05


72.43


68.14


CoOp [35]


92.19


54.74


51.81


49.69


MaPLe [15]


95.31


71.28


36.88


44.76


FT-Probe [23]


60.86


71.34


72.03


68.04


CLIP-CITE (Ours)


95.61


80.59


72.29


68.38


(c) Cross-Domain Generalization Experiments


Figure 1: (a) Prompt-based methods introduce a few trainable prompts to incorporate task-specific knowledge. (b) Fine-tuning methods adjust the whole model to adapt to the specific tasks. (c) Comparison results (%) under the cross-domain generalization setting in the limited-data regime.


EuroSAT base training set, followed by evaluating its performance on both EuroSAT and ImageNet datasets. While the prompt-based methods, i.e., CoOp [35] and MaPLe [15] significantly improve the EuroSAT dataset results, they are at the cost of sacrificing their gener-alizability on the other datasets. In particular, their performances on the ImageNet dataset severely lag behind those of the zero-shot CLIP model.


In this work, we restate the professionalism and versatility of VLMs. Professionalism highlights the ability of VLMs to excel in specific domains, categories, and tasks, while versatility highlights their capability to perform across various domains, categories, and tasks. Based on the previous analysis, the prompt-based approaches improve the VLMs’ professionalism but compromise their versatility.


When customizing VLMs to specific domains, fine-tuning the entire models would distribute task-specific knowledge across all parameters (as illustrated in Fig. 1. (b)). Unlike prompt-based tuning techniques, strategies that involve fine-tuning the entire VLMs have been relatively under-explored and under-appreciated, particularly in limited data regimes, due to the significant number of training parameters involved. One example is the FT-Probe [23], which employs a straight-


forward strategy of fine-tuning the entire model and incorporating a linear probe on top of the visual representations. This approach allows the model to preserve the model’s versatility during the adaptation for specific domains, as evident from the results achieved by FT-Probe, presented in Fig. 1(c). However, this fine-tuning strategy has demonstrated only marginal improvements in specific domains compared to prompt-based competitors. We posit that the limited supervision when tailoring the models leads to the emergence of an overfitting issue, which undermines the fine-tuning strategy’s effectiveness in specific domains. More evidence is provided in supplementary materials.


In this paper, we propose a fine-tuning method called CLIP-CITE that enhances the CLIP’s professionalism on specific domains while preserving its versatility by primarily enhanCing the capability of the Image-Text alignmEnt task. Specifically, our CLIP-CITE approach incorporates three key aspects. Firstly, to quickly equip the domain-specific information for CLIP, our CLIP-CITE connects the alignment score with the classification probability in a way that prioritizes higher alignment scores for image-text pairs belonging to the same class. Secondly, our approach fine-tunes the entire model using an image-text alignment task, aligning with the original training objective of the pre-trained CLIP model. This differs from the classification task utilized in [23], ensuring a consistent training objective throughout the adaptation process. Note that training an image-text alignment task usually requires a large batch [23, 6] in implementation, posing a significant challenge when working with limited data regimes. To overcome this issue, we propose utilizing a class-level image-text alignment task as an alternative to the original instance-level alignment task. Finally, to alleviate the catastrophic forgetting issue, we introduce a vision-language similarity distillation strategy. This strategy regularizes the model by transferring the image-text alignment relationship learned by the pre-trained CLIP model, further ensuring a minimal change in parameters. As shown in the last row of Fig. 1 (c), our CLIP-CITE enhances EuroSAT dataset performance while simultaneously upholding generalization capability on the ImageNet dataset.


In summary, our highlights are as follows:

- •
- We propose CLIP-CITE, a simple but efficient fine-tuning method that enhances the VLMs’ professionalism while maintaining their versatility under limited data supervision. CLIP-CITE comprehensively fine-tunes CLIP to enable it to promptly incorporate task-specific information through enhanced image-text alignment and safeguard the learned knowledge.
- •
- We evaluate CLIP-CITE through experiments in different settings, including few-shot image recognition, base-to-new generalization, domain generalization, and cross-domain scenarios. The experimental results demonstrate that CLIP-CITE not only sets new benchmarks in these tasks on specific datasets, but also preserves the original versatility of CLIP on other datasets.

# 2 Related work


## 2.1 Vision-Language Model


Recent years have witnessed remarkable achievements on large-scale pre-trained vision-language models [23, 13, 31, 1, 30, 12]. Representatively, CLIP, ALIGN [23, 13] jointly associate the images and their corresponding text descriptions by optimizing a contrastive objective. Training on the millions of image-text pairs, CLIP aligns the image and language space, showing the powerful generalization on downstream tasks. Based on CLIP, many works seek to transfer the model to special tasks, e.g., few-shot image recognition [35, 34, 15],


segmentation [24], and action recognition [25]. In this paper, we also leverage the benefits of multi-modal alignment and the generalization ability of CLIP. By fine-tuning the CLIP model in limited data regimes, we investigate how the model can adapt its knowledge and generalize to perform well in this particular challenging scenario.


## 2.2 Few-Shot Transfer Learning Based on CLIP


Prompt tuning [35, 34, 14, 15] and fine-tuning [27, 32, 6, 18] are two main methods to transfer the CLIP to the downstream tasks. Prompt tuning is widely used in language models [11, 19], which raises attention in vision and multi-modality areas [35, 14, 16]. Context Optimization (CoOp) [35] improves the downstream few-shot image recognition tasks via learning the soft textual prompts. Co-CoOp [34] and MaPLe [15] further boost the generalization ability through the image-condition information and multi-modal prompts, respectively. Except for the textural prompts, Visual Prompt Tuning (VPT) [14] introduces the vision prompts on the large vision models. Although these prompt tuning methods show efficient and excellent performance, they may fail to overfit the task-specific distribution.


As the alternative, fine-tuning methods directly optimize the model under task-specific situations. WiSE-FT [32], LP-FT [18] achieves the robustness of fine-tuning via a weight-ensemble manner. CLIPood [27] further finetunes the model via the text semantic similarity and model ensemble under an out-of-distribution situation. A similar work related to our method is FLYP [6], which fine-tunes the CLIP model via the pre-trained contrastive objective to obtain the multi-modal alignment ability. In comparison, our method distinguishes the supervised vision-language pairs and incorporates the task-specific into the fine-tuning process. Leveraging this improved image-text alignment task, our method aims to perform more robustly under limited supervision.


# 3 Method


In this work, we fine-tune the CLIP models [23] for the scenarios with limited data available. The architecture of CLIP includes two key components: a visual encoder denoted as θI and a text encoder denoted as θT . By aligning language and visual modalities on 400 million text-image web data, CLIP is endowed with zero-shot and open-vocabulary capabilities.


To perform zero-shot classification, CLIP utilizes handcrafted text prompts with class labels. These prompts consist of a predefined set of class labels denoted as y ∈ {y1, y2, ..., yC }, where C represents the total number of classes. Each prompt typically takes the form of “a photo of a [category]", where “[category]" corresponds to the class label name. Then, the image prediction yˆ corresponding to the class i is obtained by calculating the cosine similarity scores between the image embedding I and the text embedding T, which is formulated as:


![](figures/fileoutpart2.png)

p(ˆy|x) = exp (s (I, Ti) /τ) C c=1 exp (s (I, Ti) /τ) ,

(1)


where s(·) is the similarity metric, τ denotes the temperature parameter. By calculating the softmax probabilities using the similarity scores, CLIP can assign a class label to the image, even if it has not been explicitly trained on that specific class.


Although CLIP has demonstrated impressive zero-shot performance, its integration into specific downstream tasks still requires further refinements through subtle adjustments. Extensive prompt-based


![](figures/fileoutpart3.png)

② Supervised Contrastive Learning Text Encoder {cat} {dog} {bird} {cat} 🔥 Text Batch Maximize Similarity Image-Text Similarity Matrix Text Encoder ❄ 🔥 ① Discriminative Visual-Text Alignment Task All Class Texts ③ Visual-Language Similarity Distillation Image Encoder 🔥 Image BatchImage Encoder ❄ 🔥 : Fine-tuing ❄ : Frozen

①


②


③


Figure 2: The framework of our CLIP-CITE method. CLIP-CITE fine-tunes the whole CLIP model with a discriminative visual-text alignment task and a supervised contrastive loss to enhance the image-text alignment in downstream tasks. Moreover, a vision-language similarity distillation loss incorporates the generalization knowledge of the pre-trained CLIP model into the fine-tuned model.


methods [35, 34, 15] have been proposed to enhance CLIP’s performance in specific contexts. In this study, we investigate the underestimated fine-tuning strategy and propose to improve the fine-tuning method from the perspectives of task designing, multi-modal alignment, and knowledge preservation. As illustrated in Fig. 2, our framework comprises three components, i.e., discriminative visual-text alignment task, supersized contrastive learning, and vision-language similarity distillation.


## 3.1 Discriminative Visual-Text Alignment task


Naive fine-tuning methods for downstream classification tasks typically involve adding a randomly initialized linear classifier on top of the pre-trained visual encoder [23, 18]. The whole model is then fine-tuned using the available domain-specific data for the classification task at hand. However, this training strategy often leads to overfitting on the limited available training data, resulting in poor generalization performance on unseen data.


To address this limitation, we propose to fine-tune the model with a discriminative visual-text alignment task that combines visual-semantic alignment and image classification. Specifically, we connect the similarity scores between the visual and the text embeddings with the probability that the visual image belongs to the class associated with the text embedding, which is formulated:


![](figures/fileoutpart4.png)

p(ˆy|x) = exp (s (θI (x) , θT (ti)))  C c=1 exp (s (θI (x) , θT (tc))) ,

(2)


where s(·) is the consine similarity, θI and θT denotes the visual encoder and text encoder, respectively, ti is the text description of class i, which is obtained in the form of “a photo of a [category]", where “[category]" corresponds to one of the class labels.


Note that Eq. (2) is equivalent to initializing the parameters of the visual classifier W = {wi}iC=0, wi = θT (ti) with the embeddings of the text descriptions of all the available classes and is consistent


with the prediction of the test data. To this end, the objective loss of the discriminative visual-text alignment task is:


![](figures/fileoutpart5.png)

LDV A =−  x∈B log p(ˆy|x),

(3)


where B denotes a training batch during the fine-tuning process. To quickly adapt the model to the target classification task, we freeze the text encoder and take W and θI as the learnable parameter to fine-tune. Through fine-tuning this task, the model acquires the ability to collaboratively associate visual and textual representations, thereby enhancing its capacity to utilize semantic information effectively for the discriminative task.


## 3.2 Supervised Contrastive Learning


To preserve and enhance the representation capability of the pre-trained CLIP, we argue that aligning image and text remains essential, as it corresponds to the task employed in the training of the original CLIP models. However, it is worth noting that aligning images and texts can often require a large batch size, which may not be suitable in situations where data availability is limited.


To mitigate this limitation, we customize an image-text alignment strategy to fine-tune the whole CLIP models (including both θI and θT ) under the limited data regimes. Specifically, we adopt a supervised contrastive loss to align images and texts. Given a pair of data (x, t), where t is derived from the category of x in the form of “a photo of a [category]", the supervised contrastive loss is defined as:


![](figures/fileoutpart6.png)

LSCL =  xi∈B log exp (s (θI (xi) , θT (ti))) tj ∈B It j ̸=xi · exp (s (θI (xi) , θT (tj))) (4) +  ti∈B log exp (s (θT (ti)) , θI (xi)) xj ∈B Ixj ̸=ti · exp (s (θT (ti)) , θI (xj)) ,

where s denotes the cosine similarity, B denotes a training batch, and I denotes the category indicator function. Notably, LSCL can be


considered a special case of FLYP [6] in scenarios where there are no same class instances within the batch, employing unsupervised contrastive loss to optimize image-text alignment.


The designed supervised contrastive loss encourages the model to learn representations that bring similar images and their associated text embeddings closer together while pushing apart images and their non-matching text embeddings. By enforcing this alignment, the model can better capture the semantic relationship between images and their associated text while preserving and enhancing the representation capability of the pre-trained CLIP in specific domains.


## 3.3 Vision-Language Similarity Distillation


While fine-tuning can improve performance on downstream tasks, it would suffer from potential challenges such as catastrophic forgetting and decreased generalization capabilities on the other datasets. To remedy this issue, we introduce a novel vision-language similarity distillation loss to distill the modal consistency from the pre-trained CLIP to the fine-tuned model. Specifically, the vision-language similarity distillation loss is defined as:


![](figures/fileoutpart7.png)

LV LD =  x∈B DKL (p (ˆy|x), ˆ p (ˆy|x)),

(5)


where p (ˆy|x), derived from Eq. (1), is computed using fine-tuned models θI and θT to determine batch cosine image-text similarity scores. While pˆ(ˆy|x), also obtained by Eq. (1), applies the original CLIP models. DKL denotes the Kullback-Leibler divergence. Note that the batch cosine image-text similarity scores undergo normalization through a softmax function to establish a probability distribution.


By minimizing the Kullback-Leibler divergence between the distributions of image-text similarity calculated from the original CLIP encoders and those from the fine-tuned encoders, CLIP-CITE encourages the fine-tuned model to acquire comparable modal alignments and image-text relationship within batch as the pre-trained CLIP models. This strategy upholds modal consistency and facilitates the transfer of knowledge from the pre-trained model to the fine-tuned model.


## 3.4 Final Objective Function


![](figures/fileoutpart8.png)

To fine-tune the whole CLIP models, we combine Eq. (3), Eq. (4), and Eq. (5), obtaining the final objective loss:

![](figures/fileoutpart9.png)

L = LDV A + λ · LSCL + η · LV LD,

(6)


where λ and η are the two hypermeters to balance the items. After the fine-tuning process, we obtain the updated visual encoder θI and text encoder θT .


During inference, we use a weighted ensemble proposed by [32] to combine the fine-tuned model and the pre-trained model:


![](figures/fileoutpart10.png)

ˆθI = α · θI + (1 − α) · θzs I , ˆθT = α · θT + (1 − α) · θzs T , (7)

where α is a hyperparameter. Different from [32] that only considers ensemble in the visual modality, the text encoder in our method is optimized during the fine-tuning process, so the text modality is further considered in this work.


# 4 Experiments


## 4.1 Experiment Settings


To assess the efficacy of our method, we conduct experiments within the few-shot learning paradigm. In this setup, the model under-


goes training using some base classes, each of which is represented by a limited number of samples. Subsequently, the model’s performance is evaluated on the novel classes. Based on the origin of the classes and the domains to which the base and novel data belong, the evaluated tasks are categorized into four distinct groups: few-shot learning (FSL), domain generalization (DG), base-to-new generalization (BNG), and cross-domain generalization (CDG).


FSL. In FSL, the training data and test data are from the same classes and the same domain, which assesses the model’s effectiveness in the limited supervision scenario.


DG. In DG, the training data and test data are from the same classes but in different domains.


BNG. In BNG, the training data and test data are from different classes but in the same domain, which evaluates the model’s ability to generalize to new and previously unseen classes, thereby gauging its open-vocabulary generalization capability.


CDG. In CDG, the training data and test data are from different classes and different domains. In the experiments, the model is trained on the base classes of dataset A and evaluated on the new classes of dataset B.


Dataset Settings. For FSL, BNG, and CDG settings, we use 11 image classification datasets, i.e., ImageNet [4] and Caltech-101 [5] for generic object classification; OxfordPets [22], StanfordCars [17], Flowers [21], Food101 [2], and FGVCAircraft [20] for fine-grained visual categorization, EuroSAT [8] for satellite image classification, UCF101 [28] for action recognition, DTD [3] for texture classification, and SUN397 [33] for scene recognition. We randomly sample 16 images (shots) from each class in all the datasets mentioned above in BNG scinario. For the DG, we treat the ImageNet as the source domain, and the ImageNetV2 [26], ImageNet-Sketch [29], ImageNet-A [10] and ImageNet-R [9] as the target domains for evaluation. Implement Details. In our implementation, we leverage the pre-trained ViT-B/16 model from CLIP [23] for evaluation purposes. We employ the AdamW optimizer, incorporating the cosine annealing strategy to fine-tune our model. The initial learning rate is fixed at 5e-6, while the batch size is set to 32 for most datasets. However, for the EuroSAT dataset, we use a batch size of 16, and for ImageNet, it is increased to 64. The hyperparameters λ, η, and α are consistently set to 0.7, 0.1, and 0.5 across all experiments, respectively. We train our model for 20 epochs. All input images are randomly resized and cropped to a resolution of 224 × 224 pixels. No additional data augmentation techniques are employed, apart from random resizing and cropping. For reproducibility, we report the average results of CLIP-CITE across three distinct random seeds for each experiment.


## 4.2 Performance Comparison


Results of FSL. Fig. 3 presents the average results of four competitors and our CLIP-CITE on the 11 datasets under 1, 2, 4, 8, and 16 shots. From the results, we observe that our CLIP-CITE performs very competitively, especially under 1, 2, and 4 shots. When compared with the second-best competitor MaPLe [15] on the average results, our CLIP-CITE demonstrates performance improvements by 3.42%, 3.00%, 2.48%, 1.73%, and 1.52% in scenarios with 1, 2, 4, 8, and 16 shots, respectively. These gains underscore CLIP-CITE’s effectiveness in generalizing to downstream tasks when provided with limited labeled examples. More comparisons of each dataset are provided in the supplementary materials.


Results of BNG. Table 1 showcases the BNG performance of our CLIP-CITE in comparison to five competing methods: CoOp [35], CoCoOp [34], MaPLe [15], and CLIPood [27]. The accuracy metrics


![](figures/fileoutpart11.png)

12 4 8 16 Number…of…training…shots…per…class 45 50 55 60 65 70 75 80 85 Accuracy…(%) Average…over…11…datasets Linear…probe…CLIP CoOp CoCoOp Maple CLIP-CITE (Ours)

Figure 3: FSL Comparison results of our CILP-CITE and four competitors on the 11 datasets. All of the methods are trained on the ViT-B/16 backbone and implemented with the same experimental settings. We report the average performance of 11 datasets. are reported for both the base classes (B), new classes (N), and their harmonic mean (HM). From the results, we observe that our CLIPCITE performs the best under both B and N metrics on the average of 11 datasets, leading to a notable 2.14% improvement in the HM metric over the second-best competitor. In comparison to the original CLIP model without additional fine-tuning using the base data from the downstream task, our CLIP-CITE demonstrates a remarkable 16.14% increase in base class accuracy across an average of 11 datasets. This indicates that fine-tuning CLIP with the base data could significantly improve the professionalism of CLIP in specific domains, which is also verified by the other competitors. Besides, our CLIP-CITE also obtains 3.86% improvement over CLIP on the novel classes, which indicates that our CLIP-CITE improves the generalization capability under the open-vocabulary scenario. In contrast, while CoOp and CoCoOp also exhibit a notable enhancement in base accuracy, they compromise their capability for generalization to new classes. When compared with the competitors on the specific dataset, our CLIPCITE performs the best on 9 out of 11 datasets in terms of HM metric. Moreover, we have noticed that CLIP-CITE showcases the exceptional performance, particularly on fine-grained datasets, with remarkable results observed on EuroSAT, Cars, Flowers102, and Aircrafts datasets. This leads us to speculate that fine-tuning can acquire finer and more specialized information.


From the BNG results in Table 1, we could conclude that our method could effectively handle the overfitting and catastrophic forgetting issues as it could improve the performances on both base and novel classes from the same domain at the same time.


Results of DG. The DG performances of our method, along with six competitors, are presented in Table 2. In this evaluation, the model is trained on the few-shot ImageNet dataset and then tested on different datasets, namely ImageNetv2, ImageNet-Sketch, ImageNet-A, and ImageNet-R, which have the same class labels as ImageNet but belong to different domains. Our method demonstrates superior performance in terms of in-domain ImageNet accuracy, achieving an accuracy of 72.9%. Additionally, our method achieves a high average accuracy of 60.7% across the out-of-domain datasets, surpassing all existing methods except for ImageNet-A. These results indicate that our method is effective in handling domain shifts.


Results of CDG. The CDG comparison results of our method and three competitors are displayed in Table 3. Specifically, we fine-tune the model with the training data from various datasets and then eval-


Table 1: Comparison performances (%) on BNG task in terms of B, N, and HM metrics. The results of all the competitors are directly from the original literature. The best results are marked in bold.


![](tables/fileoutpart12.png)


Method


CLIP


CoOp


CoCoOp


MaPLe


CLIPood


Ours


Average on


B


69.34


82.69


80.47


82.28


83.90


85.48


N


74.22


63.22


71.69


75.14


74.50


77.08


HM


71.70


71.66


75.83


78.55


78.92


81.06


ImageNet


B


72.43


76.47


75.98


76.66


77.50


78.44


N


68.14


67.88


70.43


70.54


70.30


71.07


HM


70.22


71.92


73.10


73.47


73.72


74.58


Caltech101


B


96.84


98.00


97.96


97.74


98.70


98.82


N


94.00


89.81


93.81


94.36


94.60


94.28


HM


95.40


93.73


95.84


96.02


96.61


96.50


OxfordPets


B


91.17


93.67


95.20


95.43


95.70


96.01


N


97.26


95.29


97.69


97.76


96.40


97.95


HM


94.12


94.47


96.43


96.58


96.05


96.97


Cars


B


63.37


78.12


70.49


72.94


78.60


82.83


N


74.89


60.40


73.59


74.00


73.50


74.51


HM


68.65


68.13


72.01


73.47


75.96


78.45


Flowers102


B


72.08


97.60


94.87


95.92


93.50


95.98


N


77.80


59.67


71.75


72.46


74.50


76.45


HM


74.83


74.06


81.71


82.56


82.93


85.11


Food101


B


90.10


88.33


90.70


90.71


90.70


90.81


N


91.22


82.26


91.29


92.05


91.70


91.55


HM


90.66


85.19


90.99


91.38


91.20


91.18


Aircrafts


B


27.19


40.44


33.41


37.44


43.30


47.26


N


36.29


22.30


23.71


35.61


37.20


38.37


HM


31.09


28.75


27.74


36.50


40.02


42.35


SUN397


B


69.36


80.60


79.74


80.82


81.00


82.30


N


75.35


65.89


76.86


78.70


79.30


79.40


HM


72.23


72.51


78.27


79.75


80.14


80.82


DTD


B


53.24


79.44


77.01


80.36


80.80


84.26


N


59.90


41.18


56.00


59.18


58.60


64.54


HM


56.37


54.24


64.85


68.16


67.93


73.09


EuroSAT


B


56.48


92.19


87.49


94.07


97.50


95.61


N


64.05


54.74


60.04


73.23


64.10


80.59


HM


60.03


68.69


71.21


82.35


77.35


87.46


UCF101


B


70.53


84.69


82.33


83.00


85.70


87.56


N


77.50


56.05


73.45


78.66


79.30


79.01


HM


73.85


67.46


77.64


80.77


82.38


83.07


Table 2: DG performances (%). All methods are trained on the ImageNet and evaluated on ImageNet-V2 (-V2), ImageNet-S (-S), ImageNet-A (-A), and ImageNet (-R).


![](tables/fileoutpart13.png)


Method


In-Distribution ImageNet


Out-of-Distribution


-V2


-S


-A


-R


Aver.


Zero-shot


66.7


60.8


46.1


47.8


74.8


57.2


Fine-tune


68.2


61.9


46.8


46.4


75.1


57.6


CoOp


71.5


64.2


48.0


49.7


75.2


59.3


CoCoOp


71.0


64.2


48.8


50.6


76.2


59.9


MaPLe


70.7


64.1


49.1


50.9


77.0


60.3


CLIPood


71.6


64.9


49.3


50.4


77.2


60.4


CLIP-CITE (Ours)


72.9


65.8


49.6


50.0


77.5


60.7


uate the model on the test data of the ImageNet dataset. For ease of comparison with the results presented in Table 1, we report B and N performance metrics on the ImageNet dataset. From the results, we observe that our CLIP-CITE could maintain its performance on the ImageNet dataset regardless of the datasets used for training, indicating the robustness of the proposed method. In contrast, the performances of the other competitors on ImageNet drop significantly. When considering the results presented in Table 1, we observe that the existing competitors that tuning the CLIP model using a specific dataset, their performance on that dataset notably improves, particularly in terms of the B metric. However, their performance on other datasets significantly declines. This suggests that current competitors enhance their professionalism when fine-tuned with a specific dataset but at the cost of losing their versatility, an issue known as catastrophic forgetting. In contrast, our fine-tuning strategy not only


Table 3: Cross-domain generalization (CDG) evaluation (%). All the models are trained on the base training set of 10 datasets and evaluated on the ImageNet dataset. Note that vanilla CLIP achieves 72.43% and 68.14% in terms of B and N metrics on ImageNet, respectively.


![](tables/fileoutpart14.png)


Method


Caltech101


OxfordPets


Cars


Flowers102


Food101


Aircrafts


SUN397


DTD


EuroSAT


UCF101


B


N


B


N


B


N


B


N


B


N


B


N


B


N


B


N


B


N


B


N


CoOP [35]


55.72


54.62


58.20


50.38


53.72


50.91


46.88


37.43


45.53


42.51


62.02


56.15


52.52


54.59


56.94


55.87


51.81


49.69


42.64


39.76


CoCoOp [34]


70.65


66.97


68.80


59.75


58.54


56.89


50.91


55.78


66.08


62.89


55.81


56.76


69.57


67.28


65.26


62.64


60.51


58.66


51.42


54.64


MaPLe [15]


72.48


69.54


63.91


48.84


68.65


66.03


60.14


47.40


71.84


67.78


67.44


62.66


70.05


67.83


65.56


63.98


40.43


45.80


64.27


62.28


CLIP-CITE


72.24


69.57


72.38


68.44


72.30


69.04


72.06


68.68


72.87


69.03


72.54


68.49


72.33


68.89


71.95


68.44


72.29


68.38


72.81


68.59


![](figures/fileoutpart15.png)

Base New HM Average…on…11…Dataset 75 80 85Accuracy…(%) Base New HM ImageNet 70 75 80Accuracy…(%) ALL ALL FrozenTC FrozenTC FrozenTE FrozenTE

Figure 4: Ablation on different fine-tuning parts of the model. Table 4: Ablation results (%) of our CLIP-CITE with various training objectives on the BNG task of the ImageNet dataset.


![](tables/fileoutpart16.png)


LDV A


LSCL


LV LD


B


N


HM


✓


72.43


68.14


70.22


77.35


69.12


73.00


✓


78.10


70.67


74.20


✓


✓


78.49


70.76


74.43


✓


✓


77.31


70.20


73.58


✓


✓


✓


78.44


71.07


74.58


enhances the CLIP’s professionalism but also maintains its versatility. Furthermore, we observed that as the domain difference between the training dataset and the ImageNet dataset increases, the performance of the model on the ImageNet dataset degrades more severely. For example, the B performance on ImageNet falls from 72.43% to 40.43% when fine-tuning the model with Maple [15] on the EuroSAT dataset. We postulate that this is primarily because the learnable parameters of the parameter-efficient competitors primarily capture domain-and class-specific information, making them less suitable for novel classes from different domains. In contrast, our fully fine-tuning method distributes the changes in domain and category equally across the parameters of the model, resulting in small changes in parameter magnitude, which enables it to effectively handle different domains and categories simultaneously. Furthermore, our distillation strategy also benefits in mitigating catastrophic forgetting.


## 4.3 Further Analysis


Effects of different objectives. Tab. 4 displays the ablation study of our CLIP-CITE with various training objectives on the BNG task of ImageNet. The first row represents the results obtained with the basic CLIP model. When fine-tuning the model with only LDV A, it achieves a 2.78% improvement in HM compared to the naive CLIP. Additionally, the introduction of of supervised contrastive learning objective LSCL leads to further improvement in both B and N metrics. By combining both objectives (LDV A + LSCL), the performance of both B and H metrics continue to improve. Furthermore, incorporating the vision-language similarity distillation loss LV LD into the objective results in the best performance of 74.58% HM accuracy. These experimental outcomes highlight the efficacy of each objective function introduced in this work.


![](figures/fileoutpart17.png)

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 78.5 79.0 79.5 80.0 80.5 81.0 HM…(%) CLIP-CITE MaPLe CLIPood Figure 5: Comparison results with the different ensemble ratio α.

Table 5: Comparison performances (%) and training efficiency of the existing prompt learning methods and ours. All the models are trained on a single NVIDIA GeForce RTX 3090 GPU.


![](tables/fileoutpart18.png)


Method


Iterations


ImageNet


Training Resources


Base


New


HM


Training-time


GPU-usage


CLIP [23]


N/A


72.43


68.14


70.22


N/A


N/A


CoOp [35]


12.5 K


76.47


67.88


71.92


≈1h


≈ 10 G


CoCoOp [34]


80K


75.98


70.43


73.10


>7h


≈ 10 G


MaPLe [15]


10 K


76.66


70.54


73.47


≈ 45 min


≈ 10 G


CLIP-CITE


1.2K


78.44


71.07


74.58


≈ 20 min


≈ 19 G


Effects of the fine-tuning parts. In this experiment, we conduct an ablation study to examine the effects of different fine-tuning parts. The average results of 11 datasets and the results on ImageNet dataset are shown in Fig. 4. FrozenTC indicates that the text embeddings are taken as the classifiers of the visual feature representations and are frozen during optimizing Eq. (3). FrozenTE indicates that the text encoder is frozen during optimizing Eq. (4). ALL indicates that all the parameters of the model are fine-tuning during training. From the results in Fig. 4, we observe that HM performance of ALL witnesses a considerable lift compared with those of FrozenTC and FrozenTE, which concludes that comprehensive fine-tuning enhances model capabilities more effectively than partial fine-tuning.


Effects of the weight ensemble. We investigate the effect of weight ensemble in Fig. 5. The results therein lead us to the conclusion that even without the weight ensemble inference (with α set to 1.0), our method still delivers noteworthy performance with results of 85.79% (B), 73.52% (N), and 79.19% (HM) on the BNG task. Notably, it outperforms CLIPood, which integrates model weight inference ensemble, and MaPLe, by achieving a lift of 0.27% and 0.64% in the HM metric, respectively. Moreover, with the appropriate weight ensemble ratio (setting α to 0.5), we have noticed a notable improvement in both base and novel performance.


Training efficiency. Tab. 5 presents a comprehensive comparison of our CLIP-CITE and four parameter-efficient competitors. The results indicate that parameter efficiency does not necessarily translate to computational efficiency. Specifically, our model, despite fine-tuning more parameters and utilizing more GPU resources, demonstrates superior performance with significantly fewer training iterations and


![](figures/fileoutpart19.png)

0 10 20 30 40 Iterations 0 1 2 3 4 5 6 Value Training…Loss FT-Probe CLIP-CITE… 0 10 20 30 40 Iterations 20 30 40 50 60 70 80 90 100 Training…Accuracy…(%) Training…Accuracy FT-Probe CLIP-CITE… FT-Probe…Test CLIP-CITE…Test

Figure 6: Training loss and accuracy of FT-Probe and CLIP-CITE on the EuroSAT dataset.


shorter overall training time compared to the parameter-efficient competitors. Although the prompt-based methods offer parameter efficiency, they still necessitate the backpropagation of the entire model, along with numerous training iterations, to achieve convergence.


Overfitting analysis. As shown in Fig. 6, the FT-Probe model exhibits a gradual decrease in training loss and a corresponding increase in accuracy on the training set. However, the final test set accuracy is only 60.86% (the blue star), indicating the presence of overfitting. Conversely, our CLIP-CITE model also demonstrates a reduction in the loss function and a consistent improvement in training set accuracy. Notably, it achieves a significantly higher test set accuracy of 95.61%, suggesting that our approach effectively overcomes the issue of overfitting. This underscores the importance of addressing overfit-ting when fully fine-tuning models and demonstrates the effectiveness of our CLIP-CITE method.


Prompt Learning with proposed loss. To evaluate the effectiveness of full-fine-tuning, we also explore the prompt learning methods with our proposed loss. The results, detailed in Table 6, indicate that prompt learning methods experience a modest improvement with the implementation of our proposed loss functions i.e. LSCL and LV LD. Notably, our CLIP-CITE still maintains a performance edge. Besides, with the simple fine-tuning (FT-Probe), the tuned model seems to be overfitting, as shown in Fig. 1. Therefore, we propose that both full fine-tuning and well-designed loss functions are crucial in adapting VLMs to the downstream few-shot tasks.


![](tables/fileoutpart20.png)


Method


LSCL


LV LD


B


N


HM


CLIP


72.43


68.14


70.22


CoOp


76.47


67.88


71.92


CoOp


✓


76.51


67.93


71.97


CoOp


✓


✓


78.23


70.89


72.11


MaPLe


76.66


70.54


73.47


MaPLe


✓


76.70


70.67


73.56


MaPLe


✓


✓


76.71


70.89


73.69


CLIP-CITE


✓


✓


78.44


71.07


74.58


Table 6: Ablation results (%) of our CLIP-CITE and prompt learning with various training objectives on the BNG task of the ImageNet dataset.


The effect of the hyper-parameter λ and η. In Fig. 7, we ablate the different values on λ and η in Eq. (6). From the results, we observe that the performances in terms of HM are better when applying the LSCL, e.g., λ is greater than 0. It indicates that supervised vision-language alignment is necessary when fine-tuning. Besides, the vision-language similarity distillation can regularize the model well when η


is less than 0.1. In the experiments, the optical λ and η are set to 0.7 and 0.1, respectively.


![](figures/fileoutpart21.png)

0 0.2 0.4 0.6 0.8 1 70 72 74 76 78 Base New HM 0 0.2 0.4 0.6 0.8 1 70 72 74 76 78 Base New HM (a) Impacts of λ. (b) Impacts of η.

Figure 7: Impacts (%) of the hyper-parameter λ and η on the BNG performances. We report the results on the ImageNet dataset.


![](figures/fileoutpart22.png)

... 🔥 ❄ Image Encoder 🔥 Batch Images ... 🔥❄ Image Encoder 🔥 Batch Images (a) Fine-tuning previous layers. (b) Fine-tuning late layers.

Figure 8: Illustration of fine-tuned model within the distinct layers. (a) illustrates layers preceding the image encoder, while (b) delineates layers succeeding the image encoder.


![](figures/fileoutpart23.png)

0 2 4 6 8 10 12 70 72 74 76 78 Base New HM 0 2 4 6 8 10 12 68 70 72 74 76 78 Base New HM (a) Freezing the late ith layers. (b) Freezing the previous ith layers.

Figure 9: The effect of the fine-tuning layers. (a) indicates we fine-tune the previous layers and freeze the ith late layers corresponding to Fig. 8. (a), while (b) indicates we freeze the previous ith layers and fine-tune the late layers corresponding to Fig. 8. (b).


The effect of the full fine-tuning. Fig. 8 shows the different fine-tuning manners of the image encoder, e.g. fine-tuning previous layers and fine-tuning late layers. And we conduct the experiments with LDV A for ablation. Fig. 9. (a) shows the results that we fine-tune previous layers and freeze the late layers, while Fig. 9. (b) the results that we fine-tune previous layers and freeze the late layers. From the experimental results, we observe that when there are only a few frozen layers, the performance is comparable to full fine-tuning. However, as the number of frozen layers increases, the effectiveness diminishes, i.e. the last 3 frozen layers led to a decline in the results shown in Fig. 9. (a). Overall, full fine-tuning is better than partial fine-tuning.


# 5 Conclusion


In this paper, we have presented CLIP-CITE, a fine-tuning approach designed to adapt CLIP for downstream tasks in limited-data scenarios.


By devising a discriminative visual-text alignment task, implementing supervised contrastive loss, and employing visual-language similarity distillation, CLIP-CITE effectively addresses the common issues of overfitting and catastrophic forgetting encountered by existing fine-tuning methods. Our experimental results demonstrate that a carefully crafted fine-tuning strategy can enable CLIP to acquire both domain-specific and class-specific knowledge, while maintaining its versatility across other domains and classes. Notably, despite involving the tuning of more parameters, our approach offers superior computational efficiency compared to parameter-efficient prompt-based competitors.


# References

- [1]
- J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, pages 23716–23736, 2022.
- [2]
- L. Bossard, M. Guillaumin, and L. Van Gool. Food-101–mining discriminative components with random forests. In ECCV, pages 446–461, 2014.
- [3]
- M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi. Describing textures in the wild. In CVPR, pages 3606–3613, 2014.
- [4]
- J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009.
- [5]
- L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPR, pages 178–178, 2004.
- [6]
- S. Goyal, A. Kumar, S. Garg, Z. Kolter, and A. Raghunathan. Finetune like you pretrain: Improved finetuning of zero-shot vision models. In CVPR, pages 19338–19347, 2023.
- [7]
- X. Gu, T.-Y. Lin, W. Kuo, and Y. Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv:2104.13921, 2021.
- [8]
- P. Helber, B. Bischke, A. Dengel, and D. Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.
- [9]
- D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, pages 8340–8349, 2021.
- [10]
- D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song. Natural adversarial examples. In CVPR, pages 15262–15271, 2021.
- [11]
- N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly. Parameter-efficient transfer learning for nlp. In ICML, pages 2790–2799. PMLR, 2019.
- [12]
- J. Huang, Y. Li, J. Feng, X. Wu, X. Sun, and R. Ji. Clover: Towards a unified video-language alignment and fusion model. In CVPR, pages 14856–14866, 2023.
- [13]
- C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.H. Sung, Z. Li, and T. Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, pages 4904–4916, 2021.
- [14]
- M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim. Visual prompt tuning. In ECCV, pages 709–727. Springer, 2022.
- [15]
- M. U. Khattak, H. Rasheed, M. Maaz, S. Khan, and F. S. Khan. Maple: Multi-modal prompt learning. In CVPR, pages 19113–19122, 2023.
- [16]
- A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. Segment anything. arXiv:2304.02643, 2023.
- [17]
- J. Krause, M. Stark, J. Deng, and L. Fei-Fei. 3d object representations for fine-grained categorization. In ICCV workshops, pages 554–561, 2013.
- [18]
- A. Kumar, A. Raghunathan, R. Jones, T. Ma, and P. Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. arXiv:2202.10054, 2022.
- [19]
- P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35, 2023.
- [20]
- S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. arXiv:1306.5151, 2013.
- [21]
- M.-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics & Image Processing, pages 722–729, 2008.
- [22]
- O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. Jawahar. Cats and dogs. In CVPR, pages 3498–3505, 2012.
- [23]
- A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748– 8763, 2021.
- [24]
- Y. Rao, W. Zhao, G. Chen, Y. Tang, Z. Zhu, G. Huang, J. Zhou, and J. Lu. Denseclip: Language-guided dense prediction with context-aware prompting. In CVPR, pages 18082–18091, 2022.
- [25]
- H. Rasheed, M. U. Khattak, M. Maaz, S. Khan, and F. S. Khan. Fine-tuned clip models are efficient video learners. In CVPR, pages 6545– 6554, 2023.
- [26]
- B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do imagenet classifiers generalize to imagenet? In ICML, pages 5389–5400, 2019.
- [27]
- Y. Shu, X. Guo, J. Wu, X. Wang, J. Wang, and M. Long. Clipood: Generalizing clip to out-of-distributions. arXiv preprint arXiv:2302.00864, 2023.
- [28]
- K. Soomro, A. R. Zamir, and M. Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv:1212.0402, 2012.
- [29]
- H. Wang, S. Ge, Z. Lipton, and E. P. Xing. Learning robust global representations by penalizing local predictive power. In NeurIPS, 2019.
- [30]
- W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv:2208.10442, 2022.
- [31]
- Z. Wang, J. Yu, A. W. Yu, Z. Dai, Y. Tsvetkov, and Y. Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv:2108.10904, 2021.
- [32]
- M. Wortsman, G. Ilharco, J. W. Kim, M. Li, S. Kornblith, R. Roelofs, R. G. Lopes, H. Hajishirzi, A. Farhadi, H. Namkoong, et al. Robust fine-tuning of zero-shot models. In CVPR, pages 7959–7971, 2022.
- [33]
- J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In CVPR, pages 3485– 3492, 2010.
- [34]
- K. Zhou, J. Yang, C. C. Loy, and Z. Liu. Conditional prompt learning for vision-language models. In CVPR, pages 16816–16825, 2022.
- [35]
- K. Zhou, J. Yang, C. C. Loy, and Z. Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9): 2337–2348, 2022.