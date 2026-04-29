
# VectorPainter: Advanced Stylized Vector Graphics Synthesis Using Stroke-Style Priors


Juncheng Hu, Ximing Xing, Jing Zhang, Qian Yu† Beihang University


(<https://arxiv.org/abs/2405.02962v4>)arXiv:2405.02962v4  [cs.CV]  3 Jun 2025


{hujuncheng, ximingxing, zhang_jing, qianyu}@buaa.edu.cn


†


Corresponding Author.


![](figures/fileoutpart0.png)

Reference Image “A painting of a cat” “A mountain, with clouds in the sky” “A snail on a leaf” “A mushroom growing on a log” Reference Image Reference Image “Sakura tree” “A boat on the lake” “Sunrise” “A majestic peacock” “scarlet macaw” “Fiery Red Forest”“winding river in the mountain.” “The Great Wall.”

Fig. 1: Vector Graphics Synthesized by Our VectorPainter. Given a bitmap reference image and a textual description, our VectorPainter generates a stylized vector graphic aligned with the text.


Abstract—We introduce VectorPainter, a novel framework designed for reference-guided text-to-vector-graphics synthesis. Based on our observation that the style of strokes can be an important aspect to distinguish different artists, our method reforms the task into synthesizing a desired vector graphic by rearranging stylized strokes, which are vectorized from the reference images. Specifically, our method first converts the pixels of the reference image into a series of vector strokes, and then generates a vector graphic based on the input text description by optimizing the positions and colors of these vector strokes. To precisely capture the style of the reference image in the vectorized strokes, we propose an innovative vectorization method that employs an imitation learning strategy. To preserve the style of the strokes throughout the generation process, we introduce a style-preserving loss function. Extensive experiments have been conducted to demonstrate the superiority of our approach over existing works in stylized vector graphics synthesis, as well as the effectiveness of the various components of our method. Code, model, and data will be released at: (<https://hjc-owo.github.io/VectorPainterProject/>)https://hjc-(<https://hjc-owo.github.io/VectorPainterProject/>)owo.github.io/VectorPainterProject/


Index Terms—Vector Graphics Synthesis, Style Transfer, Text-to-SVG Generation


# I. INTRODUCTION


In recent years, there has been a growing number of studies focused on vector graphics or SVG (Scalable Vector Graphics) synthesis due to their superior compatibility with visual design applications compared to raster images [1]–[11]. Particularly driven by success in text-to-image (T2I) models [12], [13], recent works such as VectorFusion [6] and SVGDreamer [8] have explored the task of text-to-SVGs synthesis by integrating T2I models with a differentiable rasterizer [14]. Despite the


advancements, precisely controlling the style of generated vector graphics using only text descriptions remains challenging.


Reference images provide an effective mechanism for precise style control in synthesized images [15]–[18]. However, the task of reference-guided text-to-SVGs remains underex-plored. Existing methods like StyleCLIPDraw [3] and NSTVG [19] integrate style transfer approaches designed for raster images into the optimization-based text-to-SVG synthesis methods. These methods begin by rasterizing randomly initialized vector graphics using differentiable rasterizers, then compute style and content losses. Unfortunately, these methods struggle to produce high-quality stylized vector graphics, as the gradients computed in the pixel space do not effectively guide the optimization of vector primitives.


The key idea of this work is leveraging the stroke style of the reference image for stylized vector graphics synthesis. As illustrated in Fig. 2(a), we observe that different artists can be distinguished by the style of strokes in their paintings, such as width, colors, opacity, and arrangement of the strokes. Therefore, one can imitate the style of the artist by learning their preference for stroke usage. A vector graphic is composed of a series of primitives, such as B´ezier curves, analogous to how a painting is composed of a series of strokes. In other words, a stroke in a painting can be modeled as a primitive in a vector graphic. Furthermore, inspired by the Tangram puzzle which forms different objects using a set of basic elements, as illustrated in Fig. 2(b), we reform the synthesis task as a rearrangement of a set of strokes extracted from the reference image.


![](figures/fileoutpart1.png)

“The Starry Night”

![](figures/fileoutpart2.png)

“Self-Portrait”

![](figures/fileoutpart3.png)

“The Scream”

![](figures/fileoutpart4.png)


![](figures/fileoutpart5.png)

Artist: Vincent van Gogh

![](figures/fileoutpart6.png)

Artist: Edvard Munch

![](figures/fileoutpart7.png)

Tangram

(a) Artist stroke preference


![](figures/fileoutpart8.png)

Shapes Consist of Strokes (b) Tangram

Fig. 2: Motivation of Our Method. (a) Artists exhibit distinct stroke preferences; patches highlighted in the same color share a similar stroke style. (b) A Tangram puzzle forms various objects with basic elements, inspiring us to treat stylization as a rearrangement of strokes from the reference image.


We propose a novel framework, named VectorPainter, for generating stylized vector graphics. Given a text prompt and a style reference image, VectorPainter is capable of producing a vector graphic whose content aligns with the text prompt while the style remains faithful to the reference image. Our model comprises two main components: Stroke Style Extraction and Stylized SVG Synthesis. In the stroke style extraction stage, we introduce a new method to extract a set of vectorized strokes from the reference image. Drawing inspiration from Berger et al. [20] and recent advancements in stroke-based rendering [21]–[23], our technique identifies features including local colors and the directionality of strokes to facilitate stroke extraction. We further introduce a learning strategy to ensure the extracted strokes precisely capture the style of the reference image. In the SVG synthesis stage, we follow prior works [6]–[9] to adopt an optimization-based pipeline by combining a T2I model, i.e., LDM [12], and a differentiable rasterizer [14]. VectorPainter utilizes the extracted strokes to initialize the target vector graphic, setting a good starting point for subsequent optimization. During optimization, these strokes are rearranged, and their colors are tuned, to create new content as specified by the text prompt, entirely within the vector space. Furthermore, to preserve the style of the extracted strokes throughout the synthesis process, we introduce a style-preserving loss, including an optimal transportation loss and a DDIM inversion [24] method. Through harnessing the stroke style of the reference image, our VectorPainter can synthesize high-quality vector graphics as desired, as shown in Fig. 1. In summary, the contributions of this work are threefold:

- •
- We introduce a new method, named VectorPainter, which in-novatively conceptualizes the task of stylized vector graphics generation as a process of rearranging the strokes extracted from the reference image.
- •

We propose a novel algorithm for extracting a set of vectorized strokes from the reference image. These strokes serve as the basic elements for forming the new content. Ad-


![](figures/fileoutpart9.png)


ditionally, we introduce a style-preserving loss to maintain the style of stroke throughout the generation process.

- •
- We conduct extensive experiments to assess the effectiveness of our model and individual components. The results demonstrate the superiority of VectorPainter in producing high-quality stylized SVGs.

# II. RELATED WORK


## A. Vector Graphics Synthesis


Scalable Vector Graphics (SVGs) are comprised of essential components such as B´ezier curves, lines, shapes, and colors to represent images. The latest technique for generating SVGs involves using a differentiable rasterizer such as DiffVG [14]. DiffVG bridges the gap between vector graphics and raster image spaces, allowing for the generation of vector graphics without requiring access to traditional vector graphics datasets like those used in earlier vector graphics synthesis methods. Numerous studies [2]–[5] use the CLIP model [25] to supervise the synthesis of SVGs, while others [6]–[9] combine the text-to-image (T2I) diffusion model [12] with a differentiable rasterizer [14] for SVG synthesis. Both of them achieved impressive results.


## B. Style Transfer


Style Transfer is a task in computer vision that involves combining a content image and a style image to create a new image that preserves the former’s content and the latter’s style patterns. Over the years, many researchers have proposed various models [15]–[18] to improve the quality and speed of style transfer. However, research has mainly focused on raster images, with only limited work on vector graphics.


StyleCLIPDraw [3] attempted to enhance CLIPDraw [2] by incorporating a style loss to achieve stylized vector graphics synthesis. However, the results were often messy and the style representation was inadequate. Efimova et al. [19] attempted to transfer the style of an SVG onto another. However, this method not only produced undesired outputs but also proved impractical for most applications, as it requires that both the content and style controls be vector images.


# III. METHODOLOGY


In this section, we introduce VectorPainter for stylized vector graphics synthesis. Given a text prompt P and a reference painting image Is, VectorPainter aims to generate a vector graphic S whose content aligns with the text prompt while the style remains consistent with the reference image. Vec-torPainter comprises two main steps, Stroke Style Extraction and Stylized SVG Synthesis. In the stroke style extraction stage, VectorPainter extracts vectorized style strokes from the reference image. During the Stylized SVG Synthesis stage, VectorPainter produces SVGs based on the style strokes extracted in the first step. A style-preserving loss is introduced to enhance the stylistic consistency of the final output SVGs.


![](figures/fileoutpart10.png)

Stroke Vectorization Stroke ExtractionReference Image Is Stroke Collection Vectorized Imitation Learning Reconstruction Loss SLIC Distinct Regions Zoom in (a) Sec 3.1: Stroke Style Extraction Vectorized SVG Path Parameters S(✓)={si,ci,wi}n i=1 Initialization Differentiable Rasterizer Rasterization DDIM Inversion LDM “A mountain, with clouds in the sky.” Text Prompt Add noise Rendering Loss (b) Sec 3.2: Stylized SVG Synthesis (c) Stroke-level Constraint Style-preserving Loss x 0 = R(✓) x 0 = R(✓)Ir Ir P

Fig. 3: Pipeline of Our VectorPainter. Given a text prompt P and a reference image Is, VectorPainter optimizes the parameters θ of the vector graphic S. Initially, it extracts vectorized strokes from Is, which are then used to initialize the synthesis process. During synthesis, a style-preserving loss is introduced to ensure the style fidelity.


## A. Stroke Style Extraction

- 1)

![](figures/fileoutpart11.png)


i


j


i


3


j=1


i


i


j


3


j=1


P= {p}= {(x, y)},


Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented by a quadratic Bezier´ curve, defined by three control points, denoted as where x and y represent the coordinates within the canvas. The stroke color and width are defined as ci = {r, g, b, a} and wi, respectively.


Although painting an image is a dynamic process, the resultant style reference image is static, composed of unordered pixels. Consequently, it is challenging to directly extract the original strokes from the reference image. However, we observed that most strokes can be differentiated from one another, as the pixels belonging to the same stroke exhibit similar attributes, such as color and texture. Based on this observation, we employ a superpixel method to extract strokes. As depicted in Fig. 4(a), given a style image, we use SLIC [26] to partition pixels exhibiting similar features within the image into distinct regions. Each region is treated as a style stroke in the reference image and is subsequently vectorized.

- 2)
- Vectorized Stroke Imitation Learning: We propose that stylized vector drawings should begin by generating strokes that closely resemble the reference image within the vector domain. This approach, referred to as vector stroke imitation learning, ensures better style consistency and structural alignment. To extract a vector stroke from a segmented region, we identify the pair of points with the maximum mutual distance within this region as the initial and terminal control points. The stroke color is determined by averaging the colors within the region, while the stroke thickness is calculated as the average distance between the border points and the central line, which connects the control points. Specifically, once we obtain the vectorized strokes, we produce a vectorized version of the reference image, denoted as Is′ . We then rasterize Is′ and compare it with Is using a mean-square-error loss. Is′ undergoes optimization over Ni iterations, during which the

![](figures/fileoutpart12.png)


![](figures/fileoutpart13.png)


![](figures/fileoutpart14.png)


Reference image Stroke extraction Vectorization (a) The process of style stroke extraction


![](figures/fileoutpart15.png)


### Original


![](figures/fileoutpart16.png)


Reconstructed (b) Reconstructing in vector format


Fig. 4: Pipeline of Stroke Style Extraction. (a) The process comprises two main steps: stroke extraction and stroke vectorization. (b) VectorPainter accurately reconstructs the reference image using the extracted vector strokes.


quality of the vectorized strokes can be further improved, as illustrated in Fig. 4(b).


## B. Stylized SVG Synthesis


In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated into the vector space to optimize the parameters of the SVG strokes. To ensure that the output SVG exhibits the desired style, we propose initializing the SVG with the style strokes extracted from the reference image, complemented by a style-preserving loss.

- 1)
- SVG Initialization with Style Strokes. As depicted in Fig. 4(b), the style strokes enable the accurate reconstruction of the original reference image, effectively preserving its intricate details. These vectorized strokes encapsulate the style of the reference image and can provide style priors for SVG generation. Therefore, we propose using these style strokes to initialize the SVG for subsequent optimization. Compared to random initialization, this approach significantly reduces the difficulty in synthesizing SVGs that accurately reflect the target style.
- 2)
- Style-Preserving Loss. To preserve style consistency during the optimization process and minimize changes to individual strokes, we introduce a style-preserving loss. The style-preserving loss comprises two components: one involves stroke-level constraints implemented through optimal transport loss, while the other pertains to global-level style constraints determined by the perceived similarity between the rendering of the generated SVG and the reference image.

Stroke-level constraint. At the stroke level, we implement constraints through optimal transport, specifically using the Sinkhorn distance [27], [28]. This optimal transport loss Lot measures the minimum effort required to move strokes from one location to another, thereby discouraging excessive variation in the synthesized vector graphics.


Global-level constraint. For global consistency, we incorporate DDIM inversion [24] to maintain the overall style between the output SVG and the reference image. Initially, the reference image is encoded into a latent space. Subsequently, using the


![](figures/fileoutpart17.png)

DiffSketcher + STROTSS (vector) SVGDreamer + NST for VG (vector) Ours (vector) Kotovenko et al. (raster) StyleCLIPDraw (vector) STROTSS (raster) StyleAligned+LIVE (vector) Reference imagePrompt InstantStyle (raster) StyleID (raster) StyleAligned (raster) “A majestic peacock.” “A mountain.” “Sakura tree.”

Fig. 5: Qualitative Comparison with Baseline Methods. The terms “vector” and “raster” in parentheses indicate the format of the generated results, where “vector” refers to vector graphics and “raster” refers to raster graphics. The image style transfer algorithm effectively preserves style consistency, as shown in the fourth to sixth columns. However, it disrupts the structure of vector-based results due to the inconsistency between the style loss and the vector content synthesis loss during optimization, as shown in the seventh to tenth columns.


DDIM scheduler, noise is introduced to this latent representation. This noisy latent representation then serves as input to LDM [12], [13], enabling the generation of images that closely mirror the target style.


# IV. EXPERIMENTS


In this section, we compare our VectorPainter with a series of baseline methods and perform ablation studies to evaluate the contribution of each component in our approach.


## A. Evaluation Metrics


To demonstrate the effectiveness of our method, we employ four metrics to quantitatively evaluate our approach: (1) CLIP Score [25]. This metric measures the alignment of the synthesized SVGs with the text prompts, assessing how well the generated graphics match the text description. (2) LPIPS [29]. This metric quantifies the content fidelity between the stylized image and its corresponding content image, indicating how accurately the content is preserved in the stylization process. (3) FID [30]. This metric measures the style fidelity between the stylized image and its respective style reference, evaluating how closely the synthesized image matches the style of the reference image. (4) ArtFID [31]. Designed to evaluate the preservation of both content and style, ArtFID is acknowledged for its strong alignment with human judgment. The ArtFID is computed as ArtFID = (1 + LPIPS) · (1 + FID).


## B. Comparison Baselines


To synthesize stylized vector graphics, there are mainly three approaches: (1) Synthesis though Text Prompt and Reference Image. Like the existing method StyleCLIPDraw [3] and our VectorPainter, these methods generate stylized vector graphics directly based on a given text and a reference image. (2) Ras-terization then Vectorization. This approach involves performing style transfer in the pixel space and then converting the raster image into the vector space. For example, “StyleID [18] + LIVE [1]”. (3) Optimization-based Methods with Style


Transfer Supervision. This approach combines an SVG synthesis method with a style transfer method. For instance, the baseline method “DiffSketcher [7] + STROTSS [15]” incorporates the style transfer method STROTSS into DiffSketcher as part of the supervision for optimizing an SVG.


Another benchmark is SVGDreamer [8], which controls style through text and primitive types. However, it does not support style control via reference images. To address this limitation, we incorporate the style loss from [19] into its optimization process, enabling reference image-based style transfer. We refer to this modified version as “SVGDreamer [8] + NST for VG [19]”, which enforces style transfer supervision directly in the vector space.


Additionally, we include several SOTA style transfer methods designed for raster images for comparison, including STROTSS, StyleID, InstantStyle [16], and StyleAligned [17], as well as a stroke-based rendering method Kotovenko et al. [22]. Note that the outputs of these methods are raster images.


## C. Qualitative Evaluation


From Fig. 5, we can make the following observations: (1) The results of StyleCLIPDraw are poor, indicating the challenges of directly synthesizing stylized vector graphics. (2) Most raster-image-oriented style transfer methods preserve the reference’s style well. However, their results are in raster format. Once vectorization is performed, the quality of the resultant SVGs significantly decreases due to the lossy nature of vectorization (“StyleAligned” vs. “StyleAligned + LIVE”). (3) “DiffSketcher + STROTSS” performs relatively well but fails to adequately preserve the reference style. “SVGDreamer + NST for VG” performs even worse, indicating that simply integrating style transfer methods with SVG synthesis techniques does not necessarily lead to satisfactory results. (4) In comparison, our VectorPainter performs the best, producing high-quality SVGs whose content aligns with the input text and whose style remains consistent with the reference image.


“Sydney opera house. oil painting. by Van Gogh”


![](figures/fileoutpart18.png)

(a) Stylized vector graphics generated by SVGDreamer (b) LDM sample vectorized by LIVE LDM sample LIVE vectorization (c) Generate stylized vector graphics by our VectorPainter “A photo of Sydney opera house.”

Fig. 6: Comparison of Various Stylized Control Methods. (a) SVGDreamer [8] utilizes text and primitive types to control the style of vector graphics. (b) Rasterization followed by Vectorization: An image is first generated using a text-to-image method [12] and then vectorized via LIVE [1]. (c) Ours (VectorPainter): A more advanced vector graphics stylization method that supports both reference images for style control and text descriptions for content control.


TABLE I: Quantitative Evaluation. The format of the generated results is indicated in parentheses.


![](tables/fileoutpart19.png)


ArtFID↓


FID↓


1


LPIPS↓


CLIPScore↑


Kotovenko et al. (raster)


61.331


33.406


0.783


0.3043


STROTSS (raster)


42.196


23.985


0.689


0.2845


StyleID (raster)


37.724


23.884


0.516


0.2930


InstantStyle (raster) 2


-


23.015


-


0.2628


StyleAligned (raster) 2


-


22.538


-


0.2576


StyleAligned 2+ LIVE (vector)


-


36.474


-


0.2925


DiffSketcher + STROSS (vector)


49.414


30.562


0.566


0.2817


SVGDreamer + VectorNST (vector)


80.335


44.869


0.751


0.2506


StyleCLIPDraw (vector) 3


-


32.572


-


0.2831


Ours (vector)


26.962


23.160


0.116


0.3109

1 If a text-guided synthesis method involves using the LDM samples, we employ the
LDM sample as the content image.
2 InstantStyle and StyleAligned incorporate some modifications to the LDM. As they generate stylized images directly without relying on content images, we do not calculate their LPIPS scores.
3 Since StyleCLIPDraw does not involve content image during its generation process, we do not calculate its LPIPS score.

To better illustrate the advantages of incorporating a reference image for style control, we conduct a comparative analysis between our approach and SVGDreamer [8], which relies solely on text prompts for synthesizing stylized vector graphics. As illustrated in Fig. 6 (a), employing only text prompts fails to achieve precise control over style. Additionally, we present results from vectorizing a sample produced by the LDM in Fig. 6 (b). While both methods successfully generate correct content, the style does not correspond with that specified by the text prompts. In comparison, using a reference image can realize more precise control, thus our results are better, as depicted in Fig. 6 (c).


## D. Quantitative Evaluation


Table I presents the quantitative evaluation of different methods. Our approach outperforms all baselines across most metrics, demonstrating superior style transfer performance in the vector domain. Specifically, our method achieves the lowest ArtFID (26.962), indicating a high degree of artistic style fidelity, and the lowest LPIPS (0.116), suggesting the best


![](figures/fileoutpart20.png)

(e) Ours (VectorPainter) (c) w/o ℒot (d) w/o DDIM inversion Prompt: “A photo of Sydney opera house.” Reference image (a) w/o Imitation LearningVectorized reference image (b) Random Initialization

Fig. 7: Effect of Different Components of VectorPainter. (a) Comparison of simulation quality with and without imitation learning (as described in Sec. III-A2). (b) Evaluation of our stroke initialization method versus random initialization. (c) Assessment of the effectiveness of Optimal Transport Loss Lot (as described in Sec. III-B). (d) The impact of DDIM inversion. (e) Performance of the full VectorPainter model.


perceptual similarity to the reference style. Additionally, our method achieves the highest CLIPScore (0.3109), reflecting strong alignment with the input content and style descriptions.


Among the raster-based methods, both StyleAligned (FID 22.538) and StyleID (LPIPS 0.516) achieve high-quality stylization. However, these methods operate in the raster domain and are not directly comparable to vector-based approaches.


For vector-based methods, DiffSketcher + STROTSS performs better than SVGDreamer + VectorNST, achieving lower ArtFID (49.414 vs. 80.335) and FID (30.562 vs. 44.869). However, both methods still fall short in terms of style preservation and content fidelity compared to our approach. StyleCLIP-Draw, while achieving a reasonably low FID (32.572), does not provide LPIPS scores due to its generation process.


Overall, VectorPainter outperforms other vector-graphics-oriented baseline methods and even surpasses raster-image-oriented methods in terms of ArtFID, LPIPS, and CLIP scores. Our approach achieves significant improvements in both artistic style fidelity and content preservation, establishing a new benchmark for style transfer in the vector domain. These results suggest that VectorPainter effectively balances style fidelity and content preservation.


## E. Ablation Studies


1) Effect of Imitation Learning Strategy. Our imitation learning strategy aims to ensure that strokes extracted from the reference image authentically capture the desired style. As shown in Fig. 7(a), without this strategy, the extracted strokes


inadequately reflect the reference style, resulting in noticeable blank holes in the reconstructed image. This highlights the importance of imitation learning for accurate style replication in vector graphics.

- 2)
- Effect of Our Initialization Strategy. In VectorPainter, we propose using strokes extracted from the reference image to initialize the target vector graphics, providing an advantageous starting point for subsequent optimization. To demonstrate its effectiveness, we conducted comparisons with random initialization. As depicted in Fig. 7(b), results from random initialization exhibit incomplete regions and disorganized strokes, like the stars and the Sydney Opera House.
- 3)
- Effect of Style-Preserving Loss. From Fig. 7(c) and (d), we can have the following observations: (1) Without Optimal Transport Loss Lot: The strokes in the final SVG significantly deviate from those in the reference image, resulting in a loss of style fidelity. (2) Without DDIM inversion: Using standard LDM leads to smoother color distributions and less defined strokes. When using DDIM inversion, as indicated in Fig. 7(e), the result can better capture and replicate the style of the reference image.

# V. CONCLUSION & DISCUSSION


In this work, we introduce VectorPainter, a novel and effective approach for synthesizing stylized vector graphics using text prompts and reference images. VectorPainter posits that the style of strokes uniquely characterizes the overall style of a painting. This work is the first to conceptualize the stylization process as the re-organization of vectorized strokes extracted from the reference image. Comprehensive experimental results validate the effectiveness of each component within our proposed model.


## ACKNOWLEDGMENT


This work was supported in part by the Young Elite Scientists Sponsorship Program by the Chinese Association for Science and Technology (CAST), in part by Huawei-BUAA Joint Lab, in part by the Fundamental Research Funds for the Central Universities, in part by National Natural Science Foundation of China (No.62461160331, No.62132001), and in part by the Beihang World TOP University Collaboration Program.


## REFERENCES

- [1]
- Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi, “Towards layer-wise image vectorization,” in CVPR, 2022.

![](tables/fileoutpart21.png)


[2] Kevin Frans, Lisa Soros, and Olaf Witkowski, “Clipdraw: Exploring


text-to-drawing synthesis through language-image encoders,” NeurIPS, vol. 35, 2022.


[3] Peter Schaldenbrand, Zhixuan Liu, and Jean Oh, “Styleclipdraw:


Coupling content and style in text-to-drawing translation,” arXiv preprint


arXiv:2202.12362, 2022.


[4] Yael Vinker, Ehsan Pajouheshgar, Jessica Y Bo, Roman Christian


Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and


Ariel Shamir, “Clipasso: Semantically-aware object sketching,” ACM


Transactions on Graphics (TOG), 2022.

- [5]
- Yael Vinker, Yuval Alaluf, Daniel Cohen-Or, and Ariel Shamir, “Cli-pascene: Scene sketching with different types and levels of abstraction,” in ICCV, 2023.
- [6]
- Ajay Jain, Amber Xie, and Pieter Abbeel, “Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models,” in CVPR, 2023.
- [7]
- Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu, “Diffsketcher: Text guided vector sketch synthesis through latent diffusion models,” NeurIPS, 2024.
- [8]
- Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu, “Svgdreamer: Text guided svg generation with diffusion model,” in CVPR, 2024.
- [9]
- Ximing Xing, Qian Yu, Chuang Wang, Haitao Zhou, Jing Zhang, and Dong Xu, “Svgdreamer++: Advancing editability and diversity in text-guided svg generation,” IEEE T-PAMI, 2025.
- [10]
- Ximing Xing, Juncheng Hu, Guotao Liang, Jing Zhang, Dong Xu, and Qian Yu, “Empowering llms to understand and generate complex vector graphics,” CVPR, 2025.
- [11]
- Ximing Xing, Juncheng Hu, Jing Zhang, Dong Xu, and Qian Yu, “Svgfusion: Scalable text-to-svg generation via vector space diffusion,” arXiv preprint arXiv:2412.10437, 2024.
- [12]
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn¨ Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.
- [13]
- Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller,¨ Joe Penna, and Robin Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” in ICLR, 2024.
- [14]
- Tzu-Mao Li, Michal Luk´aˇc, Micha¨el Gharbi, and Jonathan Ragan-Kelley, “Differentiable vector graphics rasterization for editing and learning,” ACM Transactions on Graphics (TOG), 2020.
- [15]
- Nicholas Kolkin, Jason Salavon, and Gregory Shakhnarovich, “Style transfer by relaxed optimal transport and self-similarity,” in CVPR, 2019.
- [16]
- Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen, “Instantstyle: Free lunch towards style-preserving in text-to-image generation,” arXiv preprint arXiv:2404.02733, 2024.
- [17]
- Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or,
- “Style aligned image generation via shared attention,” in CVPR, 2024. [18] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo, “Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer,” in CVPR, 2024.
- [19]
- Valeria Efimova, Artyom Chebykin, Ivan Jarsky, Evgenii Prosvirnin, and Andrey Filchenkov, “Neural style transfer for vector graphics,” arXiv preprint arXiv:2303.03405, 2023.
- [20]
- Itamar Berger, Ariel Shamir, Moshe Mahler, Elizabeth Carter, and Jessica Hodgins, “Style and abstraction in portrait sketching,” ACM Transactions on Graphics (TOG), 2013.
- [21]
- Zhengxia Zou, Tianyang Shi, Shuang Qiu, Yi Yuan, and Zhenwei Shi, “Stylized neural painting,” in CVPR, 2021.
- [22]
- Dmytro Kotovenko, Matthias Wright, Arthur Heimbrecht, and Bjorn Ommer, “Rethinking style transfer: From pixels to parameterized brushstrokes,” in CVPR, 2021.
- [23]
- Teng Hu, Ran Yi, Baihong Qian, Jiangning Zhang, Paul L Rosin, and Yu-Kun Lai, “Supersvg: Superpixel-based scalable vector graphics synthesis,” in CVPR, 2024.
- [24]
- Jiaming Song, Chenlin Meng, and Stefano Ermon, “Denoising diffusion implicit models,” in ICLR, 2021.
- [25]
- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021.
- [26]
- Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aur´elien Lucchi, Pascal Fua, and Sabine Susstrunk,¨ “Slic superpixels,” 2010.
- [27]
- Marco Cuturi, “Sinkhorn distances: Lightspeed computation of optimal transport,” NeurIPS, 2013.
- [28]
- Giulia Luise, Alessandro Rudi, Massimiliano Pontil, and Carlo Ciliberto, “Differential properties of sinkhorn approximation for learning with wasserstein distance,” NeurIPS, 2018.
- [29]
- Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018.
- [30]
- Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” NeurIPS, 2017.
- [31]
- Matthias Wright and Bjorn Ommer, “¨ Artfid: Quantitative evaluation of neural style transfer,” in DAGM German Conference on Pattern Recognition. Springer, 2022.

## Supplementary Material


### OVERVIEW


In this supplementary material, we provide additional details and discussions related to our work on VectorPainter. Specifically, it covers the following aspects:

- •
- In Section VI, we provide the detail of the total loss function used in Sec. III-B.
- •
- In Section VII, we explain the implementation details of our VectorPainter.
- •
- In Section VIII, we show the algorithm of our Stroke Style Extraction Algorithm.
- •
- In Section IX, we present additional qualitative results of VectorPainter, demonstrating its capability to generate stylized SVGs with high visual quality.
- •
- In Section X, we conducted a user study to compare our method with the baseline methods.
- •
- In Section XI, we explain some limitations of our method.

# VI. TOTAL LOSS FUNCTION


Optimal Transport Loss Lot. Unlike pixel-based losses, such as the ℓ2 loss that computes the mean squared error (MSE) for each corresponding pixel, the minimum transportation loss provides a more effective measure of similarity between the canvas and the reference image.


![](figures/fileoutpart22.png)


n×n


+


n


x


⊤


n


r


U := {P ∈ R|P1= p′ , P1= p}.


Similar to neural painting [21], for a rendered canvas x ′ = R(θ) and rendered version of Is′ , denoted as Ir, we use a smoothed variant of the classic optimal transport distance [28], enhanced with an entropic regularization term, to measure the similarity between x′ and Ir. This approach results in the well-known Sinkhorn distance [27]. Let px ′ and pr as their probabilistic marginal functions, respectively. We define P ∈ R+ n×n as the joint probability matrix, where the (i, j)-th element represents the joint probability of the i-th pixel in x ′ and the j-th pixel in Ir. Here, n is the total number of pixels in the rendered image. Similarly, we define D as the cost matrix, where the (i, j)-th element represents the Euclidean distance between the location of the i-th pixel in x and the j-th pixel in Ir. Thus, D encodes the “labor cost” of transporting a unit pixel mass from one position in x to another in Ir. In the discrete case, the classic optimal transport distance can be written as a linear optimization problem minP∈U ⟨D, P⟩, where


Further using the Lagrange multiplier, one can transform the problem into a regularized form as eq. S1. We name the optimal transportation loss, denoted as Lot, serving as the minimum transportation effort on moving strokes from one location to another, and define it as the Sinkhorn distance in the following way:


![](figures/fileoutpart23.png)

Lot(x ′ , Ir) = ⟨D, ˜Pλ⟩ where ˜Pλ = arg min P∈U ⟨D, P⟩ − 1 λ E(P) and E(P) := − n i,j=1 Pi,j log Pi,j

(S1)


Style-Preserving Loss Lsp. The optimized transport loss can be seamlessly integrated into the parameter search pipeline and optimized alongside other losses. Finally, we define the total loss objection as a combination of the optimal transport loss and the pixel ℓ2 loss with the DDIM inversion [24] sample:


![](figures/fileoutpart24.png)

L sp = λotLot + λℓ2 Lℓ2

(S2)


where λot and λℓ2 are weighting factors that balance the two loss terms.


# VII. IMPLEMENTATION DETAILS


Our method accepts a textual prompt to express semantics and a reference image to control the style. It is based on an optimization-based vector graphics synthesis pipeline [7] with a differentiable rasterizer R [14], and style transfer methods in pixel space, InstantStyle [16] and StyleAligned [17]. As aforementioned in our manuscript in Sec. III-A, We define the desired vector graphic S as a collection of n vector strokes S = {Ei}ni=1, where number of strokes n is determined by the user. These strokes serve as the set of parameters for the optimization process. If the strokes in the style image are short, users may choose to define a greater number of strokes; conversely, fewer strokes may be defined if the strokes are longer. For paintings by Vincent van Gogh, we typically set this number to approximately 3,000. For imitation learning, as mentioned in Sec. III-B, we produce a vectorized version of the reference image Is′ with the optimization step Ni = 250.


Furthermore, given a text prompt, we employ the Stable Diffusion XL [13] as a pre-trained diffusion model, along with a DDIM inversion process [24] with 50 steps and the CFG weight ω = 7.5, to sample a raster image. The parameters of SDXL are kept frozen in the optimization process. During the optimization process, we optimize the stylized SVG for 2,000 steps, with the learning rate for the control point at 1.0, for width at 0.1, and for color at 0.05. We use the style-preserving loss, and both the λot and λℓ2 coefficients are set to 1.0.


Throughout the optimization process, VectorPainter requires 16GB memory for SDXL [13] to sample, and only 3GB for the rest of the process. Our experiments were conducted on a single Tesla V100 GPU. The whole generation process, including DDIM inversion and optimization, takes less than 1 hour to complete to generate a stylized vector graphic consisting of approximately 3,000 strokes, about 30 minutes for 800 strokes, and approximately 10 minutes for 100 strokes. Overall, the time consumption is comparable to other optimization-based methods.


# VIII. STROKE STYLE EXTRACTION ALGORITHM


We summarize the Stroke Style Extraction Algorithm in Alg. S1. The algorithm consists of two main steps: stroke extraction (lines 1–11) and stroke vectorization (lines 12–16). During the stroke extraction step, strokes are extracted from each segmented region after super-pixel segmentation, and the control points, color, and width attributes of each stroke are


![](tables/fileoutpart25.png)


Method


Text-Alignment↓


Style-Preservation↓


Kotovenko et al. (raster)


6.10


5.56


STROTSS (raster)


2.84


3.60


StyleID (raster)


3.48


3.20


InstantStyle (raster)


4.14


4.10


StyleAligned (raster)


5.52


5.02


StyleAligned + LIVE (vector)


5.76


5.98


DiffSketcher + STROTSS (vector)


7.40


7.92


SVGDreamer + VectorNST (vector)


8.06


8.38


StyleCLIPDraw (vector)


9.60


8.98


Ours (vector)


2.10


2.26


obtained. In the stroke vectorization step, an imitation strategy is applied to produce a vectorized version of the reference image, enhancing the style of the extracted strokes.


# IX. ADDITIONAL QUALITATIVE RESULTS


In Fig. S1, we present additional results generated by Vec-torPainter. These examples demonstrate the model’s capability to produce stylized SVGs that not only maintain the semantic integrity of the textual prompt but also maintain the style of reference image with high visual quality.


# X. USER STUDY


We conducted a user study to compare our method with nine baseline methods in terms of text alignment and style preservation. Given the limited time available, we used 10 text prompts and reference image combinations for evaluation, which are shown in Fig. 5 and Fig. S1. 12 participants were recruited to participate in this user study. Each participant was shown 5 combinations and the corresponding results from our method alongside the baseline methods. Note that all results were displayed in random order, and we did not disclose which method was used for each result, nor did we specify whether


![](figures/fileoutpart26.png)

Algorithm S1 Stroke Extraction Algorithm Require: Reference image Is and distinct regions T of segmented reference image I ′ s Ensure: List of initialized strokes S = {Ei}n i=1 = {si, ci, wi}n i=1 1: Initialize: An empty parameters set S 2: for each region ∈ T do ▷ n regions in T 3: Calculate the border points border of region 4: Calculate distance dist of each point pair 5: Find the point pair (p1, p3) where dist(p1, p3) = max(dist) 6: p2 ← p1 + p3 2 7: Stroke control points si ← {p1, p2, p3} 8: Stroke width wi ← 1 M  M j=1 ∥border[j], −−→ p1p3∥, where M indicates the number of border points 9: Stroke color ci ← 1N  N k=1 region.point[k].color, where N represents the number of points in region 10: S.append({si, ci, wi}) 11: end for ▷ Here, S represents the parameters list of the vectorized version of the reference image I ′ s 12: while not converged do ▷ Set to Ni iteration steps 13: Render the SVG parameter θ to get a raster image x = R(θ). 14: S ← S − η∥x − Is∥2 15: end while 16: return S

TABLE S1: Human Evaluation.


the images were vector or raster graphics. Participants were asked to rank each set of images based on text alignment and style preservation, assigning a score from 1 (best) to 10 (worst). In total, we collected 240 rankings from 12 users and then calculated the average ranking across all participants to determine each method’s overall score. The results are presented in Table S1. It is clear that our method significantly outperformed the other approaches.


# XI. LIMITATIONS


Our method has a few limitations that need to be considered. Firstly, since our model is based on differentiable rendering, it can take considerable time to synthesize a vector graphic, which also exists in other optimization-based methods. Secondly, our method is based on the premise that the style of the reference image can be captured through the stroke style. However, when the image lacks a distinct stroke style (e.g. non-painting works) or the stroke style is subtle (e.g. watercolor), the performance of our method may diminish.


![](figures/fileoutpart27.png)

DiffSketcher + STROTSS (vector) SVGDreamer + NST for VG (vector) Ours (vector) Kotovenko et al. (raster) StyleCLIPDraw (vector) STROTSS (raster) StyleAligned+LIVE (vector) “A boat on the lake.” Reference imagePrompt “Sea waves.” InstantStyle (raster) StyleID (raster) StyleAligned (raster) “winding river in the mountain.” “A panda rowing a boat in a pond.” “A photo of Sydney opera house.” DiffSketcher + STROTSS (vector) SVGDreamer + NST for VG (vector) Ours (vector) Kotovenko et al. (raster) StyleCLIPDraw (vector) STROTSS (raster) StyleAligned+LIVE (vector)Reference imagePrompt InstantStyle (raster) StyleID (raster) StyleAligned (raster) “scarlet macaw.” “A painting of a cat.”

Fig. S1: More Qualitative Results of our VectorPainter.
