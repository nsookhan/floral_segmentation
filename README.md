Automating field based floral surveys with machine learning

Authors: Nicholas Sookhan, Shane Sookhan, Devlin Grewal, J Scott MacIvor

Data for this repository is published on Dryad (https://datadryad.org/stash/dataset/doi:10.5061/dryad.nvx0k6f1t)

Abstract

The abundance and diversity of flowering plant species are important indicators of pollinator habitat quality, but traditional field-based surveying techniques are time-intensive. Therefore, they are often biased due to under-sampling and are difficult to scale. Aerial photography was collected across ten sites located in and around Rouge National Urban Park, Toronto, Canada using a consumer-grade drone. A convolutional neural network (CNN) was trained to semantically segment, or identify and categorize, pixel clusters which represent flowers in the collected aerial imagery. Specifically, flowers of the dominant taxa found in the depauperate fall flowering plant community were surveyed. This included yellow flowering Solidago spp., white Symphyotrichum ericoides/lanceolatum and purple Symphyotrichum novae-angliae. The CNN was trained using 930 m2 of manually annotated data, approximately 1% of the mapped landscape. The trained CNN was tested on 20% of the manually annotated data concealed during training. In addition, it was externally validated by comparing the predicted drone-derived floral abundance metrics (i.e., floral area (m2) and the number of floral segments) to the field-based count of floral units estimated for thirty-four 4 m2 plots. The CNN returned accurate multi-classification when evaluated against the testing data. It obtained a precision score of 0.769, a recall of 0.849 and an F1 score of 0.807. The automated floral abundance counting yielded estimates that were strongly correlated with field-based manual counting. In addition, flower segmentation using the trained CNN was time efficient. On average, it took roughly the same amount of time to segment the flowers occurring in an entire drone scene as it took to complete the abundance count of a single quadrat. However, the training process, particularly manual data annotation, was the most time-consuming component of the study. Overall, the analysis provided valuable insights into automated flower classification and abundance estimation using drone imagery and machine learning. The results demonstrate that these tools can be used to provide accurate and scalable estimates of pollinator habitat quality. Further research should consider diverse wildflower systems to develop the generalizability of the methods.






