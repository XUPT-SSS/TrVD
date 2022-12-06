# TrVD: Deep Semantic Extraction via AST Decomposition for Vulnerability Detection
The explosive growth of software vulnerabilities poses a serious threat to the system security and has become one of the urgent
problems of the day. Yet, existing vulnerability detection methods show limitations in reaching the balance between the detection ac-
curacy, efficiency and applicability. To this end, this paper proposes TrVD (abstract syntax Tree decomposition based Vulnerability
Detector), which exposes the indicative semantics implied deeply in the source code fragments for accurate and efficient vulnera-
bility detection following a divide-and-conquer strategy. To ease the capture of subtle semantic features, TrVD converts the AST
of a code fragment into ordered sub-trees of restricted sizes and depths with a novel decomposition algorithm. The semantics of
each sub-tree can thus be effectively collected with a carefully designed tree-structured neural network. Finally, a Transformer-style
encoder is utilized to summarize them up into a dense vector, with learning additional long-range semantics relationships among
the sub-trees and distilling the semantics that are more informative to pin down the vulnerable patterns.

##Source
###Step1:Code normalization
Normalize the code with normalization.py
```
python ./normalization.py
```
###step2: AST decomposition
Train word2vec embedding and Decompose AST with our algotithm.
```
python pipeline.py
```
###Step3: Train TrVD vulnerability detector
```
python train.py
```
