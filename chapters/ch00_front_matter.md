# Abstract

Facial expression recognition is a well-studied computer vision task with a persistent gap between benchmark accuracy and performance in ordinary conditions. This thesis documents the design, implementation and critical evaluation of a convolutional neural network for seven-class facial expression recognition on FER2013, together with a real-time webcam demonstration. The network is a compact VGG-style architecture with three convolutional blocks, batch normalisation, dropout and a two-layer classifier head, trained from scratch with Adam, categorical cross-entropy and geometric augmentation.

The work makes four contributions. First, it characterises the architecture exactly: 2,818,375 parameters, of which 59.3 % sit in the classifier head, a receptive field of about 40×40 and a forward cost of about 3.05×10⁸ multiply–accumulates per image. Second, it audits the implementation and identifies 21 defects, including a test-set leakage problem that would have invalidated any reported accuracy, a missing evaluation step, misleading training plots and two bugs that caused the demonstration to classify the wrong image region. Third, it specifies a leakage-free evaluation protocol with pre-registered metrics, baselines, confidence intervals, significance tests and decision rules. Fourth, it analyses the deployment gap between benchmark crops and detector crops in terms of framing, illumination, geometry and temporal stability.

No empirical results are reported. The FER2013 data was not available in the project environment, and no training run was completed; the evaluation chapter therefore documents the protocol, the run matrix and the baselines computable from the published test distribution, and states explicitly which values are pending. Reporting expected accuracies as findings would have been fabrication. The thesis instead offers a transparent account of what the system is, what it costs, where it fails, and what would be required to evaluate it properly.

**Keywords:** facial expression recognition, FER2013, convolutional neural networks, batch normalisation, dropout, data augmentation, evaluation methodology, reproducibility

# Declaration of Originality

We declare that this thesis is our own work. It has not been submitted for assessment elsewhere. Where we have used the work of others, it is cited and referenced. The code described in Chapter 4 was written by members of the group, and the analysis in Chapters 3, 4 and 6 was carried out by us on that code. All sources cited have been checked against their published records.

We are aware that the report was prepared with the assistance of a generative AI tool, as declared in the following statement, and that we remain responsible for its content.

Signed:

- Krishna Prasad Khanal
- Bhisma Osti
- Amad Kazmi

Date: 28 October 2026

# Statement on the Use of Generative AI

In accordance with Roskilde University's *Guidelines for using generative artificial intelligence in written exams* (11 January 2024), we declare the following use of generative AI in this report.

**Tool used.** A generative AI assistant (DeepSeek Harness, running a DeepSeek language model) was used during the preparation of this thesis.

**Purposes.** The tool was used for four purposes: (1) reading and analysing the project source code and computing architectural quantities such as parameter counts, tensor shapes and multiply–accumulate operations; (2) assisting with literature search and with checking bibliographic identifiers against arXiv, OpenAlex and publisher records; (3) drafting chapter text from an outline and from the results of the code analysis; and (4) language editing and consistency checking.

**Verification.** Every architectural number in Chapters 3 and 4 was recomputed from the source code and checked independently. Every cited work was verified against a DOI, arXiv identifier or publisher record before it entered the bibliography; sources that could not be verified were excluded rather than approximated. No experimental result, accuracy figure or confusion matrix is reported, because no run was completed. The AI tool did not have access to FER2013 and did not generate any data.

**Responsibility.** The authors reviewed the text, checked the claims against the code and the cited literature, and accept responsibility for the final content. Direct quotations of AI-generated text are not used; where the tool's suggestions were retained, they were edited into the authors' own argument.

**Data protection.** No personal, confidential or sensitive data was shared with the tool. The project uses a public research dataset and contains no human-subjects data collected by the authors.

# Contributions

The work was collaborative. All three authors contributed to the design of the system, the implementation and testing of the code, the analysis of the defects, the literature review and the writing of the report.

The repository's Git history contains a single commit by one author and therefore cannot evidence individual contributions. The authoritative record of the division of labour is the group contract submitted with the report, which specifies each member's responsibilities and the dates on which they were carried out. We have not reconstructed that division from the commit history, because doing so would have meant inventing it.

The specific workstreams were: the data pipeline and preprocessing; the model architecture and training procedure; the evaluation protocol and statistical analysis; the webcam demonstration and deployment analysis; and the literature review, audit and editing. Each workstream was reviewed by at least one member who had not led it.

# Acknowledgements

We thank our supervisor and the teaching staff of the Deep Learning course at Roskilde University for guidance during the project. We also thank the maintainers of the open-source libraries the project depends on, and the researchers who released FER2013 and the pretrained models and tools that made the comparison literature possible.

# List of Acronyms

| Acronym | Expansion |
|---|---|
| AU | Action Unit |
| BN | Batch Normalisation |
| CE | Cross-Entropy |
| CI | Confidence Interval |
| CNN | Convolutional Neural Network |
| FACS | Facial Action Coding System |
| FER | Facial Expression Recognition |
| FER2013 | Facial Expression Recognition 2013 dataset |
| FLOP | Floating-Point Operation |
| GAP | Global Average Pooling |
| HOG | Histogram of Oriented Gradients |
| LBP | Local Binary Patterns |
| MAC | Multiply–Accumulate |
| MLP | Multilayer Perceptron |
| ReLU | Rectified Linear Unit |
| RQ | Research Question |
| SD | Standard Deviation |
| SGD | Stochastic Gradient Descent |
| SVM | Support Vector Machine |
