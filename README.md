<div align="center">
    <h3>Aristotle University of Thessaloniki</h3>
    <h1>Analysis of Medical Images MedMNIST with CNN, Transfer Learning & Vision Transformers (PyTorch)</h1>
    <h3>Deep-Learning Winter Sem. 2025-2026</h3>
    <p>
        <b>Zisis Katsaros - 10666</b><br>
        <small>Prof.: P. Petrantonakis</small>
    </p>
</div>

## Overview
This project includes training medical image classifiers on OrganSMNIST dataset. Following models are trained and compared to each other: CNN built from scratch, ResNet18 and DeiT utilizing transfer learning (feature extraction and fine-tuning). 

## Results
The best performing model was ResNet18 with fine-tuning of the last two blocks, achieving an ACC of $79.35\%$ and an AUC of $0.9765$. The CNN that was trained from scratch came close, achieving up to $78.79\%$ ACC and $0.9771$ AUC. The DeiT vision transformer model had poor performance, as a result of the small dataset.

## Documentation
The extensive documentation of this project can be found [here](docs.md). Although it is in Greek, there is a handful of tables and figures so non-Greek-speaking readers can still get a sense of the project.


