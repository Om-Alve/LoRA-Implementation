## Overview
This repository contains implementation of Low Rank Adaptation(LoRA) from scratch and experimentation with various fine-tuning approaches for Natural Language Processing (NLP) tasks using PyTorch. 
The main focus is on employing LoRA (Low Rank Adaptation) alongside traditional fine-tuning techniques for efficient model adaptation. 
The experiments are conducted primarily on the IMDb reviews dataset from Stanford, aiming to classify movie reviews as positive or negative.

## Notebooks
1. **last-two-layers.ipynb**: This notebook explores a fine-tuning strategy where only the last two layers of the pre-trained model are updated. The rest of the model remains frozen during training.

2. **lora-from-scratch.ipynb**: Here, LoRA (Low Rank Adaptation) is implemented from scratch in PyTorch. LoRA is applied to the key, value, and the last classifier layer of the pre-trained model, enabling efficient adaptation while retaining model performance.

3. **normal-finetuning.ipynb**: This notebook demonstrates traditional fine-tuning, where all layers of the pre-trained model are fine-tuned on the IMDb reviews dataset.

## Results
### Experiment Details:
- All models were trained for 3 epochs.
- Initial experiments updated only the last two layers while freezing the rest of the model.
![Training the last two layers](/assets/last-two-layers.jpg)
- Subsequently, the entire model was fine-tuned, yielding the highest test accuracy.
![Finetuning the entire model](/assets/full-finetune.jpg)
- Finally, LoRA with rank=4 and alpha=8 was applied to the key, value, and the last classifier layer. This was followed by fine-tuning the entire model with just 0.11% of the total parameters.

![LoRA](/assets/lora.jpg)

### Key Findings:
- **last-two-layers.ipynb**: Limited updates to the last two layers showed some improvement in performance.
- **normal-finetuning.ipynb**: Fine-tuning all layers led to the highest test accuracy, indicating the effectiveness of comprehensive model adaptation.
- **lora-from-scratch.ipynb**: LoRA enabled efficient adaptation with minimal parameters, showcasing promising results with significantly reduced computational overhead.
