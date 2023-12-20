# ASC23-LLM inference optimization

The dataset and baseline code for ASC23 LLM inference optimization challenge. 

## Challenge description

This competition focuses on the LLM inference optimization, which requires participating teams to build an inference engine based on LLaMA-70B to achieve high throughput on the 10,000-sample dataset provided by the ASC24 Committees. 

## Dateset

The dataset used in the preliminary has 10k samples with prompts and prompt length. This dataset has the following characteristics:

+ Multi-domain Coverage: The dataset contains text data from various domains, including news, encyclopedias, novels, forums, and more, and covering different topics, styles, and viewpoints, enabling the model to have better generalization across different domain tasks.

+ Multilingual Support: The dataset includes text data from multiple languages such as English, Chinese, Korean, Spanish, etc., which allows the model to understand and generate text across different languages.

+ Large-scale Data: The dataset is sampled from a massive amount of text data, which helps improve the language understanding and generation capabilities of the model.

+ Length Diversity: Too long and too short sequences are filtered out. The dataset contains 10k samples with length range from 4 to 1024, covering a vast majority of the length range for everyday use.

In summary, the dataset provides high-quality language samples with multi-domain coverage, multilingual support, large-scale data, and diversity characteristics.

## Baseline code

The ASC24 committees supply a baseline code to benchmark throughout and total tokens. And participants could start from it and modified it for high inference performance.

Usage：After download the dataset one can use the following example to run the script. More info can be obtained using `-h`.

The parameter `--num-samples` is only used for test. The participants should test with the whole 10k dataset for this challenge.

```bash
CUDA_VISIBLE_DEVICES=0 python baseline.py --dataset /your_data_path/scrambled_sampled_dataset.json --model /your_model_path/hf_model_weights --num-samples=10
```

Besides, the model weight of LLaMA2-70B can be downloaded from: https://huggingface.co/meta-llama/Llama-2-70b or https://huggingface.co/meta-llama/Llama-2-70b-hf.

## Contact
email info@asc-events.org
