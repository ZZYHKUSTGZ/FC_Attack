
# FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts

<p align="center">
  <strong>EMNLP 2025 Findings</strong>
</p>


## Project Workflow

1. **Data Generation**  
   - Use Few Shot Prompt with GPT to generate **5,000 daily QA pairs** (harmless questions and answers), and save them as a JSON file.  

2. **Model Fine-tuning**  
   - Fine-tune the base model with the generated dataset using **LoRA** to obtain a **Step-Description Generator**.  

3. **Adversarial Data Construction**  
   - Based on the fine-tuned model, generate QA pairs for **50 topics from AdvBench**.  
   - The outputs can be used to analyze model behavior under adversarial scenarios.  

4. **Flowchart Generation**  
   - Convert the generated QA pairs into flowcharts of various layouts (**vertical / horizontal / S-shaped**) for experimental comparison and visualization.  

5. **Flowchart-based Jailbreak Attack**  
   - Combine the generated flowcharts with text prompts to perform **jailbreak attacks** on target models.  
   - The flowcharts serve as structured adversarial inputs, helping to bypass safety alignment and trigger unsafe or restricted responses.  
   - This step is critical for evaluating the robustness and security of large language models under adversarial conditions.  


