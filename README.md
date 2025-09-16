
# FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts

<h2 align="center">EMNLP 2025 Findings</h2>

<p align="center">
  <img src=".image/CompareFigure.pdf" alt="Project Workflow" width="700"/>
</p>


## Project Workflow

1. **Data Generation**  
   - Use `FC_Attack/Code/Few_Shot_Prompt` to call GPT with Few Shot Prompt and generate **5,000 daily QA pairs** (harmless questions and answers).  
   - Save the generated dataset as a JSON file for later use.
    
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


