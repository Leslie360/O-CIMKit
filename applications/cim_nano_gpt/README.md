# Generative AI (LLM): Nano-GPT on CIM

This application implements a miniature Transformer-based Large Language Model (Nano-GPT) directly executing its multi-layer perceptron (FFN) and final LM-Head mapping on an simulated **Organic Memristive Crossbar Array**.

### Motivation
Demonstrate that the underlying O-CIMKit framework can natively support cutting-edge Generative AI and AIGC workloads, bridging the gap between biological hardware platforms and the bleeding-edge Transformer architecture.

### Results
You can clearly observe the text generation quality degradations caused by physical retention drift (`drift_hours=8760` / 1 year simulation).
