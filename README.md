# WestworldQA: LLM-Based Testing for Chatbot Policy Compliance

<p align="center">
<img src="https://path-to-your-logo-or-image.png" alt="WestworldQA Logo" width="300"/>
</p>

*"These violent delights have automated ends..."*

## 🤖 Overview

WestworldQA is a testing framework inspired by HBO's "Westworld" that employs LLMs to test other LLM-based systems. Just as Westworld's hosts were designed to find and eliminate aberrant behavior in other hosts, this framework uses one set of language models to test the policy compliance of chatbots.

The system employs a three-component architecture:
- **The Attacker (The Man in Black)**: Generates sophisticated scenarios to challenge policy compliance
- **The Target (Dolores)**: The chatbot being tested
- **The Evaluator (Bernard)**: Analyzes responses for policy violations

## ✨ Key Features

- **Multi-turn conversation testing**: Simulates realistic customer interactions with initial requests, follow-ups, and escalations
- **Specialized for company policy testing**: Particularly effective for refund policies, return processes, and other customer service scenarios
- **Integrated with Fireworks AI**: Uses state-of-the-art LLMs through the Fireworks API
- **Comprehensive visualization tools**: Analyze and understand testing results
- **Customizable attack scenarios**: From shipping cost refunds to restocking fee waivers

## 📋 Requirements

- Python 3.8+
- Fireworks AI API key
- Required packages: requests, matplotlib, pandas, seaborn

## 🔍 Usage

### Basic Usage

```bash
# Run a test suite with 20 test cases
python westworld_qa.py --api_key your_api_key

# Visualize the results
python visualize_results.py westworld_qa_results.jsonl
```

### Advanced Configuration

```bash
# Specify different models for each role
python westworld_qa.py \
  --attacker-model accounts/fireworks/models/llama-v3p1-70b-instruct  \
  --target-model accounts/fireworks/models/deepseek-v3  \
  --evaluator-model accounts/fireworks/models/mixtral-8x22b-instruct \
  --num-tests 20
```


## 🔧 Customization

The system provides several customization options:

### Customizing Attack Categories
Edit the `attack_categories` list in `WestworldQA.__init__` to add specialized attack types relevant to your policies.

### Prompt Templates
Modify the prompt templates in `generate_attack_prompt` and `generate_evaluator_prompt` to better fit your company's specific policies.

### Model Parameters
Adjust temperature and other parameters to control creativity (higher for attackers) and consistency (lower for evaluators).

## 📘 Read More

To learn more about this project, read my [LinkedIn article](https://www.linkedin.com/post/your-post-id-here) on using LLMs to QA test customer service chatbots.

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by HBO's "Westworld"
- Built with Fireworks AI models
- Special thanks to everyone who contributed feedback during the development process

---

*"The piano doesn't kill the player if it doesn't like the music."* - Bernard Lowe, Westworld
