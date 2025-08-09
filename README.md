# Multi-Agent Adverse Media Screening System

A multi-agent system for adverse media screening aligned with Financial Action Task Force (FATF) taxonomy. This system automates the identification and categorization of adverse media coverage for entities in financial crime compliance.

---

## 🚀 Overview

This system addresses critical challenges in Know Your Customer (KYC) and Anti-Money Laundering (AML) compliance by providing accurate entity involvement detection and precise classification across six FATF-defined adverse media categories. It uses a modular multi-agent architecture for scalable and robust screening.

---

## 🔑 Key Features

- **Multi-Agent Architecture:** Four specialized agents working in coordination for entity disambiguation, search strategy, classification, and conflict resolution.
- **FATF Compliance:** Classification aligned with international standards for financial crime.
- **Dual-LLM Classification:** Two large language model classifiers for enhanced accuracy and conflict cross-validation.
- **Intelligent Conflict Resolution:** Multi-layer framework including external validation and manual review escalation.
- **Human-in-the-Loop:** Allows manual review triggers for quality assurance.

---

## 🏗️ System Architecture

### Core Agents

| Agent                    | Purpose                                               | Key Features                                             |
| ------------------------ | --------------------------------------------------- | --------------------------------------------------------|
| **Entity Disambiguation**| Accurate entity identification via multi-strategy   | Multi-modal context extraction, DSPy & BERT NER fallback, embedding similarity consolidation |
| **Search Strategy**      | Adaptive multi-strategy searches for adverse media  | Progressive search modes (Broad, Targeted, Deep Dive, Alternative), early termination on clean detection, quality thresholds |
| **Classification**       | FATF taxonomy classification via dual LLMs          | Independent primary & secondary classifiers, confidence scores, batch processing, token budgeting |
| **Conflict Resolution**  | Resolves classifications disagreements               | Five-layer framework, external validation (BraveSearch API), rule-based resolution, manual review escalation |

---

## 📋 FATF Taxonomy Categories

The system classifies adverse media under six FATF categories:

- **Fraud & Financial Crime:** Money laundering, fraud, tax evasion  
- **Corruption & Bribery:** Political corruption, bribery schemes  
- **Organized Crime:** Drug trafficking, arms smuggling  
- **Terrorism & Extremism:** Terrorist financing, extremist activities  
- **Sanctions Evasion:** Violating international sanctions  
- **Other Serious Crimes:** Human trafficking, environmental crimes  

---

## 🛠️ Technologies Used

- **DSPy Framework:** Modular base agents for system design  
- **Large Language Models:** Flexible LLMService interface  
- **BERT-based NER:** Named entity recognition fallback  
- **Sentence Transformers:** Semantic similarity analysis  
- **Tavily API:** Advanced content extraction  
- **BraveSearch API:** External web search validation  

---

## 🖼️ Architecture Diagram

*(Insert architecture flow diagram here to illustrate agent interactions and system workflow)*

---

## ⚙️ Installation

Clone the repository
git clone https://github.com/yourusername/adverse-media-system.git
cd adverse-media-system

Install dependencies
pip install -r requirements.txt

Setup environment variables
cp .env.example .env

Edit .env with your actual API keys
text

---

## 🔑 API Keys Required

| API Key             | Purpose                                | Notes                            |
|---------------------|--------------------------------------|---------------------------------|
| `GROQ_API_KEY`       | Primary LLM services                  | Required                        |
| `BRAVE_SEARCH_API_KEY` | Web search functionality             | Required                        |
| `TAVILY_API_KEY`     | Enhanced content extraction           | Optional but recommended        |
| `TOGETHER_API_KEY`   | Additional LLM options                 | Optional                       |

> **Note:** You will need to register with each external service provider to obtain these API keys.

---

## ⚙️ Configuration

- Configuration is handled via `BaseAgentConfig` and `SearchStrategyConfig`.
- Environment variables load automatically; you can customize:
  - LLM models (primary/secondary)
  - Search strategies and thresholds
  - Embedding and NER models
  - Retry counts, timeouts, and logging levels

---

## 🚀 Usage

Run screening on an entity from the command line:

Screen a single entity by name
python main.py "John Smith"

Screen with additional context to disambiguate
python main.py "John Smith" --context "Software engineer at Google"

text

### Example Output Statuses

- **CLEAN:** No adverse media found  
- **COMPLETED:** Adverse articles found (includes summary and listing)  
- **NEEDS_CONTEXT:** Multiple entity matches found; additional context required to disambiguate  
- **MANUAL_REVIEW:** Conflicting classification results; escalated for human review  

---

## 📝 Sample Output

=== John Smith ===
Status: COMPLETED
Found 5 articles, 2 adverse

Articles found:

CEO Charged with Fraud (adverse)

Company Annual Report (clean)

Tax Evasion Investigation (adverse)
... and 2 more

Summary: Entity has involvement in financial crimes requiring review

text

---

## ⚠️ Limitations & Warnings

- This system **has not undergone comprehensive formal evaluation**.
- Quality of results depends heavily on **external search APIs**.
- **Human oversight is essential** for compliance decisions.
- **Not production-ready** for critical compliance without further validation.

---

## 🤝 Contributing

To contribute:

1. Fork this repository  
2. Create a feature branch: `git checkout -b feature/amazing-feature`  
3. Commit your changes: `git commit -m 'Add amazing feature'`  
4. Push branch: `git push origin feature/amazing-feature`  
5. Open a Pull Request for review

---

## 📄 License

This project is licensed under the **MIT License**. See the [`LICENSE`](./LICENSE) file for details.


