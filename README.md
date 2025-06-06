��#   s c r a p e r A I 
 
# scraperAI
# Automated Disorder Information Aggregator & Summarizer (V2 - Development Build)

This Python project automates the process of gathering and summarizing information about various medical disorders. It reads a list of disorders, generates research queries for predefined topics, (currently) simulates web scraping, uses a Large Language Model (Google Gemini) to summarize content, and then compiles these summaries into individual Word documents for each disorder.

**Current Status: Development Build - Web Scraping Bypassed**
Please note: This version of the script **bypasses actual web scraping** and uses **DUMMY data** for search results. This is intended for developing and testing the summarization, symptom extraction, and document generation stages without incurring API costs or dealing with web scraping complexities.

## Features

* **Input from Word Document:** Reads a list of disorders from `disorders.docx`.
* **Dynamic Query Generation:** Creates specific search queries for various topics (therapy, medication, symptoms, diagnosis, related disorders, diet, lifestyle, sports) related to each disorder.
* **Simulated Web Scraping:** Currently uses placeholder/dummy text instead of live web searches with the `browser_use` agent.
* **LLM-Powered Summarization:** Utilizes Google Gemini (via `langchain-google-genai`) to generate comprehensive summaries from the (dummy) scraped content.
* **Symptom Extraction (Placeholder):** Includes a basic keyword-based function to extract potential symptoms from the "symptoms" summary to inform "sports" related queries. *This is a placeholder and needs improvement.*
* **Structured Report Generation:** Creates a `.docx` Word document for each disorder, containing structured summaries for all researched topics.
* **Error Handling:** Improved error handling for file operations, API calls, and data processing.
* **Verbose Logging:** Includes detailed print statements for tracing execution flow and debugging.
* **Asynchronous Operations:** Uses `asyncio` for managing operations.

## Technologies Used

* Python 3.x
* `asyncio`
* `langchain-google-genai` (for Google Gemini LLM)
* `browser_use` (Browser agent - currently bypassed for web scraping)
* `python-docx` (for reading/writing Word documents)
* `python-dotenv` (for managing environment variables)
* `csv` (intended for CSV output, currently non-functional)

## Setup and Installation

### 1. Prerequisites

* Python 3.8 or higher.
* Access to Google Gemini API.

### 2. Clone the Repository (if applicable)

```bash
git clone <https://github.com/unicalp/scraperAI.git>
