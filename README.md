# Chatbot Application
This repository contains a chatbot application built using Gradio for the user interface and various libraries from the LangChain ecosystem for natural language processing and interaction handling. The application can be used to interact with a pre-trained language model for various conversational tasks.

# Table of Contents
Installation
Usage
Files
Dependencies
Contributing
License

# Installation
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Ensure you have Python installed on your system.

# Usage
To run the chatbot application, execute the following command:

```bash
python app.py
```
This will start the Gradio interface where you can interact with the chatbot.

# Files
### app.py: 
The main application file that initializes and runs the Gradio interface for the chatbot.

### chatbot.py: 
Contains the logic and functions for handling the chatbot's responses.

### requirements.txt: 
Lists all the dependencies required to run the application.

# Dependencies
The application requires the following libraries, as specified in the requirements.txt file:

```plaintext
gradio==4.36.1
huggingface_hub==0.23.4
langchain==0.2.5
langsmith==0.1.77
langchain-core==0.2.7
langchain-chroma==0.1.1
langchain-community==0.2.5
langchain-openai==0.1.8
langchain-text-splitters==0.2.1
python-dotenv==1.0.1
pypdf==3.12.1
```

# Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.
