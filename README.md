[![Open in Codeanywhere](https://codeanywhere.com/img/open-in-codeanywhere-btn.svg)](https://app.codeanywhere.com/#https://github.com/Codeanywhere-Templates/langchain)

This is a template project for LangChain applications in [Codeanywhere](https://codeanywhere.com/). [Try it out](https://app.codeanywhere.com/#https://github.com/Codeanywhere-Templates/langchain)


### What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications. This template showcases several core LangChain capabilities including:

- **Agents & Tools**: Dynamic decision-making on which tools to use based on user input
- **Chains**: Combining LLMs with other components for specific workflows
- **Memory**: Maintaining context throughout a conversation
- **Document Loaders**: Extracting content from external sources
- **LLM Integration**: Leveraging OpenAI models for sophisticated language tasks
- **Prompt Templates**: Structured prompting for consistent LLM outputs

## Getting Started

Set your OpenAI API key:
```bash 
export OPENAI_API_KEY=your_api_key_here
```

Open the terminal and run the interactive demo:
```bash
python sample.py
```

All dependencies are pre-installed in the devcontainer.

## Features

- 🔍 **Web Search**: Query information from Wikipedia and DuckDuckGo using LangChain's search tool integrations
- 📄 **Webpage Processing**: Extract and summarize content from any URL with WebBaseLoader and LLM summarization
- 🧠 **Research Notes Generation**: Create structured research notes using LLM chains and prompt engineering
- 🌐 **Knowledge Graph Creation**: Visualize entity relationships with LLM-powered knowledge extraction
- 💬 **Conversational Memory**: Maintain context with ConversationBufferMemory

## Usage Examples

### Web Search
```bash
# Start the demo and then type:
"Search for information about quantum computing"
```

### Webpage Analysis
```bash
# Start the demo and then type:
"Summarize the webpage https://python.langchain.com/docs/get_started/introduction"
```

### Research Notes
```bash
# Start the demo and then type:
"Generate research notes on climate change"
```

### Knowledge Graphs
```bash
# Start the demo and then type:
"Create a knowledge graph about artificial intelligence"
```

### General Conversation
```bash
# The demo maintains context throughout your session:
"What can you tell me about machine learning?"
"How does it compare to deep learning?"
```

## Learn More

To learn more about LangChain, take a look at the following resources:

- [LangChain Documentation](https://python.langchain.com/docs/) - learn about LangChain features and API
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain) - check out the source code
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook) - examples and tutorials
- [LangSmith](https://smith.langchain.com/) - debugging and monitoring platform

You can check out [the official LangChain GitHub repository](https://github.com/langchain-ai/langchain) - your feedback and contributions are welcome!

## Want to contribute?

Feel free to [open a PR](https://github.com/Codeanywhere-Templates/langchain) with any suggestions for this template project 😃