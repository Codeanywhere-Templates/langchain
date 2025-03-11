import os
import warnings
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.box import ROUNDED

# Completely suppress all warnings more aggressively
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Force suppress specific deprecation warnings
original_showwarning = warnings.showwarning
def ignore_warnings(*args, **kwargs):
    pass
warnings.showwarning = ignore_warnings

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY in your .env file")
    print("Create a .env file with content: OPENAI_API_KEY=your-key-here")
    exit(1)

# Setup console for nice output
console = Console()

# Define colors for consistent styling
COLORS = {
    "primary": "cyan",
    "secondary": "green",
    "accent": "yellow",
    "info": "blue",
    "success": "green",
    "warning": "yellow",
    "error": "red"
}

# Import LangChain components - with error output suppressed
with open(os.devnull, 'w') as f:
    stderr_backup = sys.stderr
    sys.stderr = f
    # Import components that might generate warnings
    from langchain.agents import AgentType, initialize_agent, Tool
    from langchain.memory import ConversationBufferMemory
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.documents import Document
    sys.stderr = stderr_backup

# Define knowledge graph generation function
def generate_knowledge_graph(topic: str) -> str:
    """Generate a knowledge graph about a topic."""
    console.print(f"[{COLORS['info']}]Generating knowledge graph for: {topic}[/{COLORS['info']}]")
    
    try:
        # Use LLM to extract entities and relationships
        llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        prompt = ChatPromptTemplate.from_template(
            """From the topic {topic}, identify key entities (people, places, organizations, concepts) 
            and their relationships. Format as a list of triples with the format:
            
            entity1 | relationship | entity2
            
            For example:
            Albert Einstein | developed | Theory of Relativity
            
            Return at least 7 and at most 12 triples that form a coherent knowledge graph about {topic}.
            """
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"topic": topic})
        
        # Process the triples
        triples = []
        for line in result.split('\n'):
            line = line.strip()
            if ' | ' in line and len(line.split(' | ')) == 3:
                head, relation, tail = line.split(' | ')
                triples.append((head.strip(), relation.strip(), tail.strip()))
        
        # Create a text representation of the graph
        graph_representation = "### Knowledge Graph: " + topic + "\n\n"
        
        # Create entity mapping for mermaid (replace spaces with underscores)
        entity_map = {}
        node_counter = 0
        
        for head, _, tail in triples:
            if head not in entity_map:
                entity_map[head] = f"Node{node_counter}"
                node_counter += 1
            if tail not in entity_map:
                entity_map[tail] = f"Node{node_counter}"
                node_counter += 1
        
        # Generate Mermaid diagram
        mermaid_code = "graph TD\n"
        
        # Add node definitions with labels
        for entity, node_id in entity_map.items():
            # Escape quotes in entity names
            safe_entity = entity.replace('"', '\\"')
            mermaid_code += f'    {node_id}["{safe_entity}"]\n'
        
        # Add relationships
        for head, relation, tail in triples:
            # Escape quotes in relation
            safe_relation = relation.replace('"', '\\"')
            mermaid_code += f'    {entity_map[head]} -->|"{safe_relation}"| {entity_map[tail]}\n'
        
        # Add styling
        mermaid_code += "    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;\n"
        mermaid_code += "    classDef concept fill:#d4f1f9,stroke:#0096c7,stroke-width:2px;\n"
        mermaid_code += "    classDef person fill:#ffea00,stroke:#e6a700,stroke-width:2px;\n"
        
        # Apply styling based on entity type (simple heuristic)
        for entity, node_id in entity_map.items():
            if any(person_indicator in entity.lower() for person_indicator in ["dr.", "professor", "mr.", "mrs.", "ms."]):
                mermaid_code += f"    class {node_id} person;\n"
            else:
                mermaid_code += f"    class {node_id} concept;\n"
        
        # Add mermaid code to graph representation
        graph_representation += "```mermaid\n" + mermaid_code + "```\n\n"
        
        # Generate summary statistics
        entities = set()
        relationships = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relationships.add(r)
        
        graph_representation += f"**Entities:** {len(entities)}\n"
        graph_representation += f"**Relationship Types:** {len(relationships)}\n"
        graph_representation += f"**Connections:** {len(triples)}\n"
        
        # Create a nice visualization for the console
        table = Table(title=f"Knowledge Graph: {topic}", box=ROUNDED)
        table.add_column("Entity", style=COLORS["primary"])
        table.add_column("Relationship", style=COLORS["accent"])
        table.add_column("Connected Entity", style=COLORS["secondary"])
        
        for head, relation, tail in triples:
            table.add_row(head, relation, tail)
        
        console.print(table)
        
        # Print mermaid note for visualization
        console.print(f"[{COLORS['info']}]Knowledge graph generated! A visual representation would be shown in a web interface.[/{COLORS['info']}]")
        
        return graph_representation
    
    except Exception as e:
        return f"Error generating knowledge graph: {str(e)}"

# Define web search function
def web_search(query: str) -> str:
    """Search the web for information about a topic."""
    console.print(f"[{COLORS['info']}]Searching for: {query}[/{COLORS['info']}]")
    
    # Initialize search engines
    try:
        wikipedia = WikipediaAPIWrapper(top_k_results=2)
        ddg = DuckDuckGoSearchAPIWrapper()
        
        # Get results
        wiki_results = wikipedia.run(query)
        ddg_results = ddg.run(query)
        
        # Format results
        results = f"### Wikipedia:\n{wiki_results}\n\n### DuckDuckGo:\n{ddg_results}"
        return results
    
    except Exception as e:
        return f"Search error: {str(e)}"

# Define webpage processing function
def process_webpage(url: str) -> str:
    """Extract and summarize content from a webpage."""
    console.print(f"[{COLORS['info']}]Processing webpage: {url}[/{COLORS['info']}]")
    
    try:
        # Load webpage
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        if not docs:
            return f"Could not extract content from {url}"
        
        # Get content and truncate if too long
        content = docs[0].page_content
        if len(content) > 8000:
            content = content[:8000] + "..."
        
        # Use LLM to summarize
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following webpage content:\n\n{content}\n\nProvide a concise summary with key points."
        )
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"content": content})
        
        return f"### Summary of {url}\n\n{summary}"
    
    except Exception as e:
        return f"Error processing webpage: {str(e)}"

# Define research notes function
def generate_research_notes(topic: str) -> str:
    """Generate structured research notes on a topic."""
    console.print(f"[{COLORS['info']}]Generating research notes on: {topic}[/{COLORS['info']}]")
    
    try:
        # Use LLM to generate notes
        llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        prompt = ChatPromptTemplate.from_template(
            """Generate comprehensive research notes on the topic: {topic}
            
            Structure your notes with the following sections:
            1. Overview
            2. Key Concepts
            3. Important Facts
            4. Applications or Implications
            5. Open Questions
            
            Make your notes detailed and educational."""
        )
        chain = prompt | llm | StrOutputParser()
        notes = chain.invoke({"topic": topic})
        
        return f"### Research Notes: {topic}\n\n{notes}"
    
    except Exception as e:
        return f"Error generating research notes: {str(e)}"

def display_welcome():
    """Display welcome message and instructions."""
    console.clear()
    
    title = Text()
    title.append("🔗 ", style=COLORS["accent"])
    title.append("LangChain Demo", style=f"bold {COLORS['primary']}")
    
    welcome_text = """
    Welcome to the LangChain Demo! This simple application showcases some of LangChain's key capabilities:
    
    • 🔍 Web search using Wikipedia and DuckDuckGo
    • 📄 Webpage content processing and summarization
    • 🧠 Research note generation
    • 🌐 Knowledge graph creation
    • 💬 Conversational memory
    
    Type your questions or try these examples:
    - "Search for information about quantum computing"
    - "Summarize the webpage https://python.langchain.com/docs/get_started/introduction"
    - "Generate research notes on climate change"
    - "Create a knowledge graph about artificial intelligence"
    - "What can you tell me about machine learning?"
    
    Type 'exit' to quit, or 'help' to see this message again.
    """
    
    panel = Panel(
        Markdown(welcome_text),
        title=title,
        border_style=COLORS["primary"],
        padding=(1, 2)
    )
    
    console.print(panel)

def main():
    """Run the interactive LangChain demo."""
    display_welcome()
    
    # Redirect stderr during agent creation to hide warnings
    with open(os.devnull, 'w') as f:
        stderr_backup = sys.stderr
        sys.stderr = f
        
        # Define tools
        tools = [
            Tool(
                name="Web_Search",
                func=web_search,
                description="Search the web for information about a topic or question."
            ),
            Tool(
                name="Process_Webpage",
                func=process_webpage,
                description="Process a webpage URL and extract key information."
            ),
            Tool(
                name="Generate_Research_Notes",
                func=generate_research_notes,
                description="Generate structured research notes on a specific topic."
            ),
            Tool(
                name="Generate_Knowledge_Graph",
                func=generate_knowledge_graph,
                description="Create a visual knowledge graph about a topic showing relationships between key concepts."
            )
        ]
        
        # Setup LLM with streaming
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            streaming=True
        )
        
        # Setup memory - the simplest approach
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create agent - using the most reliable approach
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Simple, reliable agent type
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            early_stopping_method="generate"  # Avoids some warnings
        )
        
        # Restore stderr
        sys.stderr = stderr_backup
    
    while True:
        try:
            # Get user input
            console.print(f"\n[{COLORS['primary']}]You:[/{COLORS['primary']}] ", end="")
            # Then get the input without formatting
            user_input = input()

            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print(f"\n[{COLORS['secondary']}]Goodbye! Thanks for trying LangChain![/{COLORS['secondary']}]")
                break
            
            # Check for help command
            if user_input.lower() in ["help", "?"]:
                display_welcome()
                continue
            
            # Process input with agent
            console.print(f"\n[{COLORS['secondary']}]Assistant:[/{COLORS['secondary']}] ", end="")
            
            # Hide warnings during agent invocation
            with open(os.devnull, 'w') as f:
                stderr_backup = sys.stderr
                stdout_backup = sys.stdout
                sys.stderr = f
                
                # Use invoke instead of run to avoid deprecation warning
                # We need to keep stdout for the agent's output
                result = agent.invoke({"input": user_input})
                
                # Restore stderr and stdout
                sys.stderr = stderr_backup
            
        except KeyboardInterrupt:
            console.print(f"\n\n[{COLORS['warning']}]Exiting...[/{COLORS['warning']}]")
            break
        except Exception as e:
            console.print(f"\n[{COLORS['error']}]Error: {str(e)}[/{COLORS['error']}]")
            console.print(f"[{COLORS['info']}]Try a different question or check your setup.[/{COLORS['info']}]")

if __name__ == "__main__":
    main()