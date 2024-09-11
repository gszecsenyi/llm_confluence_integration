# llm_confluence_integration
LLM - Confluence integration

I have developed two methods. 

## AskForConfluence

One is a Pull type, where using a langchain langgraph module, LLM can directly reach the confluence system if it feels it is necessary, and can search and process the answers. 

## ConfluenceQA

The other is a push, where the confluence content is loaded into a Chroma vector database, and from there the LLM can search directly. So there is no need to have a persistent connection between LLM and Confluence. A Gradio interface was also created for this purpose.
