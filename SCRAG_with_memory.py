def CR_mem(query, memory_buffer=None, history=None):
    if memory_buffer is None:
        memory_buffer = []
    if history is None:
        history = []

    llm = OllamaFunctions(model="llama3", temperature=0, format="json")

    relevant_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=query, k=5)
    relevant_docs_content = [doc.page_content for doc in relevant_docs]
    relevant_docs_with_metadata = [{'file_name': list(doc)[1][1]['file_name'], 'page_label': list(doc)[1][1]['page_label']}
                                   for doc in relevant_docs]

    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    reranked_relevant_docs = RERANKER.rerank(query, relevant_docs_content, k=3)
    reranked_relevant_docs_content = [doc["content"] for doc in reranked_relevant_docs]
    result_indices = [item['result_index'] for item in reranked_relevant_docs]
    reranked_relevant_docs_metadata = [relevant_docs_with_metadata[i] for i in result_indices if i < len(relevant_docs_with_metadata)]

    score_prompt = """You are an evaluator tasked with assessing the relevance of a retrieved document to a user's question."""
    matched_relevant_docs = []
    result_indices = []
    for i, doc in enumerate(reranked_relevant_docs_content):
        final_prompt = score_prompt.format(question=query, context=doc)
        score = llm([HumanMessage(content=final_prompt)]).content
        if "yes" in score:
            result_indices.append(i)
            matched_relevant_docs.append(doc)

    metadata = [relevant_docs_with_metadata[i] for i in result_indices if i < len(relevant_docs_with_metadata)]
    context = "\nExtracted documents:\n" + "\n".join([f"Document {i}:::\n" + doc for i, doc in enumerate(matched_relevant_docs)])
    chat_context = "\n".join(memory_buffer)
    
    final_prompt = f"""Using the information contained in the Current conversation...Human: {query}"""
    answer = llm([HumanMessage(content=final_prompt)]).content
    
    memory_buffer.append(f"User: {query}")
    memory_buffer.append(f"Assistant: {answer}")

    return {"query": query, "response": answer, "sources": metadata}, memory_buffer
