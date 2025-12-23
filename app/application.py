import json
import re
import requests
import pdfplumber
from datetime import datetime
from sentence_transformers import SentenceTransformer
import os
import openai

import certifi
import ssl
import chromadb
from chromadb.config import Settings
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import langid
from deep_translator import GoogleTranslator

# Constants and file paths
VECTOR_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("legal_chunks")



# Function to extract text from a PDF using pdfplumber
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Improved intelligent chunking for legal documents
import itertools

def improved_chunking(text, max_length=1500, overlap=500):
    """
    Improved chunking for legal documents:
    - Splits by section/article/heading using regex (Georgian and English legal docs)
    - Keeps section headings with their content
    - Uses tunable overlap
    - Stores section titles as metadata
    """
    # Section/heading patterns (Georgian and English)
    section_patterns = [
        r'(მუხლი\s*\d+)',           # Georgian: მუხლი 1
        r'(თავი\s*\d+)',            # Georgian: თავი 1
        r'(დებულება\s*\d+)',        # Georgian: დებულება 1
        r'(Article\s*\d+)',         # English: Article 1
        r'(Section\s*\d+)',         # English: Section 1
        r'(Clause\s*\d+)'           # English: Clause 1
    ]
    section_regex = re.compile('|'.join(section_patterns), re.IGNORECASE)

    # Split text into sections by headings
    splits = [m for m in section_regex.finditer(text)]
    sections = []
    if splits:
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i+1].start() if i+1 < len(splits) else len(text)
            section_title = match.group(0)
            section_text = text[start:end].strip()
            sections.append((section_title, section_text))
    else:
        # Fallback: treat whole text as one section
        sections = [("Full Document", text.strip())]

    # Now chunk each section by length, keeping heading with each chunk
    chunks = []
    for section_title, section_text in sections:
        # Always prepend the section title to each chunk
        content = section_text
        # Split by paragraphs for more natural boundaries
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', content) if p.strip()]
        # Reconstruct text with paragraphs
        para_text = '\n\n'.join(paragraphs)
        # Sliding window chunking with overlap
        start = 0
        while start < len(para_text):
            end = min(start + max_length, len(para_text))
            chunk_body = para_text[start:end]
            # Prepend section title
            chunk_text = f"{section_title}\n{chunk_body}".strip()
            # Only add meaningful chunks
            if len(chunk_text) > 100:
                chunks.append({
                    'text': chunk_text,
                    'section_title': section_title,
                    'start_pos': start,
                    'end_pos': end
                })
            # Move window with overlap
            if end == len(para_text):
                break
            start = end - overlap
            if start < 0:
                start = 0
    return chunks


# Insert chunk into ChromaDB
def insert_text_with_schema(chunk_id, text, file_name, collection_name="legal_chunks"):
    try:
        embedding = VECTOR_MODEL.encode([text])[0].tolist()
        # print(f"Chunk {chunk_id} with embedding: {embedding}")
        # entities = extract_entities_from_chunk(text)  # DEPRECATED - not used
        entities = {}  # Empty entities for now
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        collection.add(
            ids=[str(chunk_id)],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "file_name": file_name,
                "upload_date": current_date,
                "entities": json.dumps(entities) 
            }]
        )

              
        if any(entities.values()):
            print(f"Chunk {chunk_id} inserted with entities: {json.dumps(entities, indent=2)}")
        else:
            print(f"Chunk {chunk_id} inserted (no entities found)")
        return True
    except Exception as e:
        print(f"Error preparing data for chunk {chunk_id}: {str(e)}")
        return False

# Retrieve relevant chunks from ChromaDB
def retrieve_relevant_chunks(query, top_k=5):
    try:
        query_embedding = VECTOR_MODEL.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"] #ids missing -- testing
        )
        formatted_chunks = []
        for i in range(len(results["documents"][0])):
            formatted_chunks.append({
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # ChromaDB returns distance, so invert for similarity
                "id": results["ids"][0][i],
                "file_name": results["metadatas"][0][i].get("file_name", "unknown")
            })
        return formatted_chunks
    except Exception as e:
        print(f"Error in retrieve_relevant_chunks: {str(e)}")
        return []


# Set OpenAI API key from config
import config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
openai.api_key = config.OPENAI_API_KEY

# Function to extract keywords from a question
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def extract_keywords(question):
    """Improved keyword extraction for Georgian and English questions"""
    # Detect if question is in Georgian
    if re.search(r'[\u10A0-\u10FF]', question):
        # For Georgian, extract meaningful words (avoid very short words)
        words = re.findall(r'[\u10A0-\u10FF]+', question.lower())
        # Filter out very short words and common Georgian stop words
        georgian_stop_words = {'და', 'არ', 'არის', 'არის', 'ეს', 'ის', 'რა', 'როგორ', 'სად', 'როდეს'}
        keywords = [w for w in words if len(w) > 2 and w not in georgian_stop_words]
    else:
        # For English, use existing logic but improve it
        words = re.findall(r'\b\w+\b', question.lower())
        # Add more stop words for legal context
        legal_stop_words = ENGLISH_STOP_WORDS.union({'what', 'when', 'where', 'who', 'which', 'how', 'why', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'})
        keywords = [w for w in words if w not in legal_stop_words and len(w) > 2]
    
    # Remove duplicates and return
    return list(set(keywords))

# Function to find the most relevant context from the PDF text based on keywords
def find_relevant_context(pdf_text, question, top_n=3, window=2000):
    """Improved context retrieval that works like the original but better for complex questions"""
    # Extract keywords more intelligently
    keywords = extract_keywords(question)
    print(f"Extracted keywords: {keywords}")
    
    # For complex questions, also look for related terms
    complex_terms = []
    if any(word in question.lower() for word in ['requirement', 'condition', 'restriction', 'limit', 'მოთხოვნა', 'შეზღუდვა']):
        complex_terms.extend(['requirement', 'condition', 'restriction', 'limit', 'must', 'shall', 'should', 'მოთხოვნა', 'შეზღუდვა'])
    if any(word in question.lower() for word in ['age', 'age limit', 'minimum age', 'ასაკი']):
        complex_terms.extend(['age', 'year', 'minimum', 'limit', 'ასაკი'])
    if any(word in question.lower() for word in ['category', 'type', 'classification', 'კატეგორია']):
        complex_terms.extend(['category', 'type', 'classification', 'class', 'კატეგორია'])
    
    # Combine original keywords with complex terms
    all_keywords = list(set(keywords + complex_terms))
    print(f"All search terms: {all_keywords}")
    
    # Split into paragraphs and score them
    paragraphs = [p.strip() for p in pdf_text.split('\n\n') if p.strip()]
    scored = []
    
    for para in paragraphs:
        para_lower = para.lower()
        # Score based on keyword matches
        score = sum(1 for kw in all_keywords if kw in para_lower)
        
        # Bonus for exact phrase matches
        for term in complex_terms:
            if term in para_lower:
                score += 2
        
        # Bonus for legal terms and section headers
        if any(legal_term in para_lower for legal_term in ['მუხლი', 'article', 'section', 'clause']):
            score += 1
            
        if score > 0:
            scored.append((score, para))
    
    scored.sort(reverse=True)
    print(f"Found {len(scored)} relevant paragraphs")
    
    # Get top paragraphs and expand context around them
    relevant_paragraphs = []
    for score, para in scored[:top_n]:
        idx = pdf_text.find(para)
        start = max(0, idx - window//2)
        end = min(len(pdf_text), idx + len(para) + window//2)
        context_chunk = pdf_text[start:end]
        relevant_paragraphs.append(context_chunk)
        print(f"Selected paragraph with score {score}")
    
    if relevant_paragraphs:
        return "\n\n---\n\n".join(relevant_paragraphs)
    else:
        # Fallback: return beginning of document
        print("No relevant paragraphs found, using document beginning")
        return pdf_text[:window]

# Function to interact with OpenAI API and generate response
def make_openai_request(prompt, model="gpt-4o", max_tokens=1024, system_message="You are a helpful legal assistant."):
    try:
        # Try the new OpenAI client approach first
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            # Fallback to old API if new client not available
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"Error in OpenAI request: {str(e)}")
        return None

# Replace make_ollama_request with make_openai_request in generate_response and process_question
def generate_response(user_question, document_text):
    prompt = f"""
    Contract: {document_text}
    Question: {user_question}
    Provide a detailed, accurate, and well-structured answer based only on the contract content. Quote relevant sections if possible.
    """
    return make_openai_request(prompt)

def detect_user_language(text):
    # Try langid first
    lang, _ = langid.classify(text)
    # If langid fails or is not ka/ru/en, use regex for Georgian or Russian
    if lang not in ['en', 'ka', 'ru']:
        if re.search(r'[\u10A0-\u10FF]', text):
            lang = 'ka'
        elif re.search(r'[А-Яа-яЁё]', text):
            lang = 'ru'
        else:
            lang = 'en'
    return lang

# Robust yes/no classifier by language
def classify_yes_no(user_text: str, lang_code: str) -> str | None:
    """Return 'yes', 'no', or None based on the user's reply and language."""
    text = (user_text or "").strip().lower()
    if not text:
        return None

    yes_terms_en = {"yes", "y", "yeah", "yup", "correct", "accurate", "true", "sure"}
    no_terms_en = {"no", "n", "nope", "incorrect", "not accurate", "false", "nah"}

    yes_terms_ru = {"да", "верно", "правильно", "точно", "соответствует"}
    no_terms_ru = {"нет", "неверно", "неправильно", "неточно", "не соответствует"}

    yes_terms_ka = {"დიახ", "კი", "სწორია", "ზუსტია"}
    no_terms_ka = {"არა", "არასწორია", "ზუსტი არ არის", "სწორი არ არის"}

    if lang_code == 'ru':
        if any(term in text for term in no_terms_ru):
            return 'no'
        if any(term in text for term in yes_terms_ru):
            return 'yes'
        return None

    if lang_code == 'ka':
        if any(term in text for term in no_terms_ka):
            return 'no'
        if any(term in text for term in yes_terms_ka):
            return 'yes'
        return None

    # default English
    if any(term in text for term in no_terms_en):
        return 'no'
    if any(term in text for term in yes_terms_en):
        return 'yes'
    return None

# Function to detect document language
def detect_document_language(document_text):
    """Detect the primary language of the document text"""
    try:
        # Use langid to detect language
        lang, confidence = langid.classify(document_text)
        
        # If langid fails or is not ka/ru/en, use regex for Georgian or Russian
        if lang not in ['en', 'ka', 'ru']:
            if re.search(r'[\u10A0-\u10FF]', document_text):
                lang = 'ka'
            elif re.search(r'[А-Яа-яЁё]', document_text):
                lang = 'ru'
            else:
                lang = 'en'
        
        return lang
    except Exception as e:
        print(f"Error detecting document language: {str(e)}")
        # Fallback: check for Georgian or Russian characters
        if re.search(r'[\u10A0-\u10FF]', document_text):
            return 'ka'
        elif re.search(r'[А-Яа-яЁё]', document_text):
            return 'ru'
        else:
            return 'en'

# Function to translate text using Google Translate
def translate_text(text, target_language):
    """Translate text to target language using Google Translate"""
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        print(f"Error translating text: {str(e)}")
        return text  # Return original text if translation fails


# Function to process question across multiple documents sequentially
def process_question_sequential(user_question, documents, history=[], documents_metadata=[]):
    """Process question using the new Law RAG system if available, otherwise fallback to old system"""
    try:
        # Try to use the new Law RAG system first
        try:
            from law_rag_integration import get_law_rag, query_rag_simple
            from law_rag_integration import CaseDescription
            
            law_rag = get_law_rag()
            if law_rag:
                print("🔄 Using new Law RAG system for question processing")
                
                # Create a case description from the conversation history
                # For now, we'll use a simple approach - in the future this could be enhanced
                clarification_questions = ""
                clarification_answers = ""
                
                # Extract questions and answers from history if available
                if history and len(history) >= 2:
                    # Look for the pattern of questions and answers
                    for i in range(len(history) - 1):
                        if history[i].get('role') == 'assistant' and history[i+1].get('role') == 'user':
                            clarification_questions = history[i].get('content', '')
                            clarification_answers = history[i+1].get('content', '')
                            break
                
                # Detect user language
                user_lang = detect_user_language(user_question)
                
                # Create case description
                case = CaseDescription(
                    clarification_questions=clarification_questions,
                    clarification_answers=clarification_answers,
                    user_query=user_question,
                    user_language=user_lang
                )
                
                # Query the RAG system with timeout handling
                try:
                    result = law_rag.query_with_case_description(case, top_k=5, mode="generative")
                    
                    if result and result.get('success'):
                        print(f"✅ Law RAG system returned answer with {result.get('chunks_found', 0)} chunks")
                        
                        # Format the response to match the expected structure
                        answer = result.get('answer', 'No answer received')
                        citations = result.get('citations', [])
                        
                        # Add citations to the answer if available
                        if citations:
                            citation_text = "\n\nCitations:\n" + "\n".join([f"- {citation}" for citation in citations])
                            answer += citation_text
                        
                        return json.dumps({
                            "status": "success",
                            "answer": answer,
                            "question": user_question,
                            "context_used": f"Retrieved from {result.get('chunks_found', 0)} document chunks",
                            "answer_language": "auto-detected",
                            "user_language": "auto-detected",
                            "document_language_detected": "auto-detected",
                            "search_method": "law_rag_semantic",
                            "chunks_retrieved": result.get('chunks_found', 0),
                            "document_found_in": "multiple documents",
                            "documents_searched": 1,
                            "citation": "; ".join(citations) if citations else "No specific citations"
                        }, indent=4)
                    else:
                        print(f"⚠️ Law RAG system failed: {result.get('error', 'Unknown error') if result else 'No result returned'}")
                        print("🔄 Falling back to old system...")
                except Exception as rag_error:
                    print(f"⚠️ Law RAG system error during query: {str(rag_error)}")
                    print("🔄 Falling back to old system...")
                    
        except ImportError:
            print("⚠️ Law RAG integration not available, using old system")
        except Exception as e:
            print(f"⚠️ Law RAG system error: {str(e)}")
            print("🔄 Falling back to old system...")
        
        # Fallback to old system
        print("🔄 Using old document processing system")
        
        # Detect user's question language
        user_lang = detect_user_language(user_question)
        lang_map = {'en': 'English', 'ka': 'Georgian', 'ru': 'Russian'}
        user_language_name = lang_map.get(user_lang, 'English')
        
        # Process each document sequentially
        for i, document in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}: {document['name']}")
            
            contract_text = document['text']
            document_lang = detect_document_language(contract_text)
            document_language_name = lang_map.get(document_lang, 'English')
            
            # Check if translation is needed
            needs_translation = user_lang != document_lang
            question_to_process = user_question
            
            if needs_translation:
                question_to_process = translate_text(user_question, document_lang)
            
            # Set target answer language
            target_answer_language = user_language_name
            
            # Get not found message in user's language
            not_found_messages = {
                'English': 'not found',
                'Georgian': 'არ არის ნაპოვნი',
                'Russian': 'не найдено'
            }

            print(f"Target answer language: {target_answer_language}")

            not_found_message = not_found_messages.get(target_answer_language, 'not found')
            
            # Try to find answer in current document
            retrieved_chunks = retrieve_relevant_chunks(question_to_process, top_k=5)
            print(f"Retrieved chunks: {retrieved_chunks}")

            semantic_search_quality = False
            
            if retrieved_chunks:
                good_chunks = [chunk for chunk in retrieved_chunks if chunk.get("score", 0) > 0.5]
                if good_chunks:
                    semantic_search_quality = True
                    relevant_context = "\n\n---\n\n".join([chunk["text"] for chunk in good_chunks])
                else:
                    relevant_context = find_relevant_context(contract_text, question_to_process, top_n=3, window=2000)
            else:
                relevant_context = find_relevant_context(contract_text, question_to_process, top_n=3, window=2000)
            
            # Set system message in user's language
            system_messages = {
                'English': "You are an experienced lawyer providing legal analysis. Be confident and authoritative. Always answer in English.",
                'Georgian': "თქვენ ხართ გამოცდილი იურისტი, რომელიც აძლევს იურიდიულ ანალიზს. იყავით თავდაჯერებული და ავტორიტეტული. ყოველთვის უპასუხეთ ქართულად.",
                'Russian': "Вы — опытный юрист, предоставляющий юридический анализ. Будьте уверенным и авторитетным. Всегда отвечайте на русском языке."
            }
            system_message = system_messages.get(target_answer_language, system_messages['English'])
            
            # Create prompt for legal analysis with citation instruction
            prompt = f"""
[LANGUAGE: {target_answer_language}]
Answer ONLY in {target_answer_language}.

IMPORTANT INSTRUCTIONS:
1. You must answer ONLY using information from the provided contract context below
2. If the answer is not present in the context, reply with '{not_found_message}'
3. Do NOT use any outside knowledge or make assumptions
4. For complex questions, break down your analysis step by step
5. Quote specific sections or articles when possible
6. If the question asks about requirements, conditions, or restrictions, be very specific about what you found
7. IMPORTANT: At the end of your answer, add a citation in this exact format:
   "Citation: [document_name] > [article/section reference]"
   For example: "Citation: labour_code.pdf > Article 48, Paragraph 2(b)"
   If you can't find a specific article/section, use: "Citation: [document_name] > General provisions"

CONTEXT:
{relevant_context}

QUESTION: {question_to_process}

Please provide a detailed, accurate answer based only on the above context.
"""
            
            # Get answer from OpenAI
            answer = make_openai_request(prompt, system_message=system_message, max_tokens=1024)
            print(f"Answer: {answer}")
            # Check if answer contains "not found" or similar
            
            if answer and not_found_message.lower() not in answer.lower():
                # Answer found! Translate if needed and return
                if needs_translation and answer:
                    translator = Translator()
                    translated = translator.translate(answer, dest=user_lang)
                    answer = translated.text
                
                # Extract citation if present, otherwise add default citation
                citation = ""
                if "Citation:" in answer:
                    # Citation is already in the answer
                    citation = answer.split("Citation:")[-1].strip()
                else:
                    # Add default citation
                    citation = f"{document['name']} > General provisions"
                    answer += f"\n\nCitation: {citation}"
                
                return json.dumps({
                    "status": "success",
                    "answer": answer,
                    "question": user_question,
                    "context_used": relevant_context[:300],
                    "answer_language": target_answer_language,
                    "user_language": user_lang,
                    "document_language_detected": document_lang,
                    "search_method": "semantic" if semantic_search_quality else "keyword",
                    "chunks_retrieved": len(retrieved_chunks) if retrieved_chunks else 0,
                    "document_found_in": document['name'],
                    "documents_searched": i + 1,
                    "citation": citation
                }, indent=4)
            
            print(f"Answer not found in document {i+1}, continuing to next document...")
        
        # If we get here, no answer was found in any document
        return json.dumps({
            "status": "not_found",
            "answer": not_found_messages.get(user_language_name, 'not found'),
            "question": user_question,
            "documents_searched": len(documents),
            "message": f"Answer not found in any of the {len(documents)} documents."
        }, indent=4)
        
    except Exception as e:
        print(f"Error in process_question_sequential: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "question": user_question
        }, indent=4)

# Update interactive consultation to work with multiple documents
def interactive_legal_consultation(user_question, documents, conversation_history=[]):
    """
    Interactive legal consultation that asks 3 clarifying questions, proposes summary, and provides legal advice.
    """
    try:
        # Detect user's language first
        user_lang = detect_user_language(user_question)
        lang_map = {'en': 'English', 'ka': 'Georgian', 'ru': 'Russian'}
        user_language_name = lang_map.get(user_lang, 'English')

        # Language-specific system messages for higher-quality outputs
        system_messages = {
            'English': (
                "You are an experienced lawyer conducting an initial consultation. You must think and respond only in English, "
                "using professional, confident legal language. Ask one focused question at a time to gather key facts. "
                "Sound like a lawyer who knows exactly what information they need. Be direct and authoritative."
            ),
            'Georgian': (
                "თქვენ ხართ გამოცდილი იურისტი, რომელიც ატარებს პირველად კონსულტაციას. იფიქრეთ და უპასუხეთ მხოლოდ ქართულად, "
                "პროფესიონალური, თავდაჯერებული იურიდიული ენით. დასვით ერთი ფოკუსირებული შეკითხვა, რათა მოაგროვოთ ძირითადი ფაქტები. "
                "ჟღერდით როგორც იურისტი, რომელმაც ზუსტად იცის რა ინფორმაცია სჭირდება. იყავით პირდაპირი და ავტორიტეტული."
            ),
            'Russian': (
                "Вы — опытный юрист, проводящий первичную консультацию. Думайте и отвечайте только на русском языке, "
                "профессиональным, уверенным юридическим стилем. Задавайте один целенаправленный вопрос за раз для сбора ключевых фактов. "
                "Звучите как юрист, который точно знает, какая информация ему нужна. Будьте прямым и авторитетным."
            ),
        }
        system_message = system_messages.get(user_language_name, system_messages['English'])
        
        # SIMPLIFIED TWO-PHASE SYSTEM:
        # Phase 1: Initial consultation (until first comprehensive legal analysis)
        # Phase 2: Follow-up questions (all subsequent messages)
        
        if not conversation_history:
            # Phase 1: First message - start initial consultation
            print("🔄 Phase 1: Starting initial consultation")
            pass  # Continue with clarification questions below
        else:
            # Check if we've already provided a comprehensive legal analysis
            # Look for indicators that a full legal consultation was completed
            final_response_indicators = [
                'conclusion', 'შეჯამება', 'заключение', 
                'disclaimer', 'ეს პასუხი', 'этот ответ',
                'this response is for informational purposes',
                'ეს პასუხი საინფორმაციო მიზნებისთვისაა',
                'этот ответ предназначен только для информационных целей',
                'citation(s)', 'citations:', 'მოქმედებს', 'действует',
                # Indicators for completed legal consultations
                'case assessment', 'legal analysis', 'your rights', 'recommended actions',
                'კეისის შეფასება', 'იურიდიული ანალიზი', 'თქვენი უფლებები', 'რეკომენდებული ქმედებები',
                'оценка дела', 'правовой анализ', 'ваши права', 'рекомендуемые действия',
                '1. case assessment', '2. legal analysis', '3. your rights', '4. recommended actions', '5. conclusion',
                '1. კეისის შეფასება', '2. იურიდიული ანალიზი', '3. თქვენი უფლებები', '4. რეკომენდებული ქმედებები', '5. დასკვნა'
            ]
            
            # Check if any previous message contained a comprehensive legal analysis
            # BUT exclude case summary confirmations from this check
            has_comprehensive_analysis = False
            case_summary_indicators = [
                'is this an accurate description', 'ეს თქვენი მდგომარეობის სწორად აღწერაა',
                'это верное описание вашей ситуации', 'please answer yes or no',
                'გთხოვთ, უპასუხეთ დიახ ან არა', 'пожалуйста, ответьте да или нет'
            ]
            
            for msg in conversation_history:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '').lower()
                    # Skip case summary confirmations - these are NOT comprehensive legal analysis
                    if any(indicator in content for indicator in case_summary_indicators):
                        continue
                    # Check for actual comprehensive legal analysis indicators
                    if any(indicator in content for indicator in final_response_indicators):
                        has_comprehensive_analysis = True
                        break
            
            if has_comprehensive_analysis:
                # Phase 2: Follow-up questions - provide short, direct lawyer responses
                print("🔄 Phase 2: Follow-up question - providing direct lawyer response")
                try:
                    # Build context from previous conversation for the lawyer
                    context_parts = []
                    for msg in conversation_history[-6:]:  # Last 6 messages for context
                        if msg.get('role') == 'user':
                            context_parts.append(f"Client: {msg.get('content', '')}")
                        elif msg.get('role') == 'assistant':
                            context_parts.append(f"Lawyer: {msg.get('content', '')}")
                    
                    context_summary = "\n".join(context_parts)
                    
                    # Create a prompt for short, direct lawyer response
                    if user_language_name == 'Georgian':
                        follow_up_prompt = f"""
თქვენ ხართ გამოცდილი იურისტი, რომელმაც უკვე მისცა სრული იურიდიული კონსულტაცია კლიენტს. 
კლიენტი ახლა სვამს შემდგომ კითხვას და გჭირდებათ მოკლე, პირდაპირი პასუხი.

წინა საუბარი:
{context_summary}

კლიენტის შემდგომი კითხვა: {user_question}

მოგცეთ მოკლე, პირდაპირი პასუხი (2-3 წინადადება). იყავით თავდაჯერებული და ავტორიტეტული. 
არ გამოიყენოთ სრული იურიდიული ანალიზის ფორმატი - მხოლოდ პირდაპირი, პრაქტიკული პასუხი.
"""
                    elif user_language_name == 'Russian':
                        follow_up_prompt = f"""
Вы — опытный юрист, который уже дал полную юридическую консультацию клиенту.
Клиент теперь задает дополнительный вопрос, и вам нужен краткий, прямой ответ.

Предыдущий разговор:
{context_summary}

Дополнительный вопрос клиента: {user_question}

Дайте краткий, прямой ответ (2-3 предложения). Будьте уверенным и авторитетным.
Не используйте формат полного юридического анализа - только прямой, практический ответ.
"""
                    else:
                        follow_up_prompt = f"""
You are an experienced lawyer who has already provided a full legal consultation to your client.
The client is now asking a follow-up question and you need to give a short, direct answer.

Previous conversation:
{context_summary}

Client's follow-up question: {user_question}

Give a short, direct answer (2-3 sentences). Be confident and authoritative.
Do not use the full legal analysis format - just a direct, practical response.
"""
                    
                    # Get the direct lawyer response
                    direct_response = make_openai_request(follow_up_prompt, max_tokens=200, system_message=system_message)
                    
                    return json.dumps({
                        "status": "success",
                        "answer": direct_response,
                        "question": user_question,
                        "user_language": user_language_name,
                        "conversation_history": conversation_history + [{"role": "user", "content": user_question}, {"role": "assistant", "content": direct_response}]
                    }, indent=4)
                    
                except Exception as e:
                    print(f"⚠️ Error generating follow-up response: {str(e)}, falling back to clarification")
                    # Don't reset conversation history - keep context intact
            else:
                # Phase 1: Still in initial consultation - continue with clarification questions
                print("🔄 Phase 1: Continuing initial consultation")
                
                # Check if we're stuck in a loop by looking for repeated case summaries
                # (case_summary_indicators already defined above)
                
                # If the last message is from user, look at the second-to-last for assistant message
                if conversation_history and conversation_history[-1].get('role') == 'user':
                    last_assistant_msg = conversation_history[-2].get('content', '') if len(conversation_history) > 1 else ''
                else:
                    last_assistant_msg = conversation_history[-1].get('content', '') if conversation_history else ''
                
                # Check if the last assistant message was asking for case summary confirmation
                is_asking_for_confirmation = any(indicator in last_assistant_msg.lower() for indicator in case_summary_indicators)
                
                if is_asking_for_confirmation:
                    # Check if this is the same summary being asked again (more than 2 times)
                    summary_count = 0
                    for msg in conversation_history:
                        if msg.get('role') == 'assistant' and any(indicator in msg.get('content', '').lower() for indicator in case_summary_indicators):
                            summary_count += 1
                    
                    if summary_count > 2:  # Allow up to 2 attempts before breaking
                        print("🔄 Detected repeated case summary (3+ times), breaking loop and proceeding to legal analysis")
                        return process_question_sequential(user_question, documents, conversation_history, user_language_name)
                    
                    # If we're asking for confirmation and user responded, handle the yes/no response immediately
                    print("🔄 User responding to case summary confirmation, processing yes/no response")
                    user_lang_code = {'English': 'en', 'Georgian': 'ka', 'Russian': 'ru'}.get(user_language_name, 'en')
                    yn = classify_yes_no(user_question, user_lang_code)
                    
                    if yn == 'yes':
                        print("✅ User confirmed case summary, proceeding to legal analysis")
                        return process_question_sequential(user_question, documents, conversation_history, user_language_name)
                    elif yn == 'no':
                        print("🔄 User rejected case summary, asking for full description")
                        if user_language_name == 'Georgian':
                            full_description_prompt = (
                                "თქვენი პასუხი მიუთითებს, რომ შეჯამება ზუსტი არ არის ან გაურკვეველია. "
                                "გთხოვთ, აღწერეთ თქვენი საქმე თქვენი სიტყვებით. თქვენს ტექსტზე დაყრდნობით დაგეხმარებით."
                            )
                        elif user_language_name == 'Russian':
                            full_description_prompt = (
                                "Ваш ответ указывает, что резюме неточно или недостаточно понятно. "
                                "Пожалуйста, опишите вашу ситуацию своими словами. Я помогу, опираясь на ваш текст."
                            )
                        else:
                            full_description_prompt = (
                                "Your reply indicates the summary is not accurate or unclear. "
                                "Please describe your case in your own words. I will help based on what you write."
                            )

                        response = make_openai_request(full_description_prompt, max_tokens=120, system_message=system_message)

                        return json.dumps({
                            "status": "full_description_needed",
                            "response": response,
                            "user_language": user_language_name,
                            "conversation_history": conversation_history + [{"role": "assistant", "content": response}]
                        }, indent=4)
                    else:
                        print("🔄 Unclear response to case summary, asking for clarification")
                        if user_language_name == 'Georgian':
                            clarification_prompt = "გთხოვთ, უპასუხეთ 'დიახ' ან 'არა' - ეს თქვენი მდგომარეობის სწორად აღწერაა?"
                        elif user_language_name == 'Russian':
                            clarification_prompt = "Пожалуйста, ответьте 'Да' или 'Нет' - это верное описание вашей ситуации?"
                        else:
                            clarification_prompt = "Please answer 'Yes' or 'No' - is this an accurate description of your situation?"

                        return json.dumps({
                            "status": "clarification_needed",
                            "response": clarification_prompt,
                            "user_language": user_language_name,
                            "conversation_history": conversation_history + [{"role": "assistant", "content": clarification_prompt}]
                        }, indent=4)
        
        # Now check if we need to start a new consultation
        if not conversation_history:
            # Initial consultation - use flexible questioning system
            try:
                from law_rag_integration import get_flexible_questions
                response, total_questions = get_flexible_questions(user_question, conversation_history)
                print(f"🤖 Generated {total_questions} flexible questions for case complexity analysis")
            except Exception as e:
                print(f"❌ Error with flexible questioning, falling back to simple approach: {e}")
                # Fallback to simple approach
                if user_language_name == 'Georgian':
                    response = "გთხოვთ, მოგვაწოდოთ მეტი დეტალი თქვენი საქმის შესახებ."
                elif user_language_name == 'Russian':
                    response = "Пожалуйста, предоставьте больше деталей о вашем деле."
                else:
                    response = "Please provide more details about your case."
                total_questions = 3
            
            return json.dumps({
                "status": "clarification_needed",
                "response": response,
                "question_number": 1,
                "user_language": user_language_name,
                "conversation_history": conversation_history + [{"role": "user", "content": user_question}, {"role": "assistant", "content": response}]
            }, indent=4)
        
        elif len(conversation_history) < 20:  # Still in question phase (increased limit to allow for more flexible questioning)
            # Continue asking clarifying questions
            question_number = (len(conversation_history) // 2) + 1
            
            # Determine how many questions we need based on case complexity
            from law_rag_integration import get_flexible_questions
            try:
                _, total_questions_needed = get_flexible_questions(user_question, conversation_history)
            except:
                total_questions_needed = 3  # Fallback to 3 questions
            
            # Safety check: if conversation is getting too long, force progression to case summary
            if len(conversation_history) >= 16:  # 8 question-answer pairs
                print("🔄 Conversation getting long, forcing progression to case summary")
                # Jump to case summary phase
                questions_asked = total_questions_needed
                if user_language_name == 'Georgian':
                    case_summary_prompt = f"""
მომხმარებელმა უპასუხა {questions_asked} გამ уточრებით კითხვას.

საუბრის ისტორია:
{json.dumps(conversation_history, ensure_ascii=False, indent=2)}

თქვენი ამოცანაა — მოკლედ და გასაგებად ჩამოაყალიბოთ მათი საქმის შეჯამება თქვენს სიტყვებით (ზუსტი ფაქტებით, მოკლე აღწერით). შემდეგ ჰკითხეთ: "ეს თქვენი მდგომარეობის სწორად აღწერაა? გთხოვთ, უპასუხეთ დიახ ან არა."

უპასუხეთ მხოლოდ ქართულად.
"""
                elif user_language_name == 'Russian':
                    case_summary_prompt = f"""
Пользователь ответил на {questions_asked} уточняющих вопросов.

История диалога:
{json.dumps(conversation_history, ensure_ascii=False, indent=2)}

Сформулируйте краткое и понятное резюме ситуации своими словами (только факты, краткое описание). Затем спросите: "Это верное описание вашей ситуации? Пожалуйста, ответьте Да или Нет."

Отвечайте только на русском языке.
"""
                else:
                    case_summary_prompt = f"""
The user has answered {questions_asked} clarifying questions.

Conversation history:
{json.dumps(conversation_history, indent=2)}

Provide a concise summary of their situation in your own words (facts only), then ask: "Is this an accurate description of your situation? Please answer Yes or No."

Answer only in English.
"""

                response = make_openai_request(case_summary_prompt, max_tokens=300, system_message=system_message)
                
                return json.dumps({
                    "status": "case_summary",
                    "response": response,
                    "user_language": user_language_name,
                    "conversation_history": conversation_history + [{"role": "assistant", "content": response}]
                }, indent=4)
            
            if question_number <= total_questions_needed:
                # Ask next clarifying question (localized)
                if user_language_name == 'Georgian':
                    next_question_prompt = f"""
თქვენ აგრძელებთ კონსულტაციას.

საუბარი აქამდე:
{json.dumps(conversation_history, ensure_ascii=False, indent=2)}

მოცემული ინფორმაციის საფუძველზე დასვით შემდეგი (#{question_number + 1}) გამ уточრებითი, მოკლე და საქმიანი კითხვა ქართულად (1 წინადადება, მაქსიმუმ 25 სიტყვა). იკითხეთ მხოლოდ ფაქტებზე (თარიღები, მხარეები, გარემოებები) და რა ზუსტად სჭირდება.
"""
                elif user_language_name == 'Russian':
                    next_question_prompt = f"""
Вы продолжаете консультацию.

Диалог до сих пор:
{json.dumps(conversation_history, ensure_ascii=False, indent=2)}

На основе предоставленной информации задайте следующий (#{question_number + 1}) уточняющий вопрос на русском языке — одно предложение, не более 25 слов; спрашивайте только о фактах (даты, стороны, обстоятельства) и о том, что именно требуется.
"""
                else:
                    next_question_prompt = f"""
Continue the consultation.

Conversation so far:
{json.dumps(conversation_history, indent=2)}

Ask the next (#{question_number + 1}) clarifying question in English, one sentence (max 25 words), focused on facts (dates, parties, circumstances) and what exactly is needed.
"""

                response = make_openai_request(next_question_prompt, max_tokens=200, system_message=system_message)
                
                return json.dumps({
                    "status": "clarification_needed",
                    "response": response,
                    "question_number": question_number + 1,
                    "user_language": user_language_name,
                    "conversation_history": conversation_history + [{"role": "assistant", "content": response}]
                }, indent=4)
            
            else:
                # All questions asked, propose case summary (localized)
                questions_asked = total_questions_needed
                if user_language_name == 'Georgian':
                    case_summary_prompt = f"""
მომხმარებელმა უპასუხა {questions_asked} გამ уточრებით კითხვას.

საუბრის ისტორია:
{json.dumps(conversation_history, ensure_ascii=False, indent=2)}

თქვენი ამოცანაა — მოკლედ და გასაგებად ჩამოაყალიბოთ მათი საქმის შეჯამება თქვენს სიტყვებით (ზუსტი ფაქტებით, მოკლე აღწერით). შემდეგ ჰკითხეთ: "ეს თქვენი მდგომარეობის სწორად აღწერაა? გთხოვთ, უპასუხეთ დიახ ან არა."

უპასუხეთ მხოლოდ ქართულად.
"""
                elif user_language_name == 'Russian':
                    case_summary_prompt = f"""
Пользователь ответил на {questions_asked} уточняющих вопросов.

История диалога:
{json.dumps(conversation_history, ensure_ascii=False, indent=2)}

Сформулируйте краткое и понятное резюме ситуации своими словами (только факты, краткое описание). Затем спросите: "Это верное описание вашей ситуации? Пожалуйста, ответьте Да или Нет."

Отвечайте только на русском языке.
"""
                else:
                    case_summary_prompt = f"""
The user has answered {questions_asked} clarifying questions.

Conversation history:
{json.dumps(conversation_history, indent=2)}

Provide a concise summary of their situation in your own words (facts only), then ask: "Is this an accurate description of your situation? Please answer Yes or No."

Answer only in English.
"""

                response = make_openai_request(case_summary_prompt, max_tokens=300, system_message=system_message)
                
                return json.dumps({
                    "status": "case_summary",
                    "response": response,
                    "user_language": user_language_name,
                    "conversation_history": conversation_history + [{"role": "assistant", "content": response}]
                }, indent=4)
        
        else:
            # This should not happen with the new logic, but keep as fallback
            print("🔄 Fallback: Processing response without clear state")
            
            # Safety check: if conversation is extremely long, force progression to legal analysis
            if len(conversation_history) >= 30:  # 15 message pairs
                print("🔄 Conversation extremely long, forcing progression to legal analysis")
                return process_question_sequential(user_question, documents, conversation_history, user_language_name)
            
            # Try to process as a general response
            return process_question_sequential(user_question, documents, conversation_history, user_language_name)
    
    except Exception as e:
        print(f"Error in interactive_legal_consultation: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "question": user_question
        }, indent=4)

# Update continue_legal_consultation to work with multiple documents
def continue_legal_consultation(user_response, documents, conversation_history):
    """
    Continue the interactive legal consultation when user responds to clarifying questions.
    """
    try:
        # Add user's response to conversation history
        updated_history = conversation_history + [{"role": "user", "content": user_response}]
        
        # Continue with the interactive consultation
        return interactive_legal_consultation(user_response, documents, updated_history)
    
    except Exception as e:
        print(f"Error in continue_legal_consultation: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "question": user_response
        }, indent=4)

# Update handle_full_case_description to work with multiple documents
def handle_full_case_description(user_description, documents, conversation_history=[]):
    """
    Handle when user provides full case description after rejecting the summary.
    """
    try:
        # Use sequential document processing
        return process_question_sequential(user_description, documents, conversation_history)
    
    except Exception as e:
        print(f"Error in handle_full_case_description: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "question": user_description
        }, indent=4)
