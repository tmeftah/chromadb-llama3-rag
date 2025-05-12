import threading
from fastapi import FastAPI, HTTPException,Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List,Optional
import time
from datetime import date
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import gc
import re

app = FastAPI()

class CustomInternalError(Exception):
    pass

class QuestionRequest(BaseModel):
    question: str

class CustomerData(BaseModel):
    """Pydantic model for customer and offer data."""
    customer: str 
    customer_number: int 
    offer: int 
    created_on: date 
    working_days: float
    context: str

class AnswerResponse(BaseModel):
    answer: str
    reranked_texts: List[CustomerData]
class AufwandrechnerModule:
    _instance: Optional['AufwandrechnerModule'] = None # Type hint
    _lock = threading.Lock() # Lock for thread-safe access to instance and timer
    _initialized = False # Flag to ensure __init__ runs only once on the instance

    _models_loaded = False # Flag to track if models are currently loaded
    _last_accessed_time: float = 0 # Timestamp of the last access
    _timer: Optional[threading.Timer] = None # Stores the timer thread
    # Configure the idle timeout duration in seconds
    IDLE_TIMEOUT_SECONDS = 30 * 1 # Example: 5 minutes

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Another check inside the lock in case multiple threads
                # waited and passed the first check simultaneously
                if cls._instance is None:
                    cls._instance = super(AufwandrechnerModule, cls).__new__(cls)
                    # Initialize last accessed time immediately upon instance creation
                    cls._instance._last_accessed_time = time.time()
        return cls._instance

    def __init__(self,
                 model_name="unsloth/Llama-3.2-3B-Instruct",
                 embed_model_name="sentence-transformers/all-mpnet-base-v2",
                 collection_name="unique"):
        
        if self._initialized:
            return
        
        # Increment the counter each time a new instance is created
        print("--- Initializing AufwandrechnerModuleSingleton ---")
        

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model_name = embed_model_name
        self.model_name = model_name
        self.collection_name = collection_name

        self.embed_tokenizer = None
        self.embed_model = None
        self.model = None
        self.tokenizer = None
        self.qa_model = None
        self.collection = None
        self.reranker = None

        settings = Settings(is_persistent=True)
        self.chroma_client = chromadb.Client(settings=settings, tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE)

        try:
            self.load_models()
            self._initialized = True # Set flag after successful initialization
            print("--- AufwandrechnerModuleSingleton fully initialized and models loaded ---")
        except Exception as e:
            print(f"--- Error initializing AufwandrechnerModuleSingleton: {e} ---")
            # Optionally, clear partially loaded models if error occurs
            self.clear_models() # Ensure no partial models are left
            # Consider if you want the app to fail startup or handle this later
            # For critical models, failing startup might be better.
            # If this error occurs, subsequent requests relying on this instance will fail.
            # For this example, we'll just print the error and the flag won't be set.
            # The model loading might be attempted again implicitly by dependent methods if not handled.
            # Better to raise the error to prevent using a broken instance
            raise RuntimeError("Failed to initialize AufwandrechnerModuleSingleton and load models.") from e

    def _schedule_clear_timer(self):
            """Schedules or resets the timer to clear models after IDLE_TIMEOUT_SECONDS."""           
            
            # Cancel existing timer if any
            if self._timer and self._timer.is_alive():
                self._timer.cancel()
                print("Cancelled existing clear timer.")

            # Schedule a new timer
            print(f"Scheduling clear timer for {self.IDLE_TIMEOUT_SECONDS} seconds...")
            self._timer = threading.Timer(self.IDLE_TIMEOUT_SECONDS, self._perform_clear_check)
            self._timer.daemon = True # Allow the application to exit even if timer is running
            self._timer.start()
            
           


    def _perform_clear_check(self):
            """Called by the timer. Checks if models should be cleared."""
            print("_perform_clear_check")
           
            # Calculate idle time
            idle_duration = time.time() - self._last_accessed_time
            print(f"Timer check: last accessed {idle_duration:.2f} seconds ago (timeout is {self.IDLE_TIMEOUT_SECONDS}).")

            # If idle duration is greater than or equal to the timeout AND models are loaded
            if idle_duration >= self.IDLE_TIMEOUT_SECONDS and self._models_loaded:
                print("Timeout reached and models are loaded. Performing clear.")
                self._clear_models_internal() # Use internal clear method
            else:
                # If not idle enough, or models weren't even loaded, or another request activated it
                # The next access will reschedule the timer anyway.
                # No need to reschedule from here unless we want a continuous check (less efficient)
                # The current approach is request-driven timer scheduling.
                pass # Timer finishes; it will be rescheduled on the next request if models are loaded.



    # ... rest of AufwandrechnerModule methods (load_models, clear_models, etc.)
    def load_models(self):
        """Loads all necessary models and the database collection if not already loaded."""
        with self._lock:
            if self._models_loaded:
                 print("Models already loaded, skipping load.")
                 self._last_accessed_time = time.time() # Update access time even if already loaded
                 self._schedule_clear_timer() # Reschedule timer on activity
                 return # Models are already loaded, nothing to do

            print("Loading models...")
            try:
                # Perform the actual loading if models are not marked as loaded
                self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
                self.embed_model = AutoModel.from_pretrained(self.embed_model_name).to(self.device)

                if self.device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device, torch_dtype=torch.float16)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device)

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.qa_model = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
                self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
                self.reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device=self.device)

                self._models_loaded = True # Mark models as loaded
                self._last_accessed_time = time.time() # Update access time
                print("Models loaded successfully.")
                self._schedule_clear_timer() # Schedule the clear timer after loading
            except Exception as e:
                print(f"Error loading models: {e}")
                # Clear any models that might have been partially loaded
                self._clear_models_internal()
                self._models_loaded = False # Ensure flag is false on failure
                # Consider reraising or handling this error appropriately
                raise CustomInternalError(f"Failed to load models: {e}") from e
            
    def _clear_models_internal(self):
        """Internal method to clear models. Does not handle the timer."""
        if not self._models_loaded:
             print("Models not currently loaded, skipping clear.")
             return

        print("Clearing models...")
        try:
            del self.embed_tokenizer, self.embed_model, self.qa_model, self.tokenizer, self.model, self.reranker, self.collection
            self.embed_tokenizer = None
            self.embed_model = None
            self.model = None
            self.tokenizer = None
            self.qa_model = None
            self.reranker = None
            self.collection = None # Also clear the collection reference

            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            self._models_loaded = False # Mark models as unloaded
            print("Models cleared.")
        except Exception as e:
             print(f"Error during model clear: {e}")
             # Attempt to reset references and states even on error
             self.embed_tokenizer = None
             self.embed_model = None
             self.model = None
             self.tokenizer = None
             self.qa_model = None
             self.reranker = None
             self.collection = None
             self._models_loaded = False
             # Don't re-raise, clearing is best-effort cleanup

    # Expose a public clear method, mainly for shutdown event
    def clear_models(self):
         """Public method to trigger model clearing."""
         with self._lock:
              self._clear_models_internal()
              # Optionally cancel the timer if explicitly cleared
              if self._timer and self._timer.is_alive():
                  self._timer.cancel()
                  self._timer = None

    def _get_embeddings(self, texts):
        if self.embed_tokenizer is None or self.embed_model is None:
             raise CustomInternalError("Embedding models are not loaded.")
        inputs = self.embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.embed_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu()

    def _get_keywords(self, question):
        if self.qa_model is None:
             raise CustomInternalError("QA Model is not loaded.")
        
        prompt = f"""You are an AI assistant for retrieving sales offers from a database. You look only what in the context.
    Task: Extract mentioned topics from followoing customer question: {question}

    Example 1:
    customer want OAuth with api and gpx.
    Response: OAuth with api, gpx

    Example 2:
    we want a ceu import and  color pick. Use also soap.
    Response: ceu import, color pick, soap

    Provide only questions separated by newlines."""
        messages = [{"role": "user", "content": prompt}]
        result = self.qa_model(messages, temperature=0.1, max_new_tokens=20)
        answer = result[0]["generated_text"][1]["content"] if isinstance(result[0]["generated_text"], list) else result[0]["generated_text"].split('\n')[1]
        return [x.strip() for x in answer.split(",") if x.strip()]

    def _query_db_with_keywords(self, keywords, threshold=0.8, top_k=10):
        if self.collection is None:
             raise CustomInternalError("Database collection is not loaded.")
        docs, dists, seen = [], [], set()

        for keyword in keywords:
            q_emb = self._get_embeddings(keyword)
            results = self.collection.query(query_embeddings=q_emb.numpy().tolist(), n_results=top_k, include=["documents", "distances"])

            for doc, dist in zip(results['documents'][0], results['distances'][0]):
                if abs(dist) <= threshold and doc not in seen:
                    docs.append(doc)
                    dists.append(abs(dist))
                    seen.add(doc)
        return docs, dists

    def answer_question(self, question):
        try:
            if not self._models_loaded:
                self.load_models()
            keywords = self._get_keywords(question)
            relevant_documents, _ = self._query_db_with_keywords(keywords)
            if not relevant_documents:
                return "I'm sorry, no relevant information available.", []

            if self.reranker is None:
                 raise CustomInternalError("Reranker model is not loaded.")
            reranked_results = self.reranker.rank(question, relevant_documents, return_documents=True, top_k=4)
            reranked_texts = [doc["text"] for doc in reranked_results]
            context = '\n\n'.join(reranked_texts)

            query = f"""Based on the customer request and context below, provide ONLY the total working days estimate.                 
Customer Request: {question}
Context: {context}



    Example 1:
        Working days: 2.5
        Automatic HU assignment:
        With the FaceToFace Picking, PPG assign automatically the HU according to a new Template (i.e. : HUXXXX)

        Working days: 2.5
        FaceToFace Picking:
        Label printing when PPG displays the END-Message 'Done' and when the operator push the button BoxFull.

        Total Working Days : 2.5 days

    Example 2:
        Working days: 3
        Add 10 Info-fields to :
        - REST-API 
        - Masterorderline table 
        - History table

        Working days: 2
        2 New fields
        Ø	Masterorder.Info6
        Ø	Masterorderline.Info6
        Only available in the REST import and export interface

        Total Working Days : 3 days

- If the context lacks relevant information, return exactly: Total Working Days : 0 days
- Do not include any explanation, reasoning, or additional text
- Only output the exact format 'Total Working Days : X days'"""
            messages = [{"role": "user", "content": query}]

            if self.qa_model is None:
                 raise CustomInternalError("QA Model is not loaded.")
            result = self.qa_model(messages, temperature=0.1, max_new_tokens=100)
            answer = result[0]["generated_text"][1]["content"] if isinstance(result[0]["generated_text"], list) else result[0]["generated_text"].split('\n')[1]

            return answer, reranked_texts
        except Exception as e:
            print(f"Error during answer_question: {e}")
            raise e



# --- Dependency Provider Function ---
# This function provides the single instance and ensures it's ready
def get_aufwandrechner_module():
    """Provides the single instance of AufwandrechnerModuleSingleton."""
    try:
        instance = AufwandrechnerModule(collection_name="unique")        
        instance._schedule_clear_timer()
        

        # No need to call load_models here - answer_question will do it on first use.
        # Ensure instance structure was initialized
        if not instance._initialized:
             # This should ideally not happen if __init__ is called during first __new__ call
             # but as a safeguard:
             raise RuntimeError("AufwandrechnerModuleSingleton structure failed to initialize.")
        return instance
    except RuntimeError as e:
        print(f"Error getting AufwandrechnerModuleSingleton instance: {e}")
        raise HTTPException(status_code=500, detail=f"Service initialization error: {e}") from e
    except Exception as e:
        print(f"An unexpected error occurred while getting the singleton instance: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e



def parse_customer_data_short(data_string):

    data = {}
    # Use regex to find key-value pairs formatted like "### Key ###: Value"
    matches = re.findall(r'###\s*(.*?)\s*###:\s*(.*)', data_string)

    for key, value in matches:
        key = key.strip().replace(' ', '_').lower()
        value = value.strip().replace(')', '') # Remove trailing ')' from customer name

        # Type conversion
        if key == 'customer_number' or key == 'offer':
            try:
                data[key] = int(value)
            except ValueError:
                data[key] = value
        elif key == 'created_on':
            try:
                data[key] = datetime.strptime(value, '%d.%m.%Y').date()
            except ValueError:
                data[key] = value
        elif key == 'working_days':
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
        else:
            data[key] = value

    # Handle the trailing context separately if not captured by the regex
    context_match = re.search(r'### context ###:(.*)', data_string, re.DOTALL)
    if context_match:
      data['context'] = context_match.group(1).strip()
    c = CustomerData(**data)
    return c





        
@app.get("/", response_class=HTMLResponse)
def get_index():
    # Consider serving index.html statically
    # from starlette.staticfiles import StaticFiles
    # app.mount("/", StaticFiles(directory="."), name="static")
     with open("index.html", "r", encoding="utf-8") as f:
         html_content = f.read()
     return html_content


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(request: QuestionRequest,module: AufwandrechnerModule = Depends(get_aufwandrechner_module)):
    # Create a new instance for each request.
    # This instance will load models, perform the query, and then unload.
    #module = AufwandrechnerModule(collection_name="unique")
    try:
        answer, reranked_texts = module.answer_question(f"customer want: {request.question}")
        reranked_texts = [parse_customer_data_short(each) for each in reranked_texts]        
        return {"answer": answer, "reranked_texts": reranked_texts}
    except CustomInternalError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    # The 'finally' block in answer_question ensures clear_models is called.

if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")
