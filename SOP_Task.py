import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer, util
import math

# --- Load Models ---

# QA Models
flan_tok = AutoTokenizer.from_pretrained("google/flan-t5-xl")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

ul2_tok = AutoTokenizer.from_pretrained("google/ul2")
ul2_model = AutoModelForSeq2SeqLM.from_pretrained("google/ul2")

# Language Model for Perplexity
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()

# Sentence similarity model
sbert = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")

# NLI model
nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")


# --- Story and Query ---
story_url = "https://www.owleyes.org/text/gift-magi/read/the-gift-of-the-magi#root-218982-1"
with open("gift_of_magi.txt", "r", encoding="utf-8") as f:
    S = f.read()

Q = "What sacrifice did Della make, and why was it significant?"

prompt = f"Read the following story and answer the question.\n\nStory:\n{S}\n\nQuestion:\n{Q}"


# --- Generate Answers ---

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    output = model.generate(**inputs, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

R1 = generate_answer(flan_model, flan_tok, prompt)
R2 = generate_answer(ul2_model, ul2_tok, prompt)

print("FLAN-T5 Answer:", R1)
print("UL2 Answer:", R2)


# --- Evaluation Metrics ---

# 1. Grammaticality: Perplexity
def perplexity(text):
    encodings = gpt2_tok(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = gpt2(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

ppl_R1 = perplexity(R1)
ppl_R2 = perplexity(R2)


# 2. Relevance: Cosine similarity
query_embedding = sbert.encode([Q])[0]
emb_R1 = sbert.encode([R1])[0]
emb_R2 = sbert.encode([R2])[0]

sim_R1 = util.cos_sim(emb_R1, query_embedding).item()
sim_R2 = util.cos_sim(emb_R2, query_embedding).item()


# 3. Faithfulness: Entailment
def entailment_score(premise, hypothesis):
    inputs = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs[0][2].item()  # entailment prob

entail_R1 = entailment_score(S, R1)
entail_R2 = entailment_score(S, R2)


# --- Results Summary ---
print("\n--- Evaluation Results ---")
print(f"Grammaticality (Perplexity):\n  FLAN: {ppl_R1:.2f} | UL2: {ppl_R2:.2f}")
print(f"Relevance (Cosine Similarity):\n  FLAN: {sim_R1:.4f} | UL2: {sim_R2:.4f}")
print(f"Faithfulness (Entailment Prob):\n  FLAN: {entail_R1:.4f} | UL2: {entail_R2:.4f}")
