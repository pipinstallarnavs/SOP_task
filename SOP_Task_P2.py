import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel
)
from sentence_transformers import SentenceTransformer, util
import math

# --- Load Models ---

# Mistral 7B Instruct
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
mistral = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")

# DeBERTa v3 QA
deberta_tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
deberta = AutoModelForQuestionAnswering.from_pretrained("microsoft/deberta-v3-large")

# GPT2 for perplexity
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()

# Sentence similarity
sbert = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")

# NLI for faithfulness
nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")


# --- Input Text and Query ---
with open("gift_of_magi.txt", "r", encoding="utf-8") as f:
    S = f.read()

Q = "What sacrifice did Della make, and why was it significant?"

# --- Generate Answer: Mistral (text-gen) ---
def generate_mistral(prompt):
    input_ids = mistral_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids
    input_ids = input_ids.to(mistral.device)
    output = mistral.generate(input_ids, max_length=128)
    return mistral_tok.decode(output[0], skip_special_tokens=True)

mistral_prompt = f"[INST] Read the story below and answer the question.\n\nStory:\n{S}\n\nQuestion: {Q} [/INST]"
R1 = generate_mistral(mistral_prompt)


# --- Generate Answer: DeBERTa (span-extraction) ---
def generate_deberta(context, question):
    inputs = deberta_tok(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = deberta(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    return deberta_tok.decode(inputs["input_ids"][0][start:end+1])

R2 = generate_deberta(S, Q)

print("Mistral Answer:\n", R1)
print("DeBERTa Answer:\n", R2)


# --- Evaluation Functions ---

def perplexity(text):
    enc = gpt2_tok(text, return_tensors="pt")
    input_ids = enc.input_ids
    with torch.no_grad():
        outputs = gpt2(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())

def cosine_similarity(q, a):
    emb_q = sbert.encode([q])[0]
    emb_a = sbert.encode([a])[0]
    return util.cos_sim(emb_q, emb_a).item()

def entailment_score(premise, hypothesis):
    inputs = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs[0][2].item()  # entailment


# --- Metric Evaluation ---
ppl_R1 = perplexity(R1)
ppl_R2 = perplexity(R2)

sim_R1 = cosine_similarity(Q, R1)
sim_R2 = cosine_similarity(Q, R2)

entail_R1 = entailment_score(S, R1)
entail_R2 = entailment_score(S, R2)


# --- Results ---
print("\n--- Evaluation ---")
print(f"Grammaticality (PPL):       Mistral: {ppl_R1:.2f} | DeBERTa: {ppl_R2:.2f}")
print(f"Relevance (Cosine Sim):     Mistral: {sim_R1:.4f} | DeBERTa: {sim_R2:.4f}")
print(f"Faithfulness (Entailment):  Mistral: {entail_R1:.4f} | DeBERTa: {entail_R2:.4f}")
