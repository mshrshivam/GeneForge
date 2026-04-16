from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading DNA-BERT model...")

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6").to(device)

model.eval()
for param in model.parameters():
    param.requires_grad = False

BASES = ['A', 'T', 'C', 'G']

# ---------- EMBEDDINGS ----------
def get_token_embeddings(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.squeeze(0)

# ---------- REFERENCE ----------
def generate_reference_embeddings(sequences):
    all_embeds = []
    max_len = 0

    for seq in sequences:
        emb = get_token_embeddings(seq)
        all_embeds.append(emb)
        max_len = max(max_len, emb.size(0))

    padded = []
    for emb in all_embeds:
        pad_size = max_len - emb.size(0)
        if pad_size > 0:
            emb = torch.cat([emb, torch.zeros(pad_size, emb.size(1), device=device)], dim=0)
        padded.append(emb)

    return torch.stack(padded).mean(dim=0)

# ---------- FIND TOP K PROBLEM AREAS ----------
def find_top_k_problematic_positions(token_embeddings, reference_embeddings, k=3):
    cos = torch.nn.CosineSimilarity(dim=1)
    similarities = cos(token_embeddings, reference_embeddings[:token_embeddings.size(0)])
    return torch.topk(similarities, k, largest=False).indices.tolist()

# ---------- TRY BEST BASE ----------
def choose_best_alternate_base(seq, idx, reference_embeddings):
    best_base = None
    best_score = -1
    original = seq[idx]

    for base in BASES:
        if base == original:
            continue

        new_seq = seq[:idx] + base + seq[idx+1:]
        emb = get_token_embeddings(new_seq)

        seq_embed = emb.mean(dim=0).unsqueeze(0)
        ref_embed = reference_embeddings.mean(dim=0).unsqueeze(0)

        score = torch.cosine_similarity(seq_embed, ref_embed).item()

        if score > best_score:
            best_score = score
            best_base = base

    return best_base, best_score

# ---------- SCORE ----------
def get_sequence_score(seq, reference_embeddings):
    emb = get_token_embeddings(seq)
    seq_embed = emb.mean(dim=0).unsqueeze(0)
    ref_embed = reference_embeddings.mean(dim=0).unsqueeze(0)

    return torch.cosine_similarity(seq_embed, ref_embed).item()

# ---------- MAIN EDIT FUNCTION ----------
def optimize_sequence(seq, reference_embeddings, k=3):
    token_embeds = get_token_embeddings(seq)
    indices = find_top_k_problematic_positions(token_embeds, reference_embeddings, k)

    edited_seq = seq
    changes = []

    for idx in indices:
        new_base, _ = choose_best_alternate_base(edited_seq, idx, reference_embeddings)
        old_base = edited_seq[idx]

        edited_seq = edited_seq[:idx] + new_base + edited_seq[idx+1:]

        changes.append({
            "position": idx + 1,
            "old": old_base,
            "new": new_base
        })

    final_score = get_sequence_score(edited_seq, reference_embeddings)

    return edited_seq, changes, final_score