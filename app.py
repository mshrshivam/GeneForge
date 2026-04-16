from flask import Flask, render_template, request, redirect, url_for
from aes_utils import encrypt, decrypt
from dna_model import optimize_sequence, get_sequence_score, generate_reference_embeddings
from config import AES_KEY
import urllib.parse

app = Flask(__name__)

REFERENCE_SEQUENCES = [
    "CTACTTCAAATGGGGCTACA",
    "AGTCGTACTGCATGCTCGTA",
    "ATCGCTGACAATGCTGGACA"
]

REFERENCE_EMBEDDINGS = generate_reference_embeddings(REFERENCE_SEQUENCES)

BASES = ['A', 'T', 'C', 'G']


# ---------------- CLIENT ----------------
@app.route("/", methods=["GET", "POST"])
def client():
    encrypted_result = request.args.get("result")
    
    if encrypted_result:
        encrypted_result = urllib.parse.unquote(encrypted_result)
    decrypted_result = None

    if request.method == "POST":
        sequence = request.form["sequence"].upper()

        if len(sequence) != 20 or not all(c in BASES for c in sequence):
            return render_template("client.html", error="Invalid DNA sequence")

        encrypted = encrypt(sequence, AES_KEY)
        return redirect(url_for("server", data=encrypted))

    # If returning from server
    if encrypted_result:
        try:
            decrypted_result = decrypt(encrypted_result, AES_KEY)
        except:
            decrypted_result = "Decryption failed"

    return render_template("client.html",
                           encrypted_result=encrypted_result,
                           decrypted_result=decrypted_result)


# ---------------- SERVER ----------------
@app.route("/server")
def server():
    encrypted_data = request.args.get("data")

    if not encrypted_data:
        return redirect(url_for("client"))

    # Decrypt
    decrypted = decrypt(encrypted_data, AES_KEY)

    # Process
    original_score = get_sequence_score(decrypted, REFERENCE_EMBEDDINGS)
    edited_seq, changes, new_score = optimize_sequence(decrypted, REFERENCE_EMBEDDINGS, k=3)

    # Re-encrypt output
    encrypted_output = encrypt(edited_seq, AES_KEY)

    return render_template("server.html",
                           encrypted_input=encrypted_data,
                           decrypted=decrypted,
                           edited=edited_seq,
                           changes=changes,
                           original_score=round(original_score, 4),
                           new_score=round(new_score, 4),
                           encrypted_output=encrypted_output)


if __name__ == "__main__":
    app.run(debug=True)