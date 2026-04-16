<div align="center">

```
 ██████╗ ███████╗███╗   ██╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
╚██████╔╝███████╗██║ ╚████║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

### 🔐 *Where Genomics Meets Cryptography* 🧬

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)](LICENSE)

<br/>

> *"Securing the blueprint of life — one encrypted base at a time."*

</div>

---

## 🌟 What is GenForge?

**GenForge** is a cutting-edge web application that fuses **military-grade encryption** with **AI-powered DNA optimization**. Whether you're a researcher safeguarding sensitive genomic data or exploring the intersection of bioinformatics and cybersecurity, GenForge delivers a seamless, secure, and intelligent pipeline for DNA sequence processing.

```
        🧑‍💻 CLIENT                    🖥️ SERVER
      ┌──────────┐                 ┌──────────────┐
      │  Input   │  ─── AES ───►  │   Decrypt    │
      │  DNA Seq │   Encrypted    │      +        │
      │ 20 bases │  ◄── AES ───   │  AI Optimize │
      └──────────┘   Encrypted    └──────────────┘
           │                             │
           └──── 🔒 Secure Channel ──────┘
```

---

## ✨ Feature Highlights

| Feature | Description |
|--------|-------------|
| 🔒 **AES-CBC Encryption** | Military-grade 128/192/256-bit encryption for all transmissions |
| 🧬 **DNA-BERT Optimization** | State-of-the-art transformer model trained on genomic data |
| 📊 **Cosine Similarity Scoring** | Quantitative before/after optimization metrics |
| 🔁 **Bidirectional Secure Workflow** | Client ↔ Server round-trip, fully encrypted |
| ⚡ **Real-time Processing** | Instant results via Flask's lightweight server |
| 🎯 **Position-wise Diff** | See exactly which bases were optimized and why |

---

## 🏗️ Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────────┐
│                        GENFORGE PIPELINE                        │
│                                                                  │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │  INPUT  │───►│  ENCRYPT │───►│ TRANSMIT │───►│  DECRYPT  │  │
│  │ ATCGATCG│    │ AES-CBC  │    │  Flask   │    │  AES-CBC  │  │
│  └─────────┘    └──────────┘    └──────────┘    └───────────┘  │
│                                                        │         │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐          ▼         │
│  │ DELIVER │◄───│  ENCRYPT │◄───│ OPTIMIZE │◄──── DNA-BERT      │
│  │ RESULT  │    │  AES-CBC │    │    AI    │      Embeddings     │
│  └─────────┘    └──────────┘    └──────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 🔐 Encryption Layer

```python
# Payload structure: [ IV (16 bytes) | Ciphertext ]
# Random IV ensures each transmission is unique — even for identical sequences!

Plaintext DNA  ──►  AES-CBC(key, IV)  ──►  Base64  ──►  Transmission
```

- **Algorithm**: AES (Advanced Encryption Standard)
- **Mode**: CBC (Cipher Block Chaining)
- **IV**: Cryptographically random 16-byte vector per transmission
- **Encoding**: Base64 for safe transport

### 🤖 AI Optimization Layer

```
DNA Sequence  ──►  k-mer Tokenization  ──►  DNA-BERT  ──►  Embeddings
                                                               │
Reference  ──►  k-mer Tokenization  ──►  DNA-BERT  ──►  Embeddings
                                                               │
                                               Cosine Similarity Score
                                                               │
                                          Weak Position Detection
                                                               │
                                       Base Substitution (A/T/C/G)
                                                               │
                                         Optimized Sequence 🎯
```

---

## 📂 Project Structure

```
GenForge/
│
├── 🐍 app.py              # Main Flask application (client + server routes)
├── 🔐 aes_utils.py        # AES encryption & decryption logic
├── 🧬 dna_model.py        # DNA-BERT model interface & optimization
├── ⚙️  config.py           # Secret key & configuration management
├── 📦 requirements.txt    # Python dependencies
│
└── 🎨 templates/
    ├── client.html        # User-facing input interface
    └── server.html        # Processing & results display
```

---

## ⚙️ Quick Start

### 1️⃣ Clone & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/genforge.git
cd genforge

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure Your Secret Key

Edit `config.py`:

```python
# ⚠️ Must be exactly 16, 24, or 32 bytes
AES_KEY = b'ThisIsASecretKey'   # 16 bytes — AES-128
# AES_KEY = b'ThisIsALongerSecretKey!!'  # 24 bytes — AES-192
# AES_KEY = b'ThisIsAMuchLongerSecretKey!!'  # 32 bytes — AES-256
```

### 3️⃣ Launch GenForge

```bash
python app.py
```

Open your browser: **http://127.0.0.1:5000/**

---

## 🧪 Usage Guide

### Input Rules

```
✅ Valid   : ATCGATCGATCGATCGATCG   (20 chars, A/T/C/G only)
❌ Invalid : ATCGATCGATCG           (too short)
❌ Invalid : ATCGATCGATCGATCGATCN   (invalid character 'N')
❌ Invalid : atcgatcgatcgatcgatcg   (lowercase not accepted)
```

### Example Workflow

```
INPUT  ──►  ATCGATCGATCGATCGATCG
               │
               ▼ AES Encrypt
CIPHER ──►  [IV][EncryptedData...] → (Base64)
               │
               ▼ DNA-BERT Optimize
OUTPUT ──►  ATCGTTCGATCGATCGAACG
               │
Changes:   Position 5: A→T | Position 18: T→C
Similarity:  Before: 0.847 | After: 0.923  📈
               │
               ▼ AES Encrypt
CIPHER ──►  [IV][EncryptedResult...] → (Sent back)
```

### Output Breakdown

| Field | Description |
|-------|-------------|
| 🔓 **Decrypted Sequence** | Your original DNA, confirmed received |
| ✏️ **Optimized Sequence** | AI-improved variant |
| 🔄 **Change Log** | Position-by-position mutation list |
| 📈 **Similarity Score (Before)** | Cosine similarity with reference |
| 📈 **Similarity Score (After)** | Improved score post-optimization |
| 🔐 **Encrypted Result** | Re-encrypted optimized sequence |

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────┐
│                  TECH STACK                     │
│                                                  │
│  🌐 Flask     ─── Web framework & routing        │
│  🔐 PyCryptodome ─ AES encryption engine         │
│  🤗 Transformers ─ DNA-BERT model interface      │
│  🔥 PyTorch   ─── Deep learning backend          │
│  🐍 Python    ─── Core language                  │
└─────────────────────────────────────────────────┘
```

---

## ⚠️ Known Limitations

> These are known trade-offs in the current academic implementation:

- 🔓 **No Message Authentication** — AES-CBC does not verify data integrity (no MAC/HMAC)
- 📏 **Fixed Input Length** — Sequences must be exactly 20 bases
- 📚 **Static References** — Optimization targets are predefined reference sequences
- 👤 **No Authentication** — No user login or session management

---

## 🔮 Future Roadmap

```
v1.0  ──  Current Academic Release
  │
  ├── v1.1  🔐 AES-GCM (Authenticated Encryption)
  │           Adds integrity verification to prevent tampering
  │
  ├── v1.2  🔑 ECDHE Key Exchange
  │           Secure ephemeral key negotiation — no pre-shared keys
  │
  ├── v1.3  🌍 Cloud API Deployment
  │           Dockerized, scalable, REST API with rate limiting
  │
  └── v2.0  🧬 Extended Sequence Support + Fine-tuned BERT
              Variable-length sequences & custom training datasets
```

---

## 🤝 Contributing

Contributions are welcome! If you're a student, researcher, or enthusiast, feel free to:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch: `git checkout -b feature/your-idea`
3. 💾 Commit changes: `git commit -m 'Add your feature'`
4. 📤 Push to branch: `git push origin feature/your-idea`
5. 🔃 Open a Pull Request

---

## 📜 License

```
╔══════════════════════════════════════════════════╗
║                ACADEMIC LICENSE                  ║
║                                                  ║
║  This project is developed for educational and   ║
║  research purposes only. Not intended for        ║
║  production or commercial use.                   ║
╚══════════════════════════════════════════════════╝
```

---

<div align="center">

**Built with 🧬 + 🔐 + 🤖 for the future of secure bioinformatics.**

*If GenForge helped your research, consider leaving a ⭐ on GitHub!*

</div>
