"""
Offline script: Generate text embeddings for UCF-Crime 14 categories.
Run once, save to .npy file. Used as frozen anchors during training.

Usage:
    python src/utils/generate_text_embeddings.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# 13 anomaly categories + 1 normal
CATEGORY_DESCRIPTIONS = {
    "Abuse": "A video showing physical abuse where a person violently attacks or beats another person",
    "Arrest": "A video showing an arrest where police officers restrain and detain a suspect",
    "Arson": "A video showing arson where someone deliberately sets fire to a building or property",
    "Assault": "A video showing an assault where a person physically attacks another person with violent intent",
    "Burglary": "A video showing a burglary where someone breaks into a building to steal property",
    "Explosion": "A video showing an explosion with sudden destructive blast and fire",
    "Fighting": "A video showing a fight between two or more people engaging in physical combat",
    "RoadAccidents": "A video showing a road accident with vehicles colliding or crashing",
    "Robbery": "A video showing a robbery where someone forcibly takes property from a victim",
    "Shooting": "A video showing a shooting where a person fires a gun at others",
    "Shoplifting": "A video showing shoplifting where someone steals merchandise from a store",
    "Stealing": "A video showing stealing where someone takes another person's property without permission",
    "Vandalism": "A video showing vandalism where someone deliberately destroys or damages property",
    "Normal": "A video showing normal everyday activity with no anomaly or criminal behavior",
}

CATEGORY_ORDER = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "Normal"
]


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    descriptions = [CATEGORY_DESCRIPTIONS[cat] for cat in CATEGORY_ORDER]
    embeddings = model.encode(descriptions, normalize_embeddings=True)

    print(f"Embeddings shape: {embeddings.shape}")  # [14, 384]
    print(f"Categories: {CATEGORY_ORDER}")

    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norms (should be ~1.0): min={norms.min():.4f}, max={norms.max():.4f}")

    # Save
    out_path = "text_embeddings.npz"
    np.savez(out_path,
             embeddings=embeddings,
             categories=CATEGORY_ORDER)
    print(f"Saved to {out_path}")

    # Print cosine similarity matrix
    sim = embeddings @ embeddings.T
    print("\nCosine similarity matrix:")
    for i, cat in enumerate(CATEGORY_ORDER):
        print(f"  {cat:15s}: {sim[i].round(2).tolist()}")


if __name__ == "__main__":
    main()
