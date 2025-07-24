from transformers import pipeline
# pipeline: fonction qui permet de charger un modèle pré-entraîné pour l'analyse de sentiment

# Charger le pipeline d'analyse de sentiment
sentiment_analyzer = pipeline("sentiment-analysis")
# sentiment-analysis: tache d'analyse de sentiment, en anglais

def analyze_sentiment(text: str) -> dict:
    """
    Analyse le sentiment d'un texte donné.

    Args:
        text (str): Le texte à analyser.

    Returns:
        dict: Un dictionnaire contenant le label (positif/négatif) et le score.
    """
    result = sentiment_analyzer(text)
    return {"label": result[0]["label"], "score": result[0]["score"]}