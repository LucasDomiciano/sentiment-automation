# sentiment_automation.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup

# Carregar o dataset IMDB
print("Carregando o dataset IMDB...")
df_train = pd.read_csv("IMDB Dataset.csv")
texts = df_train["review"].tolist()
labels = df_train["sentiment"].map({"positive": 1, "negative": 0}).tolist()

# Dividir em treino e teste
print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Transformar texto em números
print("Vetorizando os textos...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinar o modelo
print("Treinando o modelo...")
model = LogisticRegression(max_iter=1000)  # Aumentei max_iter para garantir convergência
model.fit(X_train_vec, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Função para prever sentimentos
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Positivo" if prediction == 1 else "Negativo"

# Automação: coletar dados fictícios (substitua por web scraping depois)
comments = ["Great product!", "Terrible service.", "It's okay."]
results = []
for comment in comments:
    sentiment = predict_sentiment(comment)
    results.append({"Comentário": comment, "Sentimento": sentiment})

# Salvar em CSV
df = pd.DataFrame(results)
df.to_csv("sentiment_report.csv", index=False)
print("Relatório gerado em 'sentiment_report.csv'!")
print(df)

# Modo interativo
while True:
    user_input = input("Digite um texto para analisar (ou 'sair'): ")
    if user_input.lower() == 'sair':
        break
    print(f"Sentimento: {predict_sentiment(user_input)}")