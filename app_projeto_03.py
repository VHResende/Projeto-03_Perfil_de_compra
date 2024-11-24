import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Configuração inicial da página
st.set_page_config(page_title="Comportamento de Navegação", layout="wide")

# Função para carregar os dados
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\User\Documents\EBAC\Cientista de Dados\Profissão Cientista de Dados\Módulo 31_Streamlit V\Exercício 02\online_shoppers_intention.csv")

# Função para realizar o pré-processamento
def preprocess_data(df):
    variaveis_navegacao = ['Administrative', 'Administrative_Duration', 
                           'Informational', 'Informational_Duration', 
                           'ProductRelated', 'ProductRelated_Duration']
    df[variaveis_navegacao] = df[variaveis_navegacao].fillna(df[variaveis_navegacao].mean())
    return df, variaveis_navegacao

# Carregando os dados
df = load_data()
df, variaveis_navegacao = preprocess_data(df)

    # Título do App
st.title(" 📊 Projeto 03: Análise do Comportamento de Navegação e Perfil de Compra")
    
st.markdown("---")

st.markdown("""
Neste projeto vamos usar a base [online shoppers purchase intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Link](https://link.springer.com/article/10.1007/s00521-018-3523-0).

A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?"

Nosso objetivo é tentar agrupar os clientes conforme seu comportamento de navegação entre páginas administrativas, informativas e de produtos.
""")
    
st.markdown("---")
    
# Exibição inicial dos dados
if st.checkbox("Mostrar os primeiros dados"):
    st.write(df.head())

# Análise descritiva
st.header("Análise Descritiva")
st.write("Estatísticas descritivas das variáveis relacionadas ao comportamento de navegação:")
st.write(df[variaveis_navegacao].describe())

# Visualização da distribuição das variáveis
st.subheader("Distribuição das variáveis de navegação")
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for i, col in enumerate(variaveis_navegacao):
    sns.histplot(df[col], bins=20, kde=True, ax=ax[i // 3, i % 3])
    ax[i // 3, i % 3].set_title(f'Distribuição de {col}')
plt.tight_layout()
st.pyplot(fig)

# Agrupamento - Escolha do número de clusters
st.header("Agrupamento de Usuários")
st.write("Os dados são escalados antes do agrupamento para padronizar as variáveis.")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[variaveis_navegacao])

# Método do Cotovelo
st.subheader("Método do Cotovelo")
if st.button("Calcular Cotovelo"):
    sse = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(k_values, sse, marker='o')
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Soma dos Erros Quadrados (SSE)")
    ax.set_title("Método do Cotovelo")
    st.pyplot(fig)

# Coeficiente de Silhueta
st.subheader("Coeficiente de Silhueta")
if st.button("Calcular Silhueta"):
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        silhouette_scores.append(silhouette_score(df_scaled, labels, sample_size=2000, random_state=42))
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), silhouette_scores, marker='o')
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Coeficiente de Silhueta")
    ax.set_title("Coeficiente de Silhueta")
    st.pyplot(fig)

# Escolha do número de clusters
num_clusters = st.slider("Escolha o número de clusters para análise", min_value=2, max_value=10, value=3)

# Implementação do K-Means com clusters escolhidos
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Análise descritiva dos clusters
st.subheader(f"Análise Descritiva para {num_clusters} Clusters")
cluster_means = df.groupby('cluster')[variaveis_navegacao].mean()
st.write(cluster_means)

# Visualização dos clusters
st.subheader("Visualização das Médias por Cluster")
fig, ax = plt.subplots()
cluster_means.plot(kind='bar', ax=ax)
ax.set_title(f'Médias das Variáveis de Navegação por Cluster ({num_clusters} Grupos)')
ax.set_ylabel("Média")
ax.set_xlabel("Cluster")
plt.xticks(rotation=0)
st.pyplot(fig)

# Avaliação de propensão à compra
st.header("Avaliação de Propensão à Compra")
st.write("Análise de variáveis fora do escopo (BounceRates e Revenue) para os clusters:")
extra_analysis = df.groupby('cluster')[['BounceRates', 'Revenue']].mean()
st.write(extra_analysis)

# Visualização de BounceRates e Revenue
st.subheader("Visualização de BounceRates e Revenue")
fig, ax = plt.subplots()
extra_analysis.plot(kind='bar', ax=ax)
ax.set_title("BounceRates e Revenue por Cluster")
ax.set_ylabel("Média")
ax.set_xlabel("Cluster")
plt.xticks(rotation=0)
st.pyplot(fig)

# Conclusão
st.header("Conclusão")
st.write("Com base nos agrupamentos, avalie os clusters mais propensos à compra (maior média em Revenue).")

# by Victor Resende
st.markdown("---")
st.write("by 📊 **Victor Resende**")

