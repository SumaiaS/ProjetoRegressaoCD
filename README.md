# Relatório Técnico: Implementação e Análise do Algoritmo k-Nearest Neighbors (kNN) Aplicado ao Instagram

**Nome do Residente:** [Seu Nome]  
**Data de Entrega:** [Data de Entrega]

## Resumo

Este projeto tem como objetivo desenvolver um modelo preditivo para a taxa de engajamento de influenciadores do Instagram, utilizando técnicas de regressão linear. O script main2.py explora dados como seguidores, posts e pontuação de influência para prever a taxa de engajamento a partir de um conjunto de dados preexistente. São implementados três modelos principais: Regressão Linear Simples, Regressão Lasso (L1) e Regressão Ridge (L2). A análise avalia os modelos por meio de métricas como Erro Médio Absoluto (MAE), Erro Quadrático Médio (MSE) e Coeficiente de Determinação (R²).

## Introdução

A análise de influenciadores é cada vez mais relevante para o marketing digital. Este projeto visa compreender as variáveis que mais influenciam a taxa de engajamento, um indicador fundamental para medir o impacto de um influenciador. As técnicas de regressão linear são usadas para entender as relações entre variáveis e prever a taxa de engajamento com base em dados históricos.

### Conjunto de Dados

O conjunto de dados utilizado neste projeto contém as seguintes variáveis:
- **rank:** Rank do influenciador.
- **channel_info:** Nome do influenciador.
- **influence_score:** Pontuação de influência do influenciador.
- **posts:** Número de postagens.
- **followers:** Número de seguidores.
- **avg_likes:** Média de curtidas por postagem.
- **60_day_eng_rate:** Taxa de engajamento nos últimos 60 dias.
- **new_post_avg_like:** Média de curtidas das novas postagens.
- **total_likes:** Total de curtidas acumuladas.
- **country:** País de origem do influenciador.

Além disso, a coluna **country** foi transformada em um valor numérico representando continentes.

## Metodologia

### Análise Exploratória

Foram analisadas distribuições de dados e possíveis correlações entre variáveis. Gráficos de dispersão foram gerados para examinar a relação entre seguidores e taxa de engajamento, e mapas de calor de correlação ajudaram a identificar variáveis com maior influência sobre a variável alvo.

### Implementação do Algoritmo

O script implementa três modelos principais de regressão:

Regressão Linear Simples
Regressão Lasso (L1) com regularização para redução de variáveis irrelevantes.
Regressão Ridge (L2) para atenuação de colinearidades entre variáveis.
A normalização das variáveis foi realizada com o StandardScaler, e a transformação da coluna country facilitou a análise. Dividimos os dados em conjuntos de treino e teste na proporção de 80/20.

### Validação e Ajuste de Hiperparâmetros

Para selecionar os melhores hiperparâmetros, utilizamos validação cruzada. As métricas de erro e ajuste dos modelos foram calculadas em cada etapa para garantir consistência e robustez nos resultados.

## Resultados

### Métricas de Avaliação

Após a implementação do modelo kNN, as seguintes métricas foram obtidas:

- **MAE:** [Valor MAE]
- **MSE:** [Valor MSE]
- **RMSE:** [Valor RMSE]
- **R²:** [Valor R²]

### Visualizações

Distribuição da Taxa de Engajamento: Histograma que mostra a distribuição da variável 60_day_eng_rate.
Relação entre Seguidores e Taxa de Engajamento: Gráfico de dispersão entre followers e 60_day_eng_rate.
Mapa de Correlação: Heatmap das correlações entre variáveis.

```python
# Exemplo de gráfico
# Exemplo de visualização em Python
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de dispersão entre 'followers' e 'avg_likes'
plt.figure(figsize=(10, 5))
sns.scatterplot(x='followers', y='avg_likes', hue='60_day_eng_rate', data=df)
plt.title('Relação entre Seguidores e Média de Curtidas')
plt.xlabel('Followers')
plt.ylabel('Avg Likes')
plt.show()

# Mapa de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.title('Mapa de Correlação das Variáveis')
plt.show()