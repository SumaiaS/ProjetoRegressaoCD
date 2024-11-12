import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Carregar os dados
df = pd.read_csv('influencers.csv')

# Visualizar os primeiros dados para entender a estrutura
print(df.head())
print(df.info())

# 2. Tratamento de dados
def convert_to_numeric(value):
    if isinstance(value, str):
        if 'k' in value:
            return float(value.replace('k', '')) * 1_000
        elif 'm' in value:
            return float(value.replace('m', '')) * 1_000_000
        elif 'b' in value:
            return float(value.replace('b', '')) * 1_000_000_000
    return float(value)

# Converter colunas relevantes
df['followers'] = df['followers'].apply(convert_to_numeric)
df['posts'] = df['posts'].apply(convert_to_numeric)
df['60_day_eng_rate'] = df['60_day_eng_rate'].str.replace('%', '').astype(float)

# Remover entradas com NaN na variável alvo
df = df.dropna(subset=['60_day_eng_rate'])

# 3. Normalização dos dados
scaler = StandardScaler()
X = df[['followers', 'posts', 'influence_score']]
y = df['60_day_eng_rate']
X_scaled = scaler.fit_transform(X)

# 4. Seleção de Recursos
selector = SelectKBest(f_classif, k='all')
X_selected = selector.fit_transform(X_scaled, y)

# 5. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 6. Implementação dos modelos com regularização
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression (L1)': Lasso(alpha=0.1),  # Ajustar o parâmetro alpha conforme necessário
    'Ridge Regression (L2)': Ridge(alpha=0.1)    # Ajustar o parâmetro alpha conforme necessário
}

# 7. Treinamento e avaliação
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'\nModelo: {name}')
    print(f'Erro Quadrático Médio (MSE): {mse}')
    print(f'Coeficiente de Determinação (R²): {r2}')
    print(f'Erro Absoluto Médio (MAE): {mae}')

    # Validação cruzada
    cv_scores = cross_val_score(model, X_selected, y, cv=5)
    print(f'Validação Cruzada (Média R²): {cv_scores.mean()}')

# 8. Análise e Visualização dos Resultados
# Visualização da distribuição da variável alvo
plt.figure(figsize=(10, 6))
sns.histplot(df['60_day_eng_rate'], bins=30, kde=True)
plt.title('Distribuição da Taxa de Engajamento')
plt.xlabel('Taxa de Engajamento')
plt.ylabel('Frequência')
plt.show()

# Relação entre seguidores e taxa de engajamento
plt.figure(figsize=(10, 6))
sns.scatterplot(x='followers', y='60_day_eng_rate', data=df)
plt.title('Relação entre Seguidores e Taxa de Engajamento')
plt.xlabel('Seguidores')
plt.ylabel('Taxa de Engajamento')
plt.xscale('log')  # Usar escala logarítmica se houver grandes diferenças
plt.show()

# 9. Interpretação dos Coeficientes
for name, model in models.items():
    if hasattr(model, 'coef_'):
        print(f'\nCoeficientes do modelo {name}: {model.coef_}')

# Visualizações Gráficas
plt.figure(figsize=(10, 6))
for name, model in models.items():
    plt.scatter(y_test, model.predict(X_test), label=name, alpha=0.5)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.title("Real vs Previsão da Taxa de Engajamento")
plt.legend()
plt.show()

# Mapa de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlação')
plt.show()
