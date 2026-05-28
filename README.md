# Laboratório 10 - Pipeline Definitivo QLoRA + RAG

## Objetivo

Este laboratório tem como objetivo demonstrar técnicas modernas de otimização para modelos de linguagem de grande porte (LLMs), analisando o impacto da quantização, do KV Cache e de mecanismos avançados de atenção em cenários semelhantes a sistemas RAG (Retrieval-Augmented Generation).

---

## Tecnologias Utilizadas

* Python
* PyTorch
* Transformers
* BitsAndBytes
* Accelerate
* TinyLlama 1.1B Chat
* QLoRA (Quantização 4-bit)
* KV Cache
* SDPA (Scaled Dot Product Attention)

---

## Modelo Utilizado

Modelo:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

O modelo foi carregado utilizando quantização em 4 bits através do BitsAndBytes, reduzindo significativamente o consumo de memória da GPU.

---

## Arquitetura Implementada

### 1. Quantização 4-bit (QLoRA)

Foi utilizada a técnica de quantização NF4 para reduzir o tamanho do modelo em memória sem necessidade de carregar os pesos em precisão total.

Configuração utilizada:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
```

---

### 2. Contexto Massivo Simulando RAG

Foi criado um contexto médico extenso simulando documentos recuperados por um sistema RAG.

O contexto é repetido múltiplas vezes para representar grandes volumes de informação que normalmente seriam recuperados por uma base vetorial.

---

### 3. KV Cache

Foram realizados dois experimentos:

* Geração sem KV Cache
* Geração com KV Cache

O objetivo foi comparar desempenho e utilização de memória.

---

### 4. FlashAttention e Fallback

O código tenta utilizar:

```text
FlashAttention-2
```

Quando o ambiente não suporta FlashAttention-2 (caso da GPU T4 do Google Colab), é utilizado automaticamente o fallback:

```text
SDPA (Scaled Dot Product Attention)
```

fornecido pelo PyTorch.

Fluxo implementado:

```text
FlashAttention-2
       ↓
     SDPA
       ↓
Atenção padrão
```

---

## Resultados Obtidos

### Modelo Quantizado

```text
VRAM utilizada após carregamento:
788.50 MB
```

---

### Sem KV Cache

```text
Tempo de execução:
105.65 segundos

Pico de VRAM:
2036.10 MB
```

---

### Com KV Cache + SDPA

```text
Tempo de execução:
5.22 segundos

Pico de VRAM:
2501.79 MB
```

---

## Análise dos Resultados

Os resultados demonstram que:

* A quantização em 4 bits reduz drasticamente o consumo de memória do modelo.
* O KV Cache reduz significativamente o tempo de geração.
* O fallback para SDPA permite manter otimizações mesmo em ambientes que não suportam FlashAttention-2.
* O contexto simulado de RAG possibilita avaliar o comportamento do modelo em cenários de grande volume de informação.

---

## Estrutura do Projeto

```text
Pipeline-definitivo-qlora-rag/
│
├── src/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Como Executar

Instalar dependências:

```bash
pip install -r requirements.txt
```

Executar:

```bash
python src/main.py
```

---

## Conclusão

O laboratório demonstrou o impacto das técnicas de otimização aplicadas a LLMs, evidenciando os benefícios da quantização 4-bit, do KV Cache e dos mecanismos modernos de atenção para reduzir custos computacionais e melhorar desempenho em cenários semelhantes a sistemas RAG.

---

## Uso de IA

Partes deste laboratório foram geradas/complementadas com IA, revisadas, testadas e validadas por Guilherme Benigno.
