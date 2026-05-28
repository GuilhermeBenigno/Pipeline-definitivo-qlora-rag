import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_peak_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def create_massive_rag_context(repetitions=600):
    base = """
Capítulo médico recuperado pelo RAG:
O paciente apresenta sintomas compatíveis com cefaleia pulsátil,
fotofobia, náuseas, sensibilidade a ruídos e piora ao esforço físico.
O quadro pode estar associado a enxaqueca, devendo ser analisado
junto ao histórico clínico, sinais de alerta, medicações em uso e
frequência das crises. Recomenda-se avaliação médica quando houver
alteração neurológica, febre, rigidez de nuca ou pior cefaleia da vida.
"""
    return base * repetitions


def load_model(use_flash_attention=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.float16
    }

    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
        print("Modelo carregado com FlashAttention-2." if use_flash_attention else "Modelo carregado sem FlashAttention.")
    except Exception as e:
        print("FlashAttention-2 não disponível. Carregando com atenção padrão.")
        print("Motivo:", e)
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_report(model, tokenizer, context, use_cache):
    model.config.use_cache = use_cache

    prompt = f"""
Você é um assistente médico corporativo.
Leia o contexto recuperado pelo RAG e gere um resumo clínico objetivo.

CONTEXTO:
{context}

RESUMO CLÍNICO:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id
        )

    end = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "use_cache": use_cache,
        "time_seconds": end - start,
        "peak_vram_mb": get_peak_vram_mb(),
        "generated_text": generated_text[-1200:]
    }


def main():
    print("=== LAB 10 - RAG + QLoRA + KV Cache + FlashAttention ===")

    print("\nCarregando modelo quantizado em 4-bit...")
    model, tokenizer = load_model(use_flash_attention=False)

    vram_after_load = get_vram_mb()
    print(f"VRAM após carregar modelo quantizado: {vram_after_load:.2f} MB")

    print("\nGerando contexto massivo simulado...")
    context = create_massive_rag_context()

    token_count = len(tokenizer.encode(context))
    print(f"Total aproximado de tokens do contexto: {token_count}")

    print("\nExecutando geração SEM KV Cache...")
    result_no_cache = generate_report(
        model=model,
        tokenizer=tokenizer,
        context=context,
        use_cache=False
    )

    print("\nResultado SEM KV Cache:")
    print(f"Tempo: {result_no_cache['time_seconds']:.2f} segundos")
    print(f"Pico VRAM: {result_no_cache['peak_vram_mb']:.2f} MB")

    del model
    torch.cuda.empty_cache()

    print("\nRecarregando modelo com tentativa de FlashAttention-2...")
    model, tokenizer = load_model(use_flash_attention=True)

    print("\nExecutando geração COM KV Cache...")
    result_with_cache = generate_report(
        model=model,
        tokenizer=tokenizer,
        context=context,
        use_cache=True
    )

    print("\nResultado COM KV Cache:")
    print(f"Tempo: {result_with_cache['time_seconds']:.2f} segundos")
    print(f"Pico VRAM: {result_with_cache['peak_vram_mb']:.2f} MB")

    print("\nResumo gerado:")
    print(result_with_cache["generated_text"])

    print("\n=== MÉTRICAS FINAIS ===")
    print(f"VRAM modelo quantizado: {vram_after_load:.2f} MB")
    print(f"Sem KV Cache - Tempo: {result_no_cache['time_seconds']:.2f}s | Pico VRAM: {result_no_cache['peak_vram_mb']:.2f} MB")
    print(f"Com KV Cache - Tempo: {result_with_cache['time_seconds']:.2f}s | Pico VRAM: {result_with_cache['peak_vram_mb']:.2f} MB")


if __name__ == "__main__":
    main()
