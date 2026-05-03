Execute uma análise completa de production readiness do repositório.

IMPORTANTE — regras anti-falso positivo:

1. NÃO reporte problemas sem evidência direta no código
   - Cite arquivo + função + trecho ou padrão observado

2. Para cada problema, inclua:
   - Evidência concreta (arquivo, linha ou comportamento observável)
   - Nível de confiança: (Alto / Médio / Baixo)
   - Impacto real: (Bloqueador / Alto / Médio / Baixo)

3. Se não houver evidência suficiente:
   - Marque como "HIPÓTESE" e NÃO inclua como problema crítico

4. NÃO sugerir melhorias genéricas (ex: "melhorar performance")
   - Toda recomendação deve estar ligada a um problema comprovado

5. NÃO penalizar:
   - código fora de escopo (benchmarks, exemplos)
   - decisões válidas específicas da linguagem

6. Seja conservador na classificação:
   - Prefira subestimar (Beta) do que superestimar (Produção)

7. Para cada nota (0–10), explique:
   - O que justificou a nota
   - O que faltou para ganhar +1 ponto

8. Gere um score de confiança final por linguagem:
   - (%) baseado na qualidade das evidências encontradas

9. Classifique cada achado como:
   - [CONFIRMADO] — evidência clara no código
   - [PROVÁVEL] — forte indício, mas não totalmente verificável
   - [HIPÓTESE] — suposição (não usar para decisão)

10. Gere uma métrica:
   False Positive Risk (0–1)

Baseado em:
- quantidade de evidência direta
- quantidade de hipóteses

Objetivo:
Produzir uma avaliação técnica confiável, auditável e sem alucinação.