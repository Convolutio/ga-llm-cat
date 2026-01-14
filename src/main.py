# File: src/main.py
# Author: Thomas Horrut
# Version: 0.1
# Description: Application of GA-LLM framework for cat behavior

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import copy
import random
from typing import TYPE_CHECKING, Callable
from ga_llm.base_gene import BaseGene
from ga_llm.config import EvolutionConfig
from ga_llm.engine import HybridEvolutionEngine
from ga_llm.llm_utils import default_llm_call
from ga_llm.evaluation import Evaluator
from ga_llm.constraints import ConstraintValidator
from ga_llm.prompts.template_manager import PromptTemplateManager

# ===================== Travel Gene Class =====================
class TravelGene(BaseGene):
    """Concrete gene representing a structured travel itinerary."""

    def __init__(self, llm_engine: Callable[[str], str]):
        super().__init__(llm_engine)
        self.days = []
        self.hotels = {}
        self.total_cost = 0.0

    def parse_from_text(self, text: str) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            prompt = PROMPTS.render("parse_format", raw_input=text)
            data = json.loads(self.llm_engine(prompt))
        self.days = data.get("days", [])
        self.hotels = data.get("hotels", {})
        self.total_cost = float(data.get("total_cost", 0))

    def to_text(self) -> str:
        return (
            f"Travel Itinerary\n"
            f"- Days: {len(self.days)} day(s)\n"
            f"- Hotel: {self.hotels.get('name', 'N/A')}\n"
            f"- Total Cost: ${self.total_cost:.2f}\n"
            f"Schedule:\n{json.dumps(self.days, indent=2)}"
        )

    def crossover(self, other: BaseGene) -> 'TravelGene':
        if TYPE_CHECKING:
            if not isinstance(other, 'TravelGene'):
                raise NotImplementedError()
        child = TravelGene(llm_engine=self.llm_engine)
        min_days = min(len(self.days), len(other.days))
        if min_days > 0:
            split = random.randint(1, min_days)
            child.days = self.days[:split] + other.days[split:]
        else:
            child.days = self.days or other.days
        child.hotels = random.choice([self.hotels, other.hotels])
        child.total_cost = (self.total_cost + other.total_cost) / 2
        return child

    def mutate(self) -> 'TravelGene':
        mutated = copy.deepcopy(self)
        mutated.total_cost *= random.uniform(0.9, 1.1)
        if mutated.days:
            idx = random.randint(0, len(mutated.days) - 1)
            prompt = PROMPTS.render("mutate_day", 
                budget=mutated.total_cost,
                current_day=json.dumps(mutated.days[idx], indent=2)
            )
            try:
                optimized = json.loads(self.llm_engine(prompt))
                mutated.days[idx] = optimized
            except json.JSONDecodeError:
                pass
        return mutated

# ===================== Prompt Templates =====================
PROMPTS = PromptTemplateManager({
    "task": """
You are an expert travel planner.

Create a comprehensive 4-day itinerary for a trip to Shanghai that satisfies the following:

**Must include:**
- Disneyland
- At least two museums
- Local cultural experiences, dining, and leisure
- Family-friendly activities

**Constraints:**
- Total cost must not exceed ¥5000, including accommodation.
- Reasonable daily travel distances

**Output Format Instructions:**
Return ONLY a single valid JSON object, with this exact structure:

{
  "days": [
    {
      "date": "YYYY-MM-DD",
      "activities": [
        {
          "time": "HH:MM",
          "location": "string",
          "description": "string",
          "estimated_cost": float
        }
      ]
    }
  ],
  "hotels": {
    "name": "string",
    "address": "string",
    "total_cost": float
  },
  "total_cost": float
}

**Important:**
- Do NOT return a list or array as the root.
- Do NOT wrap the JSON in markdown formatting (e.g., ```json).
- Do NOT include any explanation or commentary.
- Ensure the output is fully parsable by `json.loads()`.
""",
    "evaluate": """
Score this travel plan (0-10):

- Budget Compliance (0-4)
- Experience Quality (0-3)
- Practicality (0-3)

Plan:
{{solution_text}}

Return only a score like: [8]
""",
    "parse_format": """
Convert the following travel plan into a structured JSON object.

{raw_input}

**Expected Output Format:**
Return ONLY a single JSON object like this:

{
  "days": [
    {
      "date": "YYYY-MM-DD",
      "activities": [
        {
          "time": "HH:MM",
          "location": "string",
          "description": "string",
          "estimated_cost": float
        }
      ]
    }
  ],
  "hotels": {
    "name": "string",
    "address": "string",
    "total_cost": float
  },
  "total_cost": float
}

**Rules:**
- No markdown formatting
- No extra text
- JSON must parse successfully with json.loads()
""",
    "mutate_day": """
Optimize the following itinerary for one day while staying within a total trip budget of approximately ${budget:.2f}.

{current_day}

**Enhancement Goals:**
- Improve cultural diversity
- Optimize time usage
- Add meaningful local experiences

**Return ONLY the updated JSON for this day**, with this structure:

{{
  "date": "YYYY-MM-DD",
  "activities": [
    {{
      "time": "HH:MM",
      "location": "string",
      "description": "string",
      "estimated_cost": float
    }}
  ]
}}

**Rules:**
- Do NOT wrap in markdown (e.g., ```json)
- Do NOT include commentary or explanation
- Output must be directly parsable by json.loads()
"""
})

# ===================== Optional Constraints =====================
class TravelBudgetValidator(ConstraintValidator):
    """Checks if total cost stays under 5500 CNY."""
    def is_valid(self, gene: BaseGene) -> bool:
        if TYPE_CHECKING:
            if not isinstance(gene, TravelGene):
                raise NotImplementedError()
        return gene.total_cost <= 5500

# ===================== Main =====================
if __name__ == "__main__":
    config = EvolutionConfig(
        population_size=5,
        max_generations=5,
        mutation_rate=0.4,
        early_stopping_rounds=2,
        use_llm_for_crossover=True
    )

    evaluator = Evaluator(PROMPTS.get("evaluate"), default_llm_call)
    
    engine = HybridEvolutionEngine(
        config=config,
        gene_cls=TravelGene,
        task_prompt=PROMPTS.get("task"),
        eval_prompt=PROMPTS.get("evaluate"),
        llm_callback=default_llm_call
    )

    best, score = engine.evolve()
    print("=" * 60)
    print(f"Best Travel Plan Found (Score: {score:.2f}/10):")
    print("=" * 60)
    print(best.to_text())
